"""Step-level judges: score whether each step's output is a reasonable transformation of its input.

* ``HeuristicJudge`` - deterministic, offline checks (grounding, instruction compliance,
  fidelity to upstream outputs, golden-label agreement). Fast and reproducible.
* ``LLMJudge`` - LLM-as-judge using ``prompts.JUDGE_PROMPT``; combined with the heuristic
  evidence (the lower score wins) so deterministic grounding facts are never overridden.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from . import prompts
from .llm import AMOUNT_RE, CURRENCY_SYMBOLS, LLMClient, fmt_amount, purpose_type
from .models import FailureCategory, Span, StepJudgement, Trace


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def source_text(trace: Trace) -> str:
    sp = trace.span("intake")
    if sp and sp.output:
        return sp.output.get("clean_text", "")
    if sp:
        return sp.input.get("text", "")
    return ""


def amount_strings(amounts: list[dict[str, Any]]) -> list[str]:
    return [fmt_amount(float(a["value"]), a["currency"]) for a in amounts]


class HeuristicJudge:
    name = "heuristic"

    def judge(self, trace: Trace, span: Span, golden: Optional[dict[str, Any]] = None) -> StepJudgement:
        fn = getattr(self, f"_judge_{span.step_name}", None)
        if span.error:
            return StepJudgement(
                step_name=span.step_name,
                score=1,
                issues=[f"Step raised an error: {span.error}"],
                suspected_category=FailureCategory.pipeline_error,
                evidence={"error": span.error},
                confidence=span.confidence,
            )
        if fn is None or span.output is None:
            return StepJudgement(step_name=span.step_name, score=5, confidence=span.confidence)
        j: StepJudgement = fn(trace, span, golden or {})
        j.confidence = span.confidence
        return j

    # -- intake -------------------------------------------------------------
    def _judge_intake(self, trace: Trace, span: Span, golden: dict) -> StepJudgement:
        raw = _norm(span.input.get("text", ""))
        out = _norm(span.output.get("clean_text", ""))  # type: ignore[union-attr]
        ratio = len(out) / max(1, len(raw))
        if ratio < 0.8:
            return StepJudgement(
                step_name="intake", score=2, issues=[f"Intake dropped {1 - ratio:.0%} of the text"],
                suspected_category=FailureCategory.context_loss, evidence={"ratio": ratio},
            )
        return StepJudgement(step_name="intake", score=5)

    # -- extraction ---------------------------------------------------------
    def _judge_extraction(self, trace: Trace, span: Span, golden: dict) -> StepJudgement:
        src = source_text(trace)
        nsrc = _norm(src)
        ent = span.output["entities"]  # type: ignore[index]
        issues: list[str] = []
        hallucinated: list[str] = []
        for n in ent.get("names", []):
            if _norm(n) not in nsrc:
                hallucinated.append(n)
        for d in ent.get("dates", []):
            if _norm(d) not in nsrc:
                hallucinated.append(d)
        for a in ent.get("amounts", []):
            if a.get("raw") and _norm(a["raw"]) not in nsrc:
                hallucinated.append(a["raw"])
        currency_errors: list[str] = []
        for a in ent.get("amounts", []):
            m = AMOUNT_RE.search(a.get("raw", ""))
            if m and CURRENCY_SYMBOLS[m.group("cur")] != a.get("currency"):
                currency_errors.append(f"{a['raw']} labelled {a['currency']} (source says {CURRENCY_SYMBOLS[m.group('cur')]})")

        evidence: dict[str, Any] = {}
        if hallucinated:
            issues.append("Entities not present in the source document: " + ", ".join(repr(h) for h in hallucinated))
            evidence["hallucinated"] = hallucinated
            return StepJudgement(step_name="extraction", score=1, issues=issues,
                                 suspected_category=FailureCategory.extraction_hallucination, evidence=evidence)
        if currency_errors:
            issues.append("Ignored the instruction to preserve the original currency of each amount: " + "; ".join(currency_errors))
            evidence["currency_errors"] = currency_errors
            return StepJudgement(step_name="extraction", score=2, issues=issues,
                                 suspected_category=FailureCategory.prompt_failure, evidence=evidence)
        g_ent = golden.get("entities")
        if g_ent:
            missing = [x for x in g_ent.get("names", []) + g_ent.get("dates", []) if _norm(x) not in {_norm(y) for y in ent.get("names", []) + ent.get("dates", [])}]
            if missing:
                return StepJudgement(step_name="extraction", score=3, issues=[f"Missed entities: {missing}"],
                                     suspected_category=FailureCategory.context_loss, evidence={"missed": missing})
        return StepJudgement(step_name="extraction", score=5)

    # -- classification -----------------------------------------------------
    def _judge_classification(self, trace: Trace, span: Span, golden: dict) -> StepJudgement:
        predicted = span.output["doc_type"]  # type: ignore[index]
        expected = golden.get("doc_type") or purpose_type(source_text(trace))
        if expected and expected != predicted:
            return StepJudgement(
                step_name="classification", score=1,
                issues=[f"Classified as '{predicted}' but the document's primary purpose is '{expected}'"],
                suspected_category=FailureCategory.misclassification,
                evidence={"predicted": predicted, "expected": expected,
                          "reference": "golden dataset" if golden.get("doc_type") else "purpose heuristic"},
            )
        if span.confidence is not None and span.confidence <= 2:
            return StepJudgement(step_name="classification", score=3,
                                 issues=["Correct type, but with low confidence (ambiguous document)"])
        return StepJudgement(step_name="classification", score=5)

    # -- summarization ------------------------------------------------------
    def _judge_summarization(self, trace: Trace, span: Span, golden: dict) -> StepJudgement:
        summ = span.output["summary"]  # type: ignore[index]
        inp = span.input
        ent = (inp.get("extraction") or {}).get("entities", {})
        cls = (inp.get("classification") or {}).get("doc_type")
        expected_amounts = amount_strings(ent.get("amounts", []))
        issues: list[str] = []
        evidence: dict[str, Any] = {}

        misread = [a for a in summ.get("key_amounts", []) if a not in expected_amounts]
        if misread:
            issues.append(
                f"Summary amounts {misread} do not match the amounts it received from extraction {expected_amounts}"
            )
            evidence["misread_amounts"] = misread
            evidence["upstream_amounts"] = expected_amounts
            return StepJudgement(step_name="summarization", score=2, issues=issues,
                                 suspected_category=FailureCategory.propagation_error, evidence=evidence)
        if summ.get("doc_type") != cls:
            return StepJudgement(step_name="summarization", score=2,
                                 issues=[f"Summary type {summ.get('doc_type')} != classified type {cls}"],
                                 suspected_category=FailureCategory.propagation_error)

        dropped_amounts = [a for a in expected_amounts if a not in summ.get("key_amounts", [])]
        dropped_dates = [d for d in ent.get("dates", []) if d not in summ.get("key_dates", [])]
        if dropped_amounts or dropped_dates:
            issues.append(f"Dropped information received from upstream steps: dates={dropped_dates}, amounts={dropped_amounts}")
            evidence["dropped"] = dropped_dates + dropped_amounts
            return StepJudgement(step_name="summarization", score=2, issues=issues,
                                 suspected_category=FailureCategory.context_loss, evidence=evidence)
        return StepJudgement(step_name="summarization", score=5)


class LLMJudge:
    """LLM-as-judge, hybridised with heuristic evidence (lower score wins)."""

    name = "llm"

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self.heuristic = HeuristicJudge()

    def judge(self, trace: Trace, span: Span, golden: Optional[dict[str, Any]] = None) -> StepJudgement:
        h = self.heuristic.judge(trace, span, golden)
        if span.error or span.output is None:
            return h
        reference = f"Reference (golden) output for this document:\n{json.dumps(golden)}\n" if golden else ""
        prompt = prompts.JUDGE_PROMPT.format(
            step_name=span.step_name,
            prompt=span.llm_calls[0].prompt[:3000] if span.llm_calls else "",
            source=source_text(trace)[:6000],
            step_input=json.dumps(span.input, default=str)[:6000],
            step_output=json.dumps(span.output, default=str)[:4000],
            reference=reference,
        )
        try:
            data = self.llm.complete_json(prompt)
            score = max(1, min(5, int(data.get("score", 5))))
            cat_raw = str(data.get("category", "none"))
            cat = FailureCategory(cat_raw) if cat_raw in FailureCategory._value2member_map_ else FailureCategory.none
            issues = [str(i) for i in data.get("issues", [])]
        except Exception as exc:  # judge failures must never break diagnosis
            h.issues.append(f"LLM judge unavailable ({type(exc).__name__}); heuristic used")
            return h
        if h.score <= score:
            h.issues += [f"[llm-judge] {i}" for i in issues]
            return h
        return StepJudgement(step_name=span.step_name, score=score, issues=issues, suspected_category=cat,
                             evidence=h.evidence, confidence=span.confidence)
