"""Backward trace analyzer: root cause, failure category and evidence chain."""
from __future__ import annotations

import time
import uuid
from typing import Any, Optional

from .judge import HeuristicJudge, LLMJudge, _norm
from .models import Diagnosis, EvidenceItem, FailureCategory, StepJudgement, Trace

STEP_LABEL = {"intake": "Intake", "extraction": "Extraction", "classification": "Classification", "summarization": "Summarization"}
SIGNIFICANT_DROP = 2  # judge score at or below this = significant quality drop


def _label(trace: Trace, step: str) -> str:
    sp = trace.span(step)
    idx = (sp.step_index + 1) if sp else "?"
    return f"Step {idx} ({STEP_LABEL.get(step, step)})"


def _short(v: Any, n: int = 300) -> str:
    s = v if isinstance(v, str) else __import__("json").dumps(v, default=str)
    return s if len(s) <= n else s[:n] + "..."


class BackwardTraceAnalyzer:
    def __init__(self, judge: Optional[HeuristicJudge | LLMJudge] = None, golden: Optional[dict[str, dict]] = None):
        self.judge = judge or HeuristicJudge()
        self.golden = golden if golden is not None else {}

    def analyze(self, trace: Trace) -> Diagnosis:
        t0 = time.perf_counter()
        golden = self.golden.get(trace.doc_id, {})

        # Walk BACKWARD from the final step, judging each transformation against its input.
        judgements: dict[str, StepJudgement] = {}
        for span in reversed(trace.spans):
            judgements[span.step_name] = self.judge.judge(trace, span, golden)
        ordered = [judgements[s.step_name] for s in trace.spans]

        bad = [j for j in ordered if j.score <= SIGNIFICANT_DROP]
        if bad:
            root = bad[0]  # earliest step with a significant quality drop
            category = root.suspected_category
        else:
            # No step judged clearly bad: low-confidence spans are the primary suspects.
            with_conf = [j for j in ordered if j.confidence is not None]
            root = min(with_conf, key=lambda j: (j.confidence, j.score)) if with_conf else None
            category = FailureCategory.none

        downstream, chain = self._evidence_chain(trace, root, category, ordered)
        explanation = self._explain(trace, root, category, downstream, ordered)

        return Diagnosis(
            diagnosis_id=uuid.uuid4().hex[:12],
            trace_id=trace.trace_id,
            root_cause_step=root.step_name if root else None,
            category=category,
            explanation=explanation,
            evidence_chain=chain,
            judgements=ordered,
            affected_downstream=downstream,
            analysis_ms=round((time.perf_counter() - t0) * 1000, 3),
            judge=self.judge.name,
        )

    # ------------------------------------------------------------------ #
    def _evidence_chain(
        self, trace: Trace, root: Optional[StepJudgement], category: FailureCategory, ordered: list[StepJudgement]
    ) -> tuple[list[str], list[EvidenceItem]]:
        if root is None:
            return [], []
        sp = trace.span(root.step_name)
        chain = [
            EvidenceItem(
                step_name=root.step_name,
                claim="; ".join(root.issues) or "Lowest-confidence step (no significant quality drop detected)",
                input_excerpt=_short(sp.input if sp else ""),
                output_excerpt=_short(sp.output if sp else ""),
            )
        ]
        downstream: list[str] = []
        markers: list[str] = []
        ev = root.evidence
        markers += [str(x) for x in ev.get("hallucinated", [])]
        if category == FailureCategory.misclassification:
            markers.append(str(ev.get("predicted")))
        if category == FailureCategory.prompt_failure:
            markers += [s.split(" labelled ")[0] for s in ev.get("currency_errors", [])]

        root_idx = sp.step_index if sp else -1
        for later in trace.spans:
            if later.step_index <= root_idx or later.output is None:
                continue
            out_text = _norm(_short(later.output, 100000))
            hits = [m for m in markers if m and _norm(m) in out_text]
            if category == FailureCategory.prompt_failure and later.step_name == "summarization":
                # wrong currency labels propagate as wrong currency codes in key_amounts
                hits = hits or ["currency labels"]
            if hits:
                downstream.append(later.step_name)
                chain.append(
                    EvidenceItem(
                        step_name=later.step_name,
                        claim=f"Carried the upstream error forward: {', '.join(repr(h) for h in hits)}",
                        input_excerpt=_short(later.input),
                        output_excerpt=_short(later.output),
                    )
                )
        return downstream, chain

    def _explain(
        self,
        trace: Trace,
        root: Optional[StepJudgement],
        category: FailureCategory,
        downstream: list[str],
        ordered: list[StepJudgement],
    ) -> str:
        if root is None:
            return "No spans available to analyse."
        r = _label(trace, root.step_name)
        ev = root.evidence
        if category == FailureCategory.pipeline_error:
            text = f"{r} raised a hard error and aborted the pipeline: {ev.get('error')}."
        elif category == FailureCategory.extraction_hallucination:
            ents = ", ".join(f"'{h}'" for h in ev.get("hallucinated", []))
            text = f"{r} hallucinated the entit{'ies' if len(ev.get('hallucinated', [])) > 1 else 'y'} {ents} which do{'' if len(ev.get('hallucinated', [])) > 1 else 'es'} not appear in the source document."
        elif category == FailureCategory.prompt_failure:
            text = f"{r} ignored an explicit instruction (preserve the original currency of each amount): {'; '.join(ev.get('currency_errors', []))}."
        elif category == FailureCategory.misclassification:
            text = f"{r} classified the document as '{ev.get('predicted')}' but it is a '{ev.get('expected')}' (per {ev.get('reference')})."
        elif category == FailureCategory.propagation_error:
            text = (f"Upstream steps were correct, but {r} misinterpreted its input: it reported {ev.get('misread_amounts')} "
                    f"while extraction supplied {ev.get('upstream_amounts')}.")
        elif category == FailureCategory.context_loss:
            text = f"{r} dropped information that earlier steps had captured: {ev.get('dropped', ev.get('missed'))}."
        else:
            text = (f"No step showed a significant quality drop. {r} has the lowest self-reported confidence "
                    f"({root.confidence}) and is the primary suspect.")
        if downstream:
            text += " This propagated to " + " and ".join(_label(trace, d) for d in downstream) + ", which included it in its output."
        others = [j for j in ordered if j.score <= SIGNIFICANT_DROP and j.step_name != root.step_name and j.step_name not in downstream]
        if others:
            text += " Secondary issues: " + "; ".join(f"{_label(trace, j.step_name)}: {j.issues[0]}" for j in others if j.issues) + "."
        return text


def build_analyzer(settings: Any, llm: Any, golden: Optional[dict[str, dict]] = None) -> BackwardTraceAnalyzer:
    mode = settings.judge_provider
    use_llm = mode == "llm" or (mode == "auto" and settings.use_openai)
    judge = LLMJudge(llm) if use_llm else HeuristicJudge()
    return BackwardTraceAnalyzer(judge=judge, golden=golden)
