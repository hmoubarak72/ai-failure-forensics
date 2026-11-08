"""LLM providers.

``OpenAIClient`` calls the OpenAI Chat Completions API in JSON mode.

``MockLLM`` is a deterministic, offline stand-in so the whole system (pipeline, tracing,
diagnosis, evals, UI, Docker image) runs with no API key and reproducible results. It
sees ONLY the prompt text, exactly like a real model, and it reproduces the failure
modes that real LLMs exhibit on this task:

* invents a missing effective date / signatory name     -> extraction hallucination
* ignores the "preserve currency" instruction             -> prompt failure
* keyword-matches an ambiguous document to the wrong type -> misclassification
* misreads million-scale amounts from the entities JSON   -> propagation error
* keeps only the first few dates/amounts in long docs     -> context loss

Each of those behaviours stops when the prompt contains the corresponding rule from
``prompts.py``, so prompt-version upgrades have measurable effects in regression runs.
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from . import prompts
from .models import LLMCall
from .tracing import record_llm_call


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class LLMClient(ABC):
    model: str = "unknown"

    def complete_json(self, prompt: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        raw, p_tok, c_tok = self._complete(prompt)
        latency = (time.perf_counter() - t0) * 1000
        record_llm_call(
            LLMCall(
                model=self.model,
                prompt=prompt,
                raw_response=raw,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                latency_ms=round(latency, 3),
            )
        )
        return parse_json(raw)

    @abstractmethod
    def _complete(self, prompt: str) -> tuple[str, int, int]:
        ...


def parse_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.S)
    if fence:
        raw = fence.group(1)
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"LLM response was not JSON: {raw[:200]}")
    return json.loads(raw[start : end + 1])


# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
class OpenAIClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        from openai import OpenAI  # imported lazily so the mock path has no hard dep

        self.model = model
        self.temperature = temperature
        self._client = OpenAI()

    def _complete(self, prompt: str) -> tuple[str, int, int]:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
        )
        usage = resp.usage
        return (
            resp.choices[0].message.content or "{}",
            usage.prompt_tokens if usage else estimate_tokens(prompt),
            usage.completion_tokens if usage else 0,
        )


# --------------------------------------------------------------------------- #
# Deterministic mock
# --------------------------------------------------------------------------- #
CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "USD": "USD", "EUR": "EUR", "GBP": "GBP", "CAD": "CAD", "AED": "AED"}
AMOUNT_RE = re.compile(r"(?P<cur>\$|€|£|USD|EUR|GBP|CAD|AED)\s?(?P<num>\d[\d,]*(?:\.\d+)?)")
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_RE = re.compile(rf"\b(?:(?:{MONTHS}) \d{{1,2}}, \d{{4}}|\d{{4}}-\d{{2}}-\d{{2}})\b")
NAME = r"[A-Z][a-z]+(?: [A-Z]\.)? [A-Z][a-z]+(?:-[A-Z][a-z]+)?"
NAME_CUES = [
    rf"between ({NAME})(?:,| of| \()",
    rf"\band ({NAME})(?:,| of| \()",
    rf"Dear (?:Mr\. |Ms\. |Dr\. )?({NAME})",
    rf"(?:Sincerely|Regards|Best regards|Kind regards),\s*\n\s*({NAME})",
    rf"Prepared by:? ({NAME})",
    rf"Bill To:? ({NAME})",
    rf"Signed:? ({NAME})",
    rf"Attention:? ({NAME})",
    rf"Reviewed by:? ({NAME})",
]
KEY_TERMS = [
    "termination", "confidentiality", "indemnification", "renewal", "governing law", "net 30",
    "net 60", "late fee", "payment terms", "deliverables", "revenue", "operating margin", "risk",
    "recommendation", "warranty", "liability", "purchase order", "follow-up", "deadline", "audit",
]
TYPE_KEYWORDS = {
    "contract": ["agreement", "party", "parties", "shall", "term", "terminate", "hereby", "clause"],
    "invoice": ["invoice", "amount due", "bill to", "payment", "total", "due date", "subtotal", "remit"],
    "report": ["report", "findings", "analysis", "quarter", "results", "summary", "metrics", "recommendation"],
    "correspondence": ["dear", "sincerely", "regards", "writing", "letter", "thank you", "please let me know"],
}


def _between(prompt: str, tag: str) -> str:
    m = re.search(rf"<{tag}>\n?(.*?)\n?</{tag}>", prompt, re.S)
    return m.group(1) if m else ""


def fmt_amount(value: float, currency: str) -> str:
    return f"{currency} {value:,.2f}"


class MockLLM(LLMClient):
    model = "mock-llm-1"

    def _complete(self, prompt: str) -> tuple[str, int, int]:
        if "information-extraction engine" in prompt:
            out = self._extract(prompt)
        elif "document classifier" in prompt:
            out = self._classify(prompt)
        elif "structured summaries" in prompt:
            out = self._summarize(prompt)
        elif "impartial reviewer" in prompt:
            out = {"score": 3, "issues": ["mock judge: use heuristic judge"], "category": "none"}
        else:
            out = {"error": "unknown task"}
        raw = json.dumps(out, ensure_ascii=False)
        return raw, estimate_tokens(prompt), estimate_tokens(raw)

    # -- extraction ----------------------------------------------------------
    def _extract(self, prompt: str) -> dict[str, Any]:
        doc = _between(prompt, "document")
        grounded = prompts.RULE_GROUNDING in prompt
        currency_rule = prompts.RULE_CURRENCY in prompt
        lower = doc.lower()
        confidence = 5

        names: list[str] = []
        for pat in NAME_CUES:
            for m in re.finditer(pat, doc):
                if m.group(1) not in names:
                    names.append(m.group(1))

        dates = list(dict.fromkeys(DATE_RE.findall(doc)))

        amounts = []
        first_cur = None
        for m in AMOUNT_RE.finditer(doc):
            cur = CURRENCY_SYMBOLS[m.group("cur")]
            first_cur = first_cur or cur
            value = float(m.group("num").replace(",", ""))
            amounts.append({"value": value, "currency": cur, "raw": m.group(0)})

        distinct = {a["currency"] for a in amounts}
        if len(distinct) > 1 and not currency_rule:
            # Failure mode: model normalises everything to the first currency it saw,
            # ignoring the explicit "preserve the original currency" instruction.
            for a in amounts:
                a["currency"] = first_cur
            confidence = 4

        is_contract = "agreement" in lower
        if not grounded:
            if is_contract and not dates:
                # Failure mode: confidently invents an effective date for a contract.
                dates = ["January 1, 2024"]
                confidence = 4
            if re.search(r"Signed:?\s*_{3,}", doc):
                # Failure mode: fills in a blank signature line with a plausible name.
                names.append("John Smith")
                confidence = 4
        elif is_contract and not dates:
            confidence = 3  # honest uncertainty once grounded

        terms = [t for t in KEY_TERMS if t in lower]
        if len(doc.split()) < 40:
            confidence = min(confidence, 3)
        return {"names": names, "dates": dates, "amounts": amounts, "key_terms": terms, "confidence": confidence}

    # -- classification -----------------------------------------------------
    def _classify(self, prompt: str) -> dict[str, Any]:
        doc = _between(prompt, "document")
        lower = doc.lower()
        scores = {t: sum(lower.count(k) for k in kws) for t, kws in TYPE_KEYWORDS.items()}
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best, second = ranked[0], ranked[1]
        margin = best[1] - second[1]

        if prompts.RULE_PURPOSE in prompt:
            purpose = purpose_type(doc)
            if purpose:
                return {"doc_type": purpose, "rationale": f"Primary purpose indicates {purpose}.", "confidence": 4}

        competing = sum(1 for v in scores.values() if v >= 2)
        confidence = 2 if competing >= 2 else 5 if margin >= 4 else 4 if margin >= 2 else 2
        return {
            "doc_type": best[0],
            "rationale": f"Keyword evidence {dict(scores)}; '{best[0]}' scored highest.",
            "confidence": confidence,
        }

    # -- summarization ------------------------------------------------------
    def _summarize(self, prompt: str) -> dict[str, Any]:
        doc_type = _between(prompt, "doc_type").strip()
        try:
            ent = json.loads(_between(prompt, "entities"))
        except json.JSONDecodeError:
            ent = {}
        copy_rule = prompts.RULE_COPY_AMOUNTS in prompt
        names = ent.get("names", [])
        dates = ent.get("dates", [])
        amounts = ent.get("amounts", [])
        confidence = 5

        def render(a: dict[str, Any]) -> str:
            v = float(a["value"])
            if v >= 1_000_000 and not copy_rule:
                # Failure mode: misreads the entities JSON value as thousands.
                v = v / 1000
            return fmt_amount(v, a["currency"])

        key_amounts = [render(a) for a in amounts]
        key_dates = list(dates)
        if len(amounts) > 3 or len(dates) > 3:
            # Failure mode: long context; the model keeps only the first few facts.
            key_amounts = key_amounts[:3]
            key_dates = key_dates[:3]
            confidence = 4

        headline = {
            "contract": f"Agreement between {' and '.join(names[:2]) or 'unnamed parties'}",
            "invoice": f"Invoice{' for ' + names[0] if names else ''} totalling {key_amounts[-1] if key_amounts else 'an unspecified amount'}",
            "report": f"Report{' by ' + names[0] if names else ''}: key findings",
            "correspondence": f"Letter{' from ' + names[-1] if names else ''} requesting action",
        }.get(doc_type, "Document summary")
        bullets = [f"Covers {t}." for t in ent.get("key_terms", [])[:5]]
        return {
            "headline": headline,
            "parties": names,
            "key_dates": key_dates,
            "key_amounts": key_amounts,
            "bullets": bullets,
            "confidence": confidence,
        }


def purpose_type(doc: str) -> str | None:
    """Primary-purpose heuristic (what a careful reader would do)."""
    head = doc.strip().splitlines()[0].lower() if doc.strip() else ""
    lower = doc.lower()
    if re.match(r"^(dear|hello|hi)\b", head) or re.search(r"\n(sincerely|regards|best regards|kind regards),", lower):
        return "correspondence"
    if "report" in head or "memo" in head or "analysis" in head or "review" in head:
        return "report"
    if "invoice" in head:
        return "invoice"
    if "agreement" in head or "contract" in head:
        return "contract"
    return None


def get_llm(use_openai: bool, model: str = "gpt-4o-mini") -> LLMClient:
    return OpenAIClient(model=model) if use_openai else MockLLM()
