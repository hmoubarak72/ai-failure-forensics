"""Feedback-to-eval loop: flag -> diagnose -> confirm/override -> eval case -> regression runs."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from .analyzer import BackwardTraceAnalyzer
from .judge import _norm
from .llm import fmt_amount
from .models import (
    ConfirmRequest,
    Diagnosis,
    EvalCase,
    EvalCaseResult,
    EvalRun,
    FailureCategory,
    FlagRequest,
    Trace,
)
from .pipeline import Pipeline
from .store import TraceStore


class NotFound(Exception):
    pass


def expected_output(step: str, golden: dict[str, Any]) -> Optional[dict[str, Any]]:
    """What a step *should* have produced, derived from the golden dataset."""
    if not golden or golden.get("doc_type") is None:
        return None
    if step == "extraction":
        e = golden["entities"]
        return {"entities": {"names": e.get("names", []), "dates": e.get("dates", []), "amounts": e.get("amounts", [])}}
    if step == "classification":
        return {"doc_type": golden["doc_type"]}
    if step == "summarization":
        return {"summary": {"doc_type": golden["doc_type"], **golden.get("summary", {})}}
    return None


def compare_step(step: str, actual: Optional[dict[str, Any]], corrected: dict[str, Any]) -> tuple[bool, str]:
    """Does the step's current output match the human-corrected output (on the fields given)?"""
    if actual is None:
        return False, "step produced no output"
    if step == "classification":
        ok = actual.get("doc_type") == corrected.get("doc_type")
        return ok, f"got {actual.get('doc_type')}, expected {corrected.get('doc_type')}"
    if step == "extraction":
        a, c = actual.get("entities", {}), corrected.get("entities", {})
        diffs = []
        for key in ("names", "dates"):
            if key in c and {_norm(x) for x in a.get(key, [])} != {_norm(x) for x in c[key]}:
                diffs.append(f"{key}: got {a.get(key)}, expected {c[key]}")
        if "amounts" in c:
            got = {fmt_amount(float(x["value"]), x["currency"]) for x in a.get("amounts", [])}
            exp = {fmt_amount(float(x["value"]), x["currency"]) for x in c["amounts"]}
            if got != exp:
                diffs.append(f"amounts: got {sorted(got)}, expected {sorted(exp)}")
        return (not diffs), "; ".join(diffs) or "matches correction"
    if step == "summarization":
        a, c = actual.get("summary", {}), corrected.get("summary", {})
        diffs = [f"{k}: got {a.get(k)}, expected {c[k]}" for k in ("key_dates", "key_amounts", "doc_type")
                 if k in c and (set(a.get(k, [])) != set(c[k]) if isinstance(c[k], list) else a.get(k) != c[k])]
        return (not diffs), "; ".join(diffs) or "matches correction"
    return True, "no comparator; step ran"


class FeedbackService:
    def __init__(
        self,
        store: TraceStore,
        analyzer: BackwardTraceAnalyzer,
        pipeline_factory: Callable[[str], Pipeline],
        golden: Optional[dict[str, dict]] = None,
    ):
        self.store = store
        self.analyzer = analyzer
        self.pipeline_factory = pipeline_factory
        self.golden = golden if golden is not None else {}

    def _trace(self, trace_id: str) -> Trace:
        tr = self.store.get_trace(trace_id)
        if tr is None:
            raise NotFound(f"trace {trace_id} not found")
        return tr

    # -- flag ----------------------------------------------------------------
    def flag(self, trace_id: str, req: FlagRequest) -> Diagnosis:
        tr = self._trace(trace_id)
        self.store.flag_trace(trace_id, req.reason, req.flagged_by)
        diag = self.analyzer.analyze(tr)
        self.store.save_diagnosis(diag)
        return diag

    # -- confirm / override ---------------------------------------------------
    def confirm(self, trace_id: str, req: ConfirmRequest) -> EvalCase:
        tr = self._trace(trace_id)
        diag = self.store.latest_diagnosis(trace_id) or self.flag(trace_id, FlagRequest(reason="auto"))
        step = req.root_cause_step or diag.root_cause_step
        category = req.category or diag.category
        if step is None:
            raise ValueError("No root-cause step proposed; provide root_cause_step")
        overridden = (req.root_cause_step not in (None, diag.root_cause_step)) or (req.category not in (None, diag.category))
        diag.status = "overridden" if overridden else "confirmed"
        diag.confirmed_step, diag.confirmed_category = step, category
        self.store.save_diagnosis(diag)

        span = tr.span(step)
        corrected = req.corrected_output or expected_output(step, self.golden.get(tr.doc_id, {})) or (span.output if span else {}) or {}
        doc = self.store.get_document(tr.doc_id)
        if doc is None:
            raise NotFound(f"source document {tr.doc_id} not stored")
        case = EvalCase(
            case_id=uuid.uuid4().hex[:12],
            source_trace_id=trace_id,
            doc=doc,
            failing_step=step,
            category=category if isinstance(category, FailureCategory) else FailureCategory(category),
            bad_output=(span.output if span else None) or {"error": span.error if span else "missing span"},
            corrected_output=corrected,
            created_by=req.reviewer,
        )
        self.store.add_eval_case(case)
        return case

    # -- regression -----------------------------------------------------------
    def run_regression(self, version: str, created_at: Optional[datetime] = None) -> EvalRun:
        pipeline = self.pipeline_factory(version)
        results = []
        for case in self.store.list_eval_cases():
            tr = pipeline.run(case.doc, persist=False)
            span = tr.span(case.failing_step)
            passed, detail = compare_step(case.failing_step, span.output if span else None, case.corrected_output)
            results.append(EvalCaseResult(case_id=case.case_id, passed=passed, detail=detail))
        resolved = sum(r.passed for r in results)
        run = EvalRun(
            run_id=uuid.uuid4().hex[:12],
            pipeline_version=version,
            created_at=created_at or datetime.now(timezone.utc),
            total=len(results),
            resolved=resolved,
            still_failing=len(results) - resolved,
            results=results,
        )
        self.store.save_eval_run(run)
        return run
