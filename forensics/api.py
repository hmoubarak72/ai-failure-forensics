"""REST API: run the pipeline, explore traces, flag bad outputs, confirm diagnoses, run evals."""
from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from . import __version__, prompts
from .analytics import compute_analytics
from .feedback import NotFound, expected_output
from .models import ConfirmRequest, Diagnosis, EvalCase, EvalRun, FlagRequest, RawDocument, Trace
from .services import Services, build_services

app = FastAPI(
    title="Failure Forensics API",
    version=__version__,
    description="Observability, backward root-cause analysis and a feedback-to-eval loop for multi-step AI pipelines.",
)
_services: Optional[Services] = None


def svc() -> Services:
    global _services
    if _services is None:
        _services = build_services()
    return _services


def set_services(s: Services) -> None:  # used by tests
    global _services
    _services = s


class RunRequest(BaseModel):
    document: RawDocument
    pipeline_version: Optional[str] = None


class BatchRequest(BaseModel):
    documents: list[RawDocument]
    pipeline_version: Optional[str] = None


def _summary_row(tr: Trace) -> dict[str, Any]:
    return {"trace_id": tr.trace_id, "doc_id": tr.doc_id, "status": tr.status.value,
            "final_score": tr.final_score, "notes": tr.validation_notes}


@app.get("/health")
def health() -> dict[str, Any]:
    s = svc().settings
    return {"status": "ok", "version": __version__, "llm": svc().llm.model, "judge": svc().analyzer.judge.name,
            "default_pipeline_version": s.pipeline_version, "pipeline_versions": list(prompts.VERSIONS)}


@app.post("/pipeline/run")
def run_pipeline(req: RunRequest) -> dict[str, Any]:
    try:
        tr = svc().pipeline(req.pipeline_version).run(req.document)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {**_summary_row(tr), "final_output": tr.final_output}


@app.post("/pipeline/batch")
def run_batch(req: BatchRequest) -> list[dict[str, Any]]:
    p = svc().pipeline(req.pipeline_version)
    return [_summary_row(p.run(d)) for d in req.documents]


@app.get("/traces")
def list_traces(status: Optional[str] = None, flagged: Optional[bool] = None, limit: int = Query(500, le=5000)) -> list[dict[str, Any]]:
    return svc().store.list_traces(status=status, flagged=flagged, limit=limit)


@app.get("/traces/{trace_id}", response_model=Trace)
def get_trace(trace_id: str) -> Trace:
    tr = svc().store.get_trace(trace_id)
    if tr is None:
        raise HTTPException(404, "trace not found")
    return tr


@app.post("/traces/{trace_id}/flag", response_model=Diagnosis)
def flag_trace(trace_id: str, req: FlagRequest) -> Diagnosis:
    try:
        return svc().feedback.flag(trace_id, req)
    except NotFound as exc:
        raise HTTPException(404, str(exc))


@app.get("/traces/{trace_id}/diagnosis", response_model=Optional[Diagnosis])
def get_diagnosis(trace_id: str) -> Optional[Diagnosis]:
    return svc().store.latest_diagnosis(trace_id)


@app.post("/traces/{trace_id}/analyze", response_model=Diagnosis)
def analyze_trace(trace_id: str) -> Diagnosis:
    """Dry-run diagnosis without flagging (does not persist)."""
    tr = svc().store.get_trace(trace_id)
    if tr is None:
        raise HTTPException(404, "trace not found")
    return svc().analyzer.analyze(tr)


@app.post("/traces/{trace_id}/confirm", response_model=EvalCase)
def confirm_diagnosis(trace_id: str, req: ConfirmRequest) -> EvalCase:
    try:
        return svc().feedback.confirm(trace_id, req)
    except NotFound as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@app.get("/traces/{trace_id}/diff/{step_name}")
def step_diff(trace_id: str, step_name: str) -> dict[str, Any]:
    """What the step received vs. produced vs. should have produced."""
    s = svc()
    tr = s.store.get_trace(trace_id)
    if tr is None:
        raise HTTPException(404, "trace not found")
    span = tr.span(step_name)
    if span is None:
        raise HTTPException(404, "step not in trace")
    expected, source = None, None
    for case in s.store.list_eval_cases():  # a human correction beats the golden label
        if case.source_trace_id == trace_id and case.failing_step == step_name:
            expected, source = case.corrected_output, "human correction"
    if expected is None:
        expected = expected_output(step_name, s.golden.get(tr.doc_id, {}))
        source = "golden dataset" if expected is not None else None
    return {"step_name": step_name, "received": span.input, "produced": span.output,
            "expected": expected, "expected_source": source, "error": span.error}


@app.get("/evals/cases", response_model=list[EvalCase])
def list_cases() -> list[EvalCase]:
    return svc().store.list_eval_cases()


@app.post("/evals/run", response_model=EvalRun)
def run_evals(pipeline_version: Optional[str] = None) -> EvalRun:
    version = pipeline_version or svc().settings.pipeline_version
    if version not in prompts.VERSIONS:
        raise HTTPException(400, f"unknown version {version}")
    return svc().feedback.run_regression(version)


@app.get("/evals/runs", response_model=list[EvalRun])
def list_runs() -> list[EvalRun]:
    return svc().store.list_eval_runs()


@app.get("/analytics")
def analytics(granularity: str = Query("hour", pattern="^(hour|day)$")) -> dict[str, Any]:
    return compute_analytics(svc().store, granularity)
