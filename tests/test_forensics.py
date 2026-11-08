from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from forensics import api as api_module
from forensics.analyzer import BackwardTraceAnalyzer
from forensics.config import Settings
from forensics.corpus import build_corpus, write_corpus
from forensics.llm import MockLLM
from forensics.models import FailureCategory, RawDocument, TraceStatus
from forensics.pipeline import STEPS, Pipeline
from forensics.services import build_services

DOCS, GOLDEN = build_corpus()


@pytest.fixture()
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("JUDGE_PROVIDER", "heuristic")
    monkeypatch.setenv("FORENSICS_DATA_DIR", str(tmp_path))
    s = Settings()
    s.ensure_dirs()
    write_corpus(s.data_dir)
    return s


def _doc(kind: str) -> RawDocument:
    return next(d for d in DOCS if GOLDEN[d.doc_id]["expected_failure"] == kind)


def test_corpus_shape():
    assert len(DOCS) == 50
    failures = [g for g in GOLDEN.values() if g["expected_failure"] != "none"]
    assert 8 <= len(failures) <= 15
    assert {g["expected_failure"] for g in failures} >= {
        "extraction_hallucination", "misclassification", "propagation_error", "prompt_failure", "context_loss"}


def test_trace_has_one_span_per_step_with_llm_calls(settings):
    tr = Pipeline(settings=settings, llm=MockLLM(), version="v1").run(_doc("none"), persist=False)
    assert [s.step_name for s in tr.spans] == list(STEPS)
    assert len(tr.trace_id) == 32  # OpenTelemetry trace id
    for s in tr.spans[1:]:
        assert s.llm_calls and s.llm_calls[0].prompt and s.llm_calls[0].raw_response
        assert 1 <= s.confidence <= 5 and s.latency_ms >= 0 and s.total_tokens > 0
    assert tr.status in (TraceStatus.success, TraceStatus.degraded)


def test_hard_error_marks_trace_failed(settings):
    tr = Pipeline(settings=settings, llm=MockLLM()).run(_doc("pipeline_error"), persist=False)
    assert tr.status == TraceStatus.failure
    assert tr.spans[0].error and len(tr.spans) == 1


@pytest.mark.parametrize("kind,step", [
    ("extraction_hallucination", "extraction"),
    ("prompt_failure", "extraction"),
    ("misclassification", "classification"),
    ("propagation_error", "summarization"),
    ("context_loss", "summarization"),
    ("pipeline_error", "intake"),
])
def test_root_cause_and_category(settings, kind, step):
    tr = Pipeline(settings=settings, llm=MockLLM(), version="v1").run(_doc(kind), persist=False)
    d = BackwardTraceAnalyzer(golden=GOLDEN).analyze(tr)
    assert d.root_cause_step == step
    assert d.category == FailureCategory(kind)
    assert d.evidence_chain and d.explanation


def test_hallucination_evidence_chain_reaches_summary(settings):
    doc = next(d for d in DOCS if "Signed: ____" in d.text)
    tr = Pipeline(settings=settings, llm=MockLLM(), version="v1").run(doc, persist=False)
    d = BackwardTraceAnalyzer(golden=GOLDEN).analyze(tr)
    assert "John Smith" in d.explanation
    assert d.affected_downstream == ["summarization"]


def test_clean_documents_have_no_root_cause(settings):
    an = BackwardTraceAnalyzer(golden=GOLDEN)
    p = Pipeline(settings=settings, llm=MockLLM(), version="v1")
    for doc in [d for d in DOCS if GOLDEN[d.doc_id]["expected_failure"] == "none"]:
        assert an.analyze(p.run(doc, persist=False)).category == FailureCategory.none, doc.doc_id


def test_prompt_versions_fix_failures(settings):
    an = BackwardTraceAnalyzer(golden=GOLDEN)
    for version, still_failing in [("v2", {"misclassification", "propagation_error", "context_loss"}),
                                   ("v3", {"context_loss"})]:
        p = Pipeline(settings=settings, llm=MockLLM(), version=version)
        cats = {an.analyze(p.run(d, persist=False)).category.value
                for d in DOCS if GOLDEN[d.doc_id]["expected_failure"] not in ("none", "pipeline_error")}
        assert cats - {"none"} == still_failing


def test_api_flag_confirm_eval_loop(settings):
    services = build_services(settings)
    api_module.set_services(services)
    client = TestClient(api_module.app)
    assert client.get("/health").json()["status"] == "ok"

    doc = _doc("misclassification")
    res = client.post("/pipeline/run", json={"document": doc.model_dump(), "pipeline_version": "v1"}).json()
    tid = res["trace_id"]
    assert client.get(f"/traces/{tid}").status_code == 200

    diag = client.post(f"/traces/{tid}/flag", json={"reason": "wrong type"}).json()
    assert diag["root_cause_step"] == "classification" and diag["category"] == "misclassification"

    diff = client.get(f"/traces/{tid}/diff/classification").json()
    assert diff["expected"] == {"doc_type": "correspondence"} and diff["produced"]["doc_type"] == "invoice"

    case = client.post(f"/traces/{tid}/confirm", json={"reviewer": "test"}).json()
    assert case["failing_step"] == "classification"
    assert services.settings.eval_dataset_path.read_text().count("\n") == 1

    assert client.post("/evals/run", params={"pipeline_version": "v1"}).json()["resolved"] == 0
    assert client.post("/evals/run", params={"pipeline_version": "v3"}).json()["resolved"] == 1

    a = client.get("/analytics").json()
    assert a["failure_types"] == {"misclassification": 1}
    assert [r["resolved"] for r in a["regression_trend"]] == [0, 1]
    assert (settings.traces_dir / f"{tid}.json").exists()


def test_override_diagnosis(settings):
    services = build_services(settings)
    api_module.set_services(services)
    client = TestClient(api_module.app)
    tid = client.post("/pipeline/run", json={"document": _doc("context_loss").model_dump()}).json()["trace_id"]
    client.post(f"/traces/{tid}/flag", json={})
    case = client.post(f"/traces/{tid}/confirm", json={"category": "prompt_failure", "corrected_output": {"summary": {"key_dates": []}}}).json()
    assert case["category"] == "prompt_failure"
    assert client.get(f"/traces/{tid}/diagnosis").json()["status"] == "overridden"
