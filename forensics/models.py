"""Typed data contracts for every pipeline stage, trace, diagnosis and eval case.

Every intermediate structure is a Pydantic model. Clean step boundaries are what
make the tracing meaningful: each span records exactly one typed input and one
typed output, so the analyzer can judge every transformation in isolation.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Pipeline I/O
# --------------------------------------------------------------------------- #
class DocType(str, Enum):
    contract = "contract"
    invoice = "invoice"
    report = "report"
    correspondence = "correspondence"


class RawDocument(BaseModel):
    doc_id: str
    title: str = ""
    text: str
    source_format: str = "text"  # text | markdown | pdf_text


class IntakeOutput(BaseModel):
    doc_id: str
    title: str
    clean_text: str
    char_count: int
    word_count: int
    source_format: str


class Amount(BaseModel):
    value: float
    currency: str
    raw: str = ""


class Entities(BaseModel):
    names: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    amounts: list[Amount] = Field(default_factory=list)
    key_terms: list[str] = Field(default_factory=list)


class ExtractionOutput(BaseModel):
    doc_id: str
    entities: Entities
    confidence: int = Field(ge=1, le=5)


class ClassificationOutput(BaseModel):
    doc_id: str
    doc_type: DocType
    rationale: str = ""
    confidence: int = Field(ge=1, le=5)


class Summary(BaseModel):
    doc_type: DocType
    headline: str
    parties: list[str] = Field(default_factory=list)
    key_dates: list[str] = Field(default_factory=list)
    key_amounts: list[str] = Field(default_factory=list)
    bullets: list[str] = Field(default_factory=list)


class SummarizationOutput(BaseModel):
    doc_id: str
    summary: Summary
    confidence: int = Field(ge=1, le=5)


class StepInput(BaseModel):
    """Generic wrapper used when a step consumes several upstream outputs."""

    intake: IntakeOutput
    extraction: Optional[ExtractionOutput] = None
    classification: Optional[ClassificationOutput] = None


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #
class TraceStatus(str, Enum):
    success = "success"
    degraded = "degraded"
    failure = "failure"


class SpanStatus(str, Enum):
    ok = "ok"
    error = "error"


class LLMCall(BaseModel):
    model: str
    prompt: str
    raw_response: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0


class Span(BaseModel):
    span_id: str
    trace_id: str
    step_index: int
    step_name: str
    started_at: datetime = Field(default_factory=utcnow)
    latency_ms: float = 0.0
    input: dict[str, Any] = Field(default_factory=dict)
    output: Optional[dict[str, Any]] = None
    llm_calls: list[LLMCall] = Field(default_factory=list)
    confidence: Optional[int] = None
    status: SpanStatus = SpanStatus.ok
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return sum(c.prompt_tokens + c.completion_tokens for c in self.llm_calls)


class Trace(BaseModel):
    trace_id: str
    doc_id: str
    pipeline_version: str
    created_at: datetime = Field(default_factory=utcnow)
    spans: list[Span] = Field(default_factory=list)
    final_output: Optional[dict[str, Any]] = None
    status: TraceStatus = TraceStatus.success
    final_score: float = 1.0
    validation_notes: list[str] = Field(default_factory=list)

    def span(self, step_name: str) -> Optional[Span]:
        return next((s for s in self.spans if s.step_name == step_name), None)


# --------------------------------------------------------------------------- #
# Diagnosis
# --------------------------------------------------------------------------- #
class FailureCategory(str, Enum):
    extraction_hallucination = "extraction_hallucination"
    misclassification = "misclassification"
    propagation_error = "propagation_error"
    prompt_failure = "prompt_failure"
    context_loss = "context_loss"
    pipeline_error = "pipeline_error"  # hard exception (outside the LLM-quality taxonomy)
    none = "none"


class StepJudgement(BaseModel):
    step_name: str
    score: int = Field(ge=1, le=5)
    issues: list[str] = Field(default_factory=list)
    suspected_category: FailureCategory = FailureCategory.none
    evidence: dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[int] = None


class EvidenceItem(BaseModel):
    step_name: str
    claim: str
    input_excerpt: str = ""
    output_excerpt: str = ""


class Diagnosis(BaseModel):
    diagnosis_id: str
    trace_id: str
    root_cause_step: Optional[str]
    category: FailureCategory
    explanation: str
    evidence_chain: list[EvidenceItem] = Field(default_factory=list)
    judgements: list[StepJudgement] = Field(default_factory=list)
    affected_downstream: list[str] = Field(default_factory=list)
    analysis_ms: float = 0.0
    judge: str = "heuristic"
    created_at: datetime = Field(default_factory=utcnow)
    # human review
    status: str = "proposed"  # proposed | confirmed | overridden
    confirmed_step: Optional[str] = None
    confirmed_category: Optional[FailureCategory] = None


# --------------------------------------------------------------------------- #
# Feedback / evals
# --------------------------------------------------------------------------- #
class FlagRequest(BaseModel):
    reason: str = ""
    flagged_by: str = "anonymous"


class ConfirmRequest(BaseModel):
    root_cause_step: Optional[str] = None  # override; default = proposed
    category: Optional[FailureCategory] = None  # override; default = proposed
    corrected_output: Optional[dict[str, Any]] = None  # human-corrected step output
    reviewer: str = "anonymous"
    notes: str = ""


class EvalCase(BaseModel):
    case_id: str
    source_trace_id: str
    doc: RawDocument
    failing_step: str
    category: FailureCategory
    bad_output: dict[str, Any]
    corrected_output: dict[str, Any]
    created_at: datetime = Field(default_factory=utcnow)
    created_by: str = "anonymous"


class EvalCaseResult(BaseModel):
    case_id: str
    passed: bool
    detail: str = ""


class EvalRun(BaseModel):
    run_id: str
    pipeline_version: str
    created_at: datetime = Field(default_factory=utcnow)
    total: int
    resolved: int
    still_failing: int
    results: list[EvalCaseResult] = Field(default_factory=list)
