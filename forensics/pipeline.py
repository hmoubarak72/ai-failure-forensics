"""The 4-step document pipeline: intake -> extraction -> classification -> summarization.

Each step is an isolated function: typed Pydantic input in, typed Pydantic output out.
Instrumenting a step is one line: ``@traced_step("name")``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

from .config import Settings, get_settings
from .llm import LLMClient, get_llm
from .models import (
    ClassificationOutput,
    DocType,
    Entities,
    ExtractionOutput,
    IntakeOutput,
    RawDocument,
    StepInput,
    SummarizationOutput,
    Summary,
    Trace,
    TraceStatus,
)
from . import prompts
from .store import TraceStore
from .tracing import Tracer, traced_step

STEPS = ("intake", "extraction", "classification", "summarization")


@dataclass
class Runtime:
    llm: LLMClient
    version: str


def _clamp_conf(v: object, default: int = 3) -> int:
    try:
        return max(1, min(5, int(v)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #
@traced_step("intake")
def intake(doc: RawDocument, rt: Runtime) -> IntakeOutput:
    text = doc.text.replace("\r\n", "\n")
    if doc.source_format == "pdf_text":
        text = re.sub(r"-\n(\w)", r"\1", text)  # de-hyphenate PDF line breaks
        text = re.sub(r"\n?Page \d+ of \d+\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not text:
        raise ValueError("Document is empty after normalisation")
    return IntakeOutput(
        doc_id=doc.doc_id,
        title=doc.title or text.splitlines()[0][:80],
        clean_text=text,
        char_count=len(text),
        word_count=len(text.split()),
        source_format=doc.source_format,
    )


@traced_step("extraction")
def extraction(inp: IntakeOutput, rt: Runtime) -> ExtractionOutput:
    data = rt.llm.complete_json(prompts.extraction_prompt(inp.clean_text, rt.version))
    conf = _clamp_conf(data.pop("confidence", 3))
    return ExtractionOutput(doc_id=inp.doc_id, entities=Entities.model_validate(data), confidence=conf)


@traced_step("classification")
def classification(inp: IntakeOutput, rt: Runtime) -> ClassificationOutput:
    data = rt.llm.complete_json(prompts.classification_prompt(inp.clean_text, rt.version))
    raw_type = str(data.get("doc_type", "")).strip().lower()
    doc_type = DocType(raw_type) if raw_type in DocType._value2member_map_ else DocType.report
    return ClassificationOutput(
        doc_id=inp.doc_id,
        doc_type=doc_type,
        rationale=str(data.get("rationale", "")),
        confidence=_clamp_conf(data.get("confidence", 3)),
    )


@traced_step("summarization")
def summarization(inp: StepInput, rt: Runtime) -> SummarizationOutput:
    assert inp.extraction is not None and inp.classification is not None
    entities_json = inp.extraction.entities.model_dump_json(indent=1)
    data = rt.llm.complete_json(
        prompts.summarization_prompt(
            inp.intake.clean_text, inp.classification.doc_type.value, entities_json, rt.version
        )
    )
    conf = _clamp_conf(data.pop("confidence", 3))
    summary = Summary(
        doc_type=inp.classification.doc_type,
        headline=str(data.get("headline", "")),
        parties=[str(x) for x in data.get("parties", [])],
        key_dates=[str(x) for x in data.get("key_dates", [])],
        key_amounts=[str(x) for x in data.get("key_amounts", [])],
        bullets=[str(x) for x in data.get("bullets", [])],
    )
    return SummarizationOutput(doc_id=inp.intake.doc_id, summary=summary, confidence=conf)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
class Pipeline:
    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm: Optional[LLMClient] = None,
        store: Optional[TraceStore] = None,
        version: Optional[str] = None,
    ):
        self.settings = settings or get_settings()
        self.version = version or self.settings.pipeline_version
        if self.version not in prompts.VERSIONS:
            raise ValueError(f"Unknown pipeline version {self.version}; choose from {prompts.VERSIONS}")
        self.llm = llm or get_llm(self.settings.use_openai, self.settings.openai_model)
        self.store = store
        self.tracer = Tracer(console_export=self.settings.otel_console)

    def run(self, doc: RawDocument, persist: bool = True) -> Trace:
        rt = Runtime(llm=self.llm, version=self.version)
        with self.tracer.trace(doc.doc_id, self.version) as tr:
            try:
                i = intake(doc, rt)
                e = extraction(i, rt)
                c = classification(i, rt)
                s = summarization(StepInput(intake=i, extraction=e, classification=c), rt)
                tr.final_output = s.model_dump(mode="json")
                self._assess(tr)
            except Exception as exc:  # the failing span already holds the error
                tr.status = TraceStatus.failure
                tr.final_score = 0.0
                tr.validation_notes.append(f"Pipeline aborted: {type(exc).__name__}: {exc}")
        if persist and self.store is not None:
            self.store.save_trace(tr, doc)
        return tr

    def _assess(self, tr: Trace) -> None:
        """Cheap automatic checks run on every trace (no judge, no human)."""
        threshold = self.settings.low_confidence_threshold
        confs = [s.confidence for s in tr.spans if s.confidence is not None]
        notes: list[str] = []
        low = [s.step_name for s in tr.spans if s.confidence is not None and s.confidence <= threshold]
        if low:
            notes.append(f"Low self-reported confidence in: {', '.join(low)}")
        summ = (tr.final_output or {}).get("summary", {})
        if not summ.get("headline"):
            notes.append("Empty summary headline")
        if summ.get("doc_type") in ("contract", "invoice") and not summ.get("key_dates"):
            notes.append("Contract/invoice summary has no dates")
        tr.validation_notes = notes
        tr.final_score = round(sum(confs) / (5 * len(confs)), 3) if confs else 0.0
        tr.status = TraceStatus.degraded if notes else TraceStatus.success


def entities_json(e: Entities) -> str:
    return json.dumps(e.model_dump(mode="json"), indent=1)
