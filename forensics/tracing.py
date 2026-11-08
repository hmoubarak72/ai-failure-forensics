"""Tracing layer: one Trace per pipeline run, one Span per step.

Design
------
* ``Tracer.trace(...)`` opens a trace context (contextvar) and an OpenTelemetry root span.
* ``@traced_step("name")`` is the one-line instrumentation for a step. It captures the
  serialized input, serialized output, latency, errors and the step's self-reported
  confidence, and opens a child OpenTelemetry span carrying the same data as attributes.
* LLM clients call ``record_llm_call(...)`` which attaches prompt, raw response, token
  counts and latency to whichever span is currently active.

The custom ``Span`` models are the source of truth stored as JSON + SQLite; the
OpenTelemetry spans make the same data exportable to any OTLP backend (Jaeger,
Honeycomb, Grafana Tempo) by setting ``OTEL_EXPORTER_OTLP_ENDPOINT``.
"""
from __future__ import annotations

import contextvars
import functools
import json
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode
from pydantic import BaseModel

from .models import LLMCall, Span, SpanStatus, Trace

_current_trace: contextvars.ContextVar[Optional[Trace]] = contextvars.ContextVar("current_trace", default=None)
_current_span: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar("current_span", default=None)

_provider_configured = False


def _configure_otel(console: bool = False) -> None:
    global _provider_configured
    if _provider_configured:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": "failure-forensics"}))
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        try:  # optional dependency
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
        except ImportError:  # pragma: no cover
            pass
    if console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    otel_trace.set_tracer_provider(provider)
    _provider_configured = True


def _serialize(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, (list, tuple)):
        return [_serialize(o) for o in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def _truncate(s: str, n: int = 4000) -> str:
    return s if len(s) <= n else s[:n] + "...<truncated>"


def _capture_input(args: tuple, kwargs: dict) -> dict[str, Any]:
    """Serialize the typed (Pydantic) inputs of a step; runtime objects are skipped."""
    typed = [a for a in args if isinstance(a, BaseModel)]
    typed_kw = {k: v for k, v in kwargs.items() if isinstance(v, BaseModel)}
    if len(typed) == 1 and not typed_kw:
        return _serialize(typed[0])
    out: dict[str, Any] = {f"arg{i}": _serialize(a) for i, a in enumerate(typed)}
    out.update({k: _serialize(v) for k, v in typed_kw.items()})
    return out


def current_trace() -> Optional[Trace]:
    return _current_trace.get()


def record_llm_call(call: LLMCall) -> None:
    """Attach an LLM call to the active span (no-op outside a traced step)."""
    span = _current_span.get()
    if span is not None:
        span.llm_calls.append(call)
    otel_span = otel_trace.get_current_span()
    if otel_span.is_recording():
        otel_span.add_event(
            "llm.call",
            {
                "llm.model": call.model,
                "llm.prompt_tokens": call.prompt_tokens,
                "llm.completion_tokens": call.completion_tokens,
                "llm.latency_ms": call.latency_ms,
            },
        )


class Tracer:
    def __init__(self, console_export: bool = False):
        _configure_otel(console_export)
        self._otel = otel_trace.get_tracer("failure-forensics")

    @contextmanager
    def trace(self, doc_id: str, pipeline_version: str) -> Iterator[Trace]:
        with self._otel.start_as_current_span("pipeline.run") as root:
            trace_id = format(root.get_span_context().trace_id, "032x")
            tr = Trace(trace_id=trace_id, doc_id=doc_id, pipeline_version=pipeline_version)
            root.set_attribute("forensics.doc_id", doc_id)
            root.set_attribute("forensics.pipeline_version", pipeline_version)
            token = _current_trace.set(tr)
            try:
                yield tr
            finally:
                root.set_attribute("forensics.status", tr.status.value)
                _current_trace.reset(token)


def traced_step(step_name: str) -> Callable:
    """Decorator: instrument a pipeline step with one line of code."""

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            tr = _current_trace.get()
            if tr is None:  # untraced execution (e.g. unit tests of a single step)
                return fn(*args, **kwargs)

            span = Span(
                span_id=uuid.uuid4().hex[:16],
                trace_id=tr.trace_id,
                step_index=len(tr.spans),
                step_name=step_name,
                input=_capture_input(args, kwargs),
            )
            tr.spans.append(span)
            tok = _current_span.set(span)
            tracer = otel_trace.get_tracer("failure-forensics")
            t0 = time.perf_counter()
            with tracer.start_as_current_span(f"step.{step_name}") as otel_span:
                otel_span.set_attribute("forensics.step", step_name)
                otel_span.set_attribute("forensics.step_index", span.step_index)
                otel_span.set_attribute("forensics.input", _truncate(json.dumps(span.input, default=str)))
                try:
                    result = fn(*args, **kwargs)
                    span.output = _serialize(result)
                    conf = getattr(result, "confidence", None)
                    span.confidence = int(conf) if conf is not None else None
                    otel_span.set_attribute("forensics.output", _truncate(json.dumps(span.output, default=str)))
                    if span.confidence is not None:
                        otel_span.set_attribute("forensics.confidence", span.confidence)
                    return result
                except Exception as exc:  # record and re-raise
                    span.status = SpanStatus.error
                    span.error = f"{type(exc).__name__}: {exc}"
                    otel_span.record_exception(exc)
                    otel_span.set_status(Status(StatusCode.ERROR, span.error))
                    raise
                finally:
                    span.latency_ms = round((time.perf_counter() - t0) * 1000, 3)
                    otel_span.set_attribute("forensics.tokens", span.total_tokens)
                    _current_span.reset(tok)

        wrapper.step_name = step_name  # type: ignore[attr-defined]
        return wrapper

    return decorator
