"""Wiring: build the store, pipeline, analyzer and feedback service from settings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .analyzer import BackwardTraceAnalyzer, build_analyzer
from .config import Settings, get_settings
from .corpus import load_golden
from .feedback import FeedbackService
from .llm import LLMClient, get_llm
from .pipeline import Pipeline
from .store import TraceStore


@dataclass
class Services:
    settings: Settings
    store: TraceStore
    llm: LLMClient
    analyzer: BackwardTraceAnalyzer
    feedback: FeedbackService
    golden: dict

    def pipeline(self, version: Optional[str] = None) -> Pipeline:
        return Pipeline(settings=self.settings, llm=self.llm, store=self.store, version=version)

    def reload_golden(self) -> None:
        self.golden.clear()
        self.golden.update(load_golden(self.settings.data_dir / "golden.json"))


def build_services(settings: Optional[Settings] = None) -> Services:
    settings = settings or get_settings()
    store = TraceStore.from_settings(settings)
    llm = get_llm(settings.use_openai, settings.openai_model)
    golden = load_golden(settings.data_dir / "golden.json")
    analyzer = build_analyzer(settings, llm, golden)

    def factory(version: str) -> Pipeline:
        return Pipeline(settings=settings, llm=llm, store=store, version=version)

    feedback = FeedbackService(store, analyzer, factory, golden)
    return Services(settings, store, llm, analyzer, feedback, golden)
