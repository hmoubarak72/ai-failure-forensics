"""Runtime configuration, read from environment variables (12-factor style)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("FORENSICS_DATA_DIR", "./data")))
    pipeline_version: str = field(default_factory=lambda: _env("PIPELINE_VERSION", "v1"))
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "auto"))  # auto|openai|mock
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    judge_provider: str = field(default_factory=lambda: _env("JUDGE_PROVIDER", "auto"))  # auto|llm|heuristic
    low_confidence_threshold: int = field(default_factory=lambda: int(_env("LOW_CONFIDENCE_THRESHOLD", "2")))
    otel_console: bool = field(default_factory=lambda: _env("OTEL_CONSOLE_EXPORT", "0") == "1")

    @property
    def traces_dir(self) -> Path:
        return self.data_dir / "traces"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "forensics.db"

    @property
    def eval_dataset_path(self) -> Path:
        return self.data_dir / "eval_dataset.jsonl"

    @property
    def use_openai(self) -> bool:
        if self.llm_provider == "openai":
            return True
        if self.llm_provider == "mock":
            return False
        return bool(os.environ.get("OPENAI_API_KEY"))

    def ensure_dirs(self) -> None:
        self.traces_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
