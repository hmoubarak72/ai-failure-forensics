"""Storage: one human-readable JSON file per trace + a SQLite index for queries.

    data/
      traces/<trace_id>.json     full trace (spans, prompts, raw responses)
      forensics.db               SQLite: traces, documents, diagnoses, eval_cases, eval_runs
      eval_dataset.jsonl         append-only eval dataset (git-friendly)
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import Diagnosis, EvalCase, EvalRun, RawDocument, Trace

SCHEMA = """
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    final_score REAL NOT NULL,
    pipeline_version TEXT NOT NULL,
    path TEXT NOT NULL,
    min_confidence INTEGER,
    total_tokens INTEGER,
    total_latency_ms REAL,
    flagged INTEGER NOT NULL DEFAULT 0,
    flag_reason TEXT,
    flagged_by TEXT,
    flagged_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_traces_created ON traces(created_at);
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS diagnoses (
    diagnosis_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    category TEXT NOT NULL,
    root_cause_step TEXT,
    analysis_ms REAL,
    json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_diag_trace ON diagnoses(trace_id);
CREATE TABLE IF NOT EXISTS eval_cases (
    case_id TEXT PRIMARY KEY,
    source_trace_id TEXT NOT NULL,
    category TEXT NOT NULL,
    failing_step TEXT NOT NULL,
    created_at TEXT NOT NULL,
    json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_runs (
    run_id TEXT PRIMARY KEY,
    pipeline_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    total INTEGER NOT NULL,
    resolved INTEGER NOT NULL,
    json TEXT NOT NULL
);
"""


class TraceStore:
    def __init__(self, db_path: Path, traces_dir: Path, eval_dataset_path: Optional[Path] = None):
        self.db_path = Path(db_path)
        self.traces_dir = Path(traces_dir)
        self.eval_dataset_path = Path(eval_dataset_path) if eval_dataset_path else self.db_path.parent / "eval_dataset.jsonl"
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    @classmethod
    def from_settings(cls, settings: Any) -> "TraceStore":
        return cls(settings.db_path, settings.traces_dir, settings.eval_dataset_path)

    # -- helpers ------------------------------------------------------------
    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _query(self, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    # -- traces -------------------------------------------------------------
    def save_trace(self, trace: Trace, doc: Optional[RawDocument] = None) -> Path:
        path = self.traces_dir / f"{trace.trace_id}.json"
        path.write_text(trace.model_dump_json(indent=2), encoding="utf-8")
        confs = [s.confidence for s in trace.spans if s.confidence is not None]
        self._exec(
            """INSERT OR REPLACE INTO traces
               (trace_id, doc_id, created_at, status, final_score, pipeline_version, path,
                min_confidence, total_tokens, total_latency_ms)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                trace.trace_id,
                trace.doc_id,
                trace.created_at.isoformat(),
                trace.status.value,
                trace.final_score,
                trace.pipeline_version,
                str(path),
                min(confs) if confs else None,
                sum(s.total_tokens for s in trace.spans),
                round(sum(s.latency_ms for s in trace.spans), 3),
            ),
        )
        if doc is not None:
            self._exec("INSERT OR REPLACE INTO documents (doc_id, json) VALUES (?,?)", (doc.doc_id, doc.model_dump_json()))
        return path

    def get_trace(self, trace_id: str) -> Optional[Trace]:
        rows = self._query("SELECT path FROM traces WHERE trace_id=?", (trace_id,))
        if not rows:
            return None
        p = Path(rows[0]["path"])
        if not p.exists():
            p = self.traces_dir / f"{trace_id}.json"
        return Trace.model_validate_json(p.read_text(encoding="utf-8"))

    def list_traces(self, status: Optional[str] = None, flagged: Optional[bool] = None, limit: int = 500) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM traces WHERE 1=1", []
        if status:
            sql += " AND status=?"
            params.append(status)
        if flagged is not None:
            sql += " AND flagged=?"
            params.append(1 if flagged else 0)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        return self._query(sql, tuple(params))

    def flag_trace(self, trace_id: str, reason: str, flagged_by: str) -> None:
        self._exec(
            "UPDATE traces SET flagged=1, flag_reason=?, flagged_by=?, flagged_at=? WHERE trace_id=?",
            (reason, flagged_by, datetime.now(timezone.utc).isoformat(), trace_id),
        )

    def get_document(self, doc_id: str) -> Optional[RawDocument]:
        rows = self._query("SELECT json FROM documents WHERE doc_id=?", (doc_id,))
        return RawDocument.model_validate_json(rows[0]["json"]) if rows else None

    # -- diagnoses ----------------------------------------------------------
    def save_diagnosis(self, d: Diagnosis) -> None:
        self._exec(
            """INSERT OR REPLACE INTO diagnoses
               (diagnosis_id, trace_id, created_at, status, category, root_cause_step, analysis_ms, json)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                d.diagnosis_id,
                d.trace_id,
                d.created_at.isoformat(),
                d.status,
                (d.confirmed_category or d.category).value,
                d.confirmed_step or d.root_cause_step,
                d.analysis_ms,
                d.model_dump_json(),
            ),
        )

    def latest_diagnosis(self, trace_id: str) -> Optional[Diagnosis]:
        rows = self._query("SELECT json FROM diagnoses WHERE trace_id=? ORDER BY created_at DESC LIMIT 1", (trace_id,))
        return Diagnosis.model_validate_json(rows[0]["json"]) if rows else None

    def list_diagnoses(self) -> list[Diagnosis]:
        return [Diagnosis.model_validate_json(r["json"]) for r in self._query("SELECT json FROM diagnoses ORDER BY created_at")]

    # -- evals --------------------------------------------------------------
    def add_eval_case(self, case: EvalCase) -> None:
        self._exec(
            "INSERT OR REPLACE INTO eval_cases (case_id, source_trace_id, category, failing_step, created_at, json) VALUES (?,?,?,?,?,?)",
            (case.case_id, case.source_trace_id, case.category.value, case.failing_step, case.created_at.isoformat(), case.model_dump_json()),
        )
        with self._lock, self.eval_dataset_path.open("a", encoding="utf-8") as fh:
            fh.write(case.model_dump_json() + "\n")

    def list_eval_cases(self) -> list[EvalCase]:
        return [EvalCase.model_validate_json(r["json"]) for r in self._query("SELECT json FROM eval_cases ORDER BY created_at")]

    def save_eval_run(self, run: EvalRun) -> None:
        self._exec(
            "INSERT INTO eval_runs (run_id, pipeline_version, created_at, total, resolved, json) VALUES (?,?,?,?,?,?)",
            (run.run_id, run.pipeline_version, run.created_at.isoformat(), run.total, run.resolved, run.model_dump_json()),
        )

    def list_eval_runs(self) -> list[EvalRun]:
        return [EvalRun.model_validate_json(r["json"]) for r in self._query("SELECT json FROM eval_runs ORDER BY created_at")]

    def close(self) -> None:
        self._conn.close()
