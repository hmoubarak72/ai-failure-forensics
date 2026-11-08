"""Failure analytics: where should the team invest engineering effort?"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

from .store import TraceStore


def _bucket(ts: str, granularity: str) -> str:
    dt = datetime.fromisoformat(ts)
    return dt.strftime("%Y-%m-%d %H:00" if granularity == "hour" else "%Y-%m-%d")


def compute_analytics(store: TraceStore, granularity: str = "hour") -> dict[str, Any]:
    traces = store.list_traces(limit=100000)
    diags = store.list_diagnoses()
    latest: dict[str, Any] = {}
    for d in diags:  # keep the latest diagnosis per trace
        latest[d.trace_id] = d
    diags = [d for d in latest.values() if d.root_cause_step]

    types = Counter((d.confirmed_category or d.category).value for d in diags)
    steps = Counter(d.confirmed_step or d.root_cause_step for d in diags)
    flagged = {t["trace_id"]: t for t in traces if t["flagged"]}

    over_time: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "failed": 0, "degraded": 0, "flagged": 0})
    for t in traces:
        b = over_time[_bucket(t["created_at"], granularity)]
        b["total"] += 1
        b["failed"] += t["status"] == "failure"
        b["degraded"] += t["status"] == "degraded"
        b["flagged"] += bool(t["flagged"])
    timeline = [
        {"bucket": k, **v, "failure_rate": round((v["failed"] + v["flagged"]) / v["total"], 3) if v["total"] else 0.0}
        for k, v in sorted(over_time.items())
    ]

    ttrc = []  # time from human flag to diagnosis
    for d in diags:
        t = flagged.get(d.trace_id)
        if t and t["flagged_at"]:
            ttrc.append(max(0.0, (d.created_at - datetime.fromisoformat(t["flagged_at"])).total_seconds() * 1000))
    analysis_ms = [d.analysis_ms for d in diags]

    by_version: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "flagged": 0, "failure": 0, "degraded": 0})
    for t in traces:
        v = by_version[t["pipeline_version"]]
        v["total"] += 1
        v["flagged"] += bool(t["flagged"])
        v["failure"] += t["status"] == "failure"
        v["degraded"] += t["status"] == "degraded"

    runs = store.list_eval_runs()
    return {
        "totals": {
            "traces": len(traces),
            "flagged": len(flagged),
            "diagnosed": len(diags),
            "confirmed": sum(d.status in ("confirmed", "overridden") for d in diags),
            "eval_cases": len(store.list_eval_cases()),
            "status": dict(Counter(t["status"] for t in traces)),
        },
        "failure_types": dict(types.most_common()),
        "failing_steps": dict(steps.most_common()),
        "failure_rate_over_time": timeline,
        "by_pipeline_version": dict(by_version),
        "avg_time_to_root_cause_ms": round(sum(ttrc) / len(ttrc), 2) if ttrc else None,
        "avg_analysis_ms": round(sum(analysis_ms) / len(analysis_ms), 3) if analysis_ms else None,
        "regression_trend": [
            {"run_id": r.run_id, "created_at": r.created_at.isoformat(), "pipeline_version": r.pipeline_version,
             "total": r.total, "resolved": r.resolved, "resolved_pct": round(100 * r.resolved / r.total, 1) if r.total else 0.0}
            for r in runs
        ],
    }
