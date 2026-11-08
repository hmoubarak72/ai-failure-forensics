"""Command line interface.

  forensics seed                 write the 50-document demo corpus + golden labels
  forensics run [--version v1]   run the corpus through the pipeline
  forensics demo                 full portfolio scenario (traffic, flags, diagnoses, evals, trend)
  forensics analyze TRACE_ID     print the root-cause diagnosis for one trace
  forensics evals [--version]    re-run the eval dataset against a pipeline version
  forensics serve                start the REST API
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timedelta, timezone

from .analytics import compute_analytics
from .config import get_settings
from .corpus import load_documents, load_golden, write_corpus
from .feedback import compare_step, expected_output
from .models import ConfirmRequest, FlagRequest, Trace
from .services import build_services


def _seed(settings) -> None:
    dp, gp = write_corpus(settings.data_dir)
    print(f"Wrote {dp} and {gp}")


def reviewer_says_bad(tr: Trace, golden: dict) -> bool:
    """Simulated human reviewer: compares the pipeline's work against the known answer."""
    if tr.status.value == "failure":
        return True
    for step in ("extraction", "classification", "summarization"):
        exp = expected_output(step, golden)
        sp = tr.span(step)
        if exp and sp and not compare_step(step, sp.output, exp)[0]:
            return True
    return False


def cmd_demo(args) -> None:
    settings = get_settings()
    if args.reset and settings.data_dir.exists():
        # clear contents, not the directory itself (it may be a mounted volume)
        for child in settings.data_dir.iterdir():
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        settings.ensure_dirs()
    _seed(settings)
    s = build_services(settings)
    s.reload_golden()
    docs = load_documents(settings.data_dir / "documents.jsonl")
    start = datetime.now(timezone.utc) - timedelta(days=15)
    schedule = [("v1", 0), ("v2", 5), ("v3", 10)]  # (version, first day)
    per_day = 10

    print(f"LLM: {s.llm.model} | judge: {s.analyzer.judge.name}")
    for version, day0 in schedule:
        pipe = s.pipeline(version)
        flagged = confirmed = 0
        for idx, doc in enumerate(docs):
            day = day0 + idx // per_day
            tr = pipe.run(doc, persist=False)
            tr.created_at = start + timedelta(days=day, minutes=idx)
            s.store.save_trace(tr, doc)
            if reviewer_says_bad(tr, s.golden.get(doc.doc_id, {})):
                s.feedback.flag(tr.trace_id, FlagRequest(reason="output does not match source", flagged_by="demo-reviewer"))
                flagged += 1
                if version == "v1":  # new failures become eval cases; later versions re-surface known ones
                    s.feedback.confirm(tr.trace_id, ConfirmRequest(reviewer="demo-reviewer"))
                    confirmed += 1
            if (idx + 1) % per_day == 0:
                s.feedback.run_regression(version, created_at=start + timedelta(days=day, hours=23))
        print(f"[{version}] {len(docs)} docs | flagged {flagged} | new eval cases {confirmed}")

    a = compute_analytics(s.store, "day")
    print("\nFailure types:  ", a["failure_types"])
    print("Failing steps:  ", a["failing_steps"])
    print("Avg analysis ms:", a["avg_analysis_ms"])
    print("\nRegression trend (known issues resolved):")
    seen = set()
    for r in a["regression_trend"]:
        key = (r["pipeline_version"], r["resolved"], r["total"])
        if key not in seen:
            seen.add(key)
            print(f"  {r['created_at'][:10]}  {r['pipeline_version']}  {r['resolved']}/{r['total']} ({r['resolved_pct']}%)")


def cmd_run(args) -> None:
    settings = get_settings()
    if not (settings.data_dir / "documents.jsonl").exists():
        _seed(settings)
    s = build_services(settings)
    pipe = s.pipeline(args.version)
    for doc in load_documents(settings.data_dir / "documents.jsonl"):
        tr = pipe.run(doc)
        print(f"{tr.trace_id}  {doc.doc_id}  {tr.status.value:<8}  score={tr.final_score}")


def cmd_analyze(args) -> None:
    s = build_services()
    tr = s.store.get_trace(args.trace_id)
    if tr is None:
        sys.exit(f"trace {args.trace_id} not found")
    d = s.analyzer.analyze(tr)
    print(json.dumps(d.model_dump(mode="json"), indent=2) if args.json else d.explanation)


def cmd_evals(args) -> None:
    s = build_services()
    run = s.feedback.run_regression(args.version or s.settings.pipeline_version)
    print(f"{run.pipeline_version}: {run.resolved}/{run.total} known issues resolved")
    for r in run.results:
        print(f"  {'PASS' if r.passed else 'FAIL'}  {r.case_id}  {r.detail}")


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("forensics.api:app", host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="forensics", description="Failure forensics for AI pipelines")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("seed").set_defaults(fn=lambda a: _seed(get_settings()))
    r = sub.add_parser("run"); r.add_argument("--version", default=None); r.set_defaults(fn=cmd_run)
    d = sub.add_parser("demo"); d.add_argument("--no-reset", dest="reset", action="store_false"); d.set_defaults(fn=cmd_demo)
    an = sub.add_parser("analyze"); an.add_argument("trace_id"); an.add_argument("--json", action="store_true"); an.set_defaults(fn=cmd_analyze)
    e = sub.add_parser("evals"); e.add_argument("--version", default=None); e.set_defaults(fn=cmd_evals)
    sv = sub.add_parser("serve"); sv.add_argument("--host", default="0.0.0.0"); sv.add_argument("--port", type=int, default=8000); sv.set_defaults(fn=cmd_serve)
    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
