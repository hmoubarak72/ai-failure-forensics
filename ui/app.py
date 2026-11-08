"""Streamlit trace explorer for Failure Forensics.

Talks to the REST API only (FORENSICS_API_URL), so the UI and the API scale and deploy
independently - exactly how it runs under docker compose.
"""
from __future__ import annotations

import difflib
import json
import os
from typing import Any, Optional

import httpx
import pandas as pd
import streamlit as st

API = os.environ.get("FORENSICS_API_URL", "http://localhost:8000").rstrip("/")
STEPS = ["intake", "extraction", "classification", "summarization"]
CATEGORIES = ["extraction_hallucination", "misclassification", "propagation_error", "prompt_failure",
              "context_loss", "pipeline_error", "none"]
COLORS = {"healthy": "#2e9d5b", "low": "#e0b100", "root": "#d64545", "error": "#8a1c1c", "missing": "#9aa0a6"}

st.set_page_config(page_title="Failure Forensics", page_icon="🔎", layout="wide")


# --------------------------------------------------------------------------- #
# API helpers
# --------------------------------------------------------------------------- #
def api(method: str, path: str, **kw: Any) -> Any:
    try:
        r = httpx.request(method, f"{API}{path}", timeout=120, **kw)
    except httpx.HTTPError as exc:
        st.error(f"API unreachable at {API}: {exc}")
        st.stop()
    if r.status_code >= 400:
        st.error(f"{method} {path} -> {r.status_code}: {r.text}")
        return None
    return r.json()


def pretty(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def node_state(span: Optional[dict], diag: Optional[dict], threshold: int = 2) -> str:
    if span is None:
        return "missing"
    if span.get("error"):
        return "error"
    if diag:
        root = diag.get("confirmed_step") or diag.get("root_cause_step")
        if root == span["step_name"] and (diag.get("confirmed_category") or diag.get("category")) != "none":
            return "root"
        j = next((j for j in diag.get("judgements", []) if j["step_name"] == span["step_name"]), None)
        if j and j["score"] <= 3:
            return "low"
    if span.get("confidence") is not None and span["confidence"] <= threshold:
        return "low"
    return "healthy"


def graph_dot(spans: dict[str, dict], diag: Optional[dict]) -> str:
    lines = ['digraph G { rankdir=LR; graph [size="10,2.4" ranksep=0.5 nodesep=0.25]; node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=11 fontcolor=white];']
    for i, step in enumerate(STEPS):
        sp = spans.get(step)
        state = node_state(sp, diag)
        conf = f"\\nconf {sp['confidence']}/5" if sp and sp.get("confidence") is not None else ""
        lat = f"\\n{sp['latency_ms']:.1f} ms" if sp else "\\nnot run"
        lines.append(f'{step} [label="{i + 1}. {step.title()}{conf}{lat}" fillcolor="{COLORS[state]}"];')
    lines += ["intake -> extraction;", "intake -> classification;", "extraction -> summarization;",
              "classification -> summarization;", "}"]
    return "\n".join(lines)


def diff_html(a: Any, b: Any) -> str:
    left, right = pretty(a).splitlines(), pretty(b).splitlines()
    return difflib.HtmlDiff(wrapcolumn=70).make_table(left, right, "produced", "expected", context=False)


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_explorer() -> None:
    st.header("🔎 Trace Explorer")
    c1, c2, c3 = st.columns([1, 1, 2])
    status = c1.selectbox("Status", ["any", "success", "degraded", "failure"])
    flagged = c2.selectbox("Flagged", ["any", "flagged", "not flagged"])
    params: dict[str, Any] = {"limit": 1000}
    if status != "any":
        params["status"] = status
    if flagged != "any":
        params["flagged"] = flagged == "flagged"
    rows = api("GET", "/traces", params=params) or []
    if not rows:
        st.info("No traces yet. Run `forensics demo` or use the *Run a document* page.")
        return
    df = pd.DataFrame(rows)
    df["label"] = df.apply(lambda r: f"{r.doc_id} · {r.pipeline_version} · {r.status}{' · 🚩' if r.flagged else ''} · {r.trace_id[:10]}", axis=1)
    default = st.session_state.get("selected_trace")
    ids = df["trace_id"].tolist()
    choice = c3.selectbox("Trace", ids, index=ids.index(default) if default in ids else 0,
                          format_func=lambda t: df.set_index("trace_id").loc[t, "label"])
    st.session_state["selected_trace"] = choice

    tr = api("GET", f"/traces/{choice}")
    diag = api("GET", f"/traces/{choice}/diagnosis")
    spans = {s["step_name"]: s for s in tr["spans"]}

    m = st.columns(5)
    m[0].metric("Status", tr["status"])
    m[1].metric("Final score", f"{tr['final_score']:.2f}")
    m[2].metric("Pipeline", tr["pipeline_version"])
    m[3].metric("Tokens", sum(c["prompt_tokens"] + c["completion_tokens"] for s in tr["spans"] for c in s["llm_calls"]))
    m[4].metric("Latency", f"{sum(s['latency_ms'] for s in tr['spans']):.1f} ms")
    if tr.get("validation_notes"):
        st.caption("Automatic checks: " + " · ".join(tr["validation_notes"]))

    st.graphviz_chart(graph_dot(spans, diag), width="stretch")
    st.caption("🟢 healthy  🟡 low confidence / judge concern  🔴 identified root cause  🟥 hard error  ⚪ not run")

    # ---- flag + diagnosis --------------------------------------------------
    left, right = st.columns([3, 2])
    with right:
        st.subheader("Flag & diagnose")
        reason = st.text_input("Why is this output bad?", key=f"reason-{choice}")
        if st.button("🚩 Flag as bad output", type="primary", width="stretch"):
            diag = api("POST", f"/traces/{choice}/flag", json={"reason": reason, "flagged_by": "ui"})
            st.rerun()
        if diag:
            cat = diag.get("confirmed_category") or diag["category"]
            st.markdown(f"**Root cause:** `{diag.get('confirmed_step') or diag['root_cause_step']}` · **Category:** `{cat}` · "
                        f"**Status:** {diag['status']} · {diag['analysis_ms']:.2f} ms ({diag['judge']} judge)")
            st.info(diag["explanation"])
            with st.expander("Evidence chain", expanded=True):
                for ev in diag["evidence_chain"]:
                    st.markdown(f"**{ev['step_name']}** — {ev['claim']}")
                    st.code(f"input:  {ev['input_excerpt']}\noutput: {ev['output_excerpt']}", language="text")
            with st.expander("Judge scores per step"):
                st.dataframe(pd.DataFrame([{"step": j["step_name"], "judge score": j["score"], "self-confidence": j["confidence"],
                                            "suspected": j["suspected_category"], "issues": "; ".join(j["issues"])}
                                           for j in diag["judgements"]]), hide_index=True, width="stretch")
            st.markdown("**Confirm or override**")
            root = diag.get("confirmed_step") or diag["root_cause_step"] or STEPS[0]
            step = st.selectbox("Root-cause step", STEPS, index=STEPS.index(root) if root in STEPS else 0, key=f"cs-{choice}")
            category = st.selectbox("Failure category", CATEGORIES, index=CATEGORIES.index(cat) if cat in CATEGORIES else 0, key=f"cc-{choice}")
            d = api("GET", f"/traces/{choice}/diff/{step}") or {}
            corrected = st.text_area("Corrected output for this step (JSON)", pretty(d.get("expected") or d.get("produced") or {}),
                                     height=180, key=f"co-{choice}-{step}")
            if st.button("✅ Confirm → add to eval dataset", width="stretch"):
                try:
                    body = {"root_cause_step": step, "category": category, "corrected_output": json.loads(corrected), "reviewer": "ui"}
                except json.JSONDecodeError as exc:
                    st.error(f"Corrected output is not valid JSON: {exc}")
                else:
                    case = api("POST", f"/traces/{choice}/confirm", json=body)
                    if case:
                        st.success(f"Eval case {case['case_id']} created ({case['category']} @ {case['failing_step']}).")

    # ---- span details + diff -------------------------------------------------
    with left:
        st.subheader("Span details")
        sel = st.radio("Pipeline step (node)", STEPS, horizontal=True, key=f"node-{choice}",
                       index=STEPS.index(diag["root_cause_step"]) if diag and diag.get("root_cause_step") in STEPS else 0)
        sp = spans.get(sel)
        if sp is None:
            st.warning("This step did not run (an upstream step failed).")
            return
        k = st.columns(4)
        k[0].metric("Confidence", sp["confidence"] if sp["confidence"] is not None else "–")
        k[1].metric("Latency", f"{sp['latency_ms']:.2f} ms")
        k[2].metric("Tokens", sum(c["prompt_tokens"] + c["completion_tokens"] for c in sp["llm_calls"]))
        k[3].metric("Status", sp["status"])
        if sp.get("error"):
            st.error(sp["error"])
        t_io, t_prompt, t_diff = st.tabs(["Input / Output", "LLM prompt & raw response", "Diff view"])
        with t_io:
            a, b = st.columns(2)
            a.markdown("**Input**"); a.code(pretty(sp["input"]), language="json")
            b.markdown("**Output**"); b.code(pretty(sp["output"]), language="json")
        with t_prompt:
            if not sp["llm_calls"]:
                st.caption("No LLM call in this step (deterministic code).")
            for c in sp["llm_calls"]:
                st.caption(f"{c['model']} · {c['prompt_tokens']} prompt + {c['completion_tokens']} completion tokens · {c['latency_ms']:.2f} ms")
                st.code(c["prompt"], language="text")
                st.markdown("**Raw response**"); st.code(c["raw_response"], language="json")
        with t_diff:
            d = api("GET", f"/traces/{choice}/diff/{sel}") or {}
            a, b, c = st.columns(3)
            a.markdown("**Received**"); a.code(pretty(d.get("received")), language="json")
            b.markdown("**Produced**"); b.code(pretty(d.get("produced")), language="json")
            c.markdown(f"**Should have produced** ({d.get('expected_source') or 'no reference'})")
            c.code(pretty(d.get("expected")), language="json")
            if d.get("expected") is not None and d.get("produced") is not None:
                st.markdown("**Divergence (produced vs expected)**")
                st.html(f"<style>table.diff{{font-family:monospace;font-size:12px}} .diff_add{{background:#c8f7c5}} .diff_chg{{background:#fff3b0}} .diff_sub{{background:#ffc9c9}}</style>{diff_html(d['produced'], d['expected'])}")


def page_evals() -> None:
    st.header("🧪 Eval dataset & regression tracking")
    health = api("GET", "/health") or {}
    versions = health.get("pipeline_versions", ["v1", "v2", "v3"])
    c1, c2 = st.columns([1, 3])
    version = c1.selectbox("Pipeline version", versions, index=len(versions) - 1)
    if c1.button("▶ Re-run eval dataset", type="primary"):
        run = api("POST", "/evals/run", params={"pipeline_version": version})
        if run:
            st.success(f"{run['pipeline_version']}: {run['resolved']}/{run['total']} known issues resolved")
    runs = api("GET", "/evals/runs") or []
    if runs:
        df = pd.DataFrame([{"run": i + 1, "date": r["created_at"][:16], "version": r["pipeline_version"], "resolved": r["resolved"],
                            "still failing": r["still_failing"], "resolved %": round(100 * r["resolved"] / r["total"], 1) if r["total"] else 0}
                           for i, r in enumerate(runs)])
        c2.markdown("**Known issues resolved over time**")
        c2.line_chart(df.set_index("run")[["resolved", "still failing"]])
        st.dataframe(df, hide_index=True, width="stretch")
        latest = runs[-1]
        st.markdown(f"**Latest run** ({latest['pipeline_version']})")
        st.dataframe(pd.DataFrame(latest["results"]), hide_index=True, width="stretch")
    cases = api("GET", "/evals/cases") or []
    st.subheader(f"Eval cases ({len(cases)})")
    if cases:
        st.dataframe(pd.DataFrame([{"case": c["case_id"], "doc": c["doc"]["doc_id"], "failing step": c["failing_step"],
                                    "category": c["category"], "from trace": c["source_trace_id"][:12], "created": c["created_at"][:16]}
                                   for c in cases]), hide_index=True, width="stretch")


def page_analytics() -> None:
    st.header("📊 Failure analytics")
    gran = st.radio("Granularity", ["day", "hour"], horizontal=True)
    a = api("GET", "/analytics", params={"granularity": gran})
    if not a:
        return
    t = a["totals"]
    k = st.columns(6)
    k[0].metric("Traces", t["traces"]); k[1].metric("Flagged", t["flagged"]); k[2].metric("Diagnosed", t["diagnosed"])
    k[3].metric("Eval cases", t["eval_cases"])
    k[4].metric("Avg analysis", f"{a['avg_analysis_ms']} ms" if a["avg_analysis_ms"] is not None else "–")
    k[5].metric("Avg flag→root cause", f"{a['avg_time_to_root_cause_ms']} ms" if a["avg_time_to_root_cause_ms"] is not None else "–")
    c1, c2 = st.columns(2)
    if a["failure_types"]:
        c1.markdown("**Most common failure types**")
        c1.bar_chart(pd.Series(a["failure_types"], name="count"), horizontal=True)
    if a["failing_steps"]:
        c2.markdown("**Which step fails most often**")
        c2.bar_chart(pd.Series(a["failing_steps"], name="count"), horizontal=True)
    if a["failure_rate_over_time"]:
        st.markdown("**Failure rate over time** (hard failures + human-flagged)")
        st.line_chart(pd.DataFrame(a["failure_rate_over_time"]).set_index("bucket")[["failure_rate"]])
    if a["by_pipeline_version"]:
        st.markdown("**By pipeline version**")
        st.dataframe(pd.DataFrame(a["by_pipeline_version"]).T, width="stretch")


def page_run() -> None:
    st.header("▶ Run a document")
    health = api("GET", "/health") or {}
    st.caption(f"LLM: {health.get('llm')} · judge: {health.get('judge')}")
    c1, c2, c3 = st.columns(3)
    doc_id = c1.text_input("Document id", "adhoc-001")
    fmt = c2.selectbox("Source format", ["text", "markdown", "pdf_text"])
    version = c3.selectbox("Pipeline version", health.get("pipeline_versions", ["v1", "v2", "v3"]))
    text = st.text_area("Document text", height=260, placeholder="Paste a contract, invoice, report or letter…")
    if st.button("Run pipeline", type="primary") and text.strip():
        res = api("POST", "/pipeline/run", json={"document": {"doc_id": doc_id, "text": text, "source_format": fmt}, "pipeline_version": version})
        if res:
            st.success(f"Trace {res['trace_id']} · {res['status']} · score {res['final_score']}")
            st.session_state["selected_trace"] = res["trace_id"]
            st.json(res["final_output"])
            st.caption("Open the Trace Explorer to inspect it.")


PAGES = {"Trace Explorer": page_explorer, "Evals & Regression": page_evals, "Analytics": page_analytics, "Run a document": page_run}
st.sidebar.title("Failure Forensics")
st.sidebar.caption("Where did this AI pipeline go wrong?")
page = st.sidebar.radio("Navigate", list(PAGES))
st.sidebar.caption(f"API: {API}")
PAGES[page]()
