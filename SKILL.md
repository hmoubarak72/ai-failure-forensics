---
name: ai-failure-forensics
description: Run, query and extend AI Failure Forensics (hmoubarak72/ai-failure-forensics) - trace LLM pipeline runs, flag bad outputs, diagnose root-cause steps, grow the eval dataset, run regressions, read failure analytics.
---

# AI Failure Forensics

Observability layer for a 4-step LLM document pipeline
(intake → extraction → classification → summarization). It traces every step, diagnoses
where a bad output originated (backward walk + step judges), turns confirmed diagnoses into
eval cases, and tracks regressions across prompt versions (v1/v2/v3).

Repo: https://github.com/hmoubarak72/ai-failure-forensics

Use this skill when the user asks to: run documents through the pipeline, find out why an
output is wrong, flag/confirm a diagnosis, add eval cases, run regression evals, compare
prompt versions, report failure analytics, demo the project, or extend it (new step,
new prompt version, new failure category, OpenAI mode, Docker deployment).

## 1. Get a running instance

Work out where you are first, then pick the first option that works.

**A. Already running?** `curl -s localhost:8000/health`. If it returns `"status":"ok"`, skip to section 2.

**B. Docker (preferred: matches production).**
```bash
git clone https://github.com/hmoubarak72/ai-failure-forensics.git && cd ai-failure-forensics
cp .env.example .env          # leave OPENAI_API_KEY empty for offline mock mode
docker compose up --build -d  # API :8000 (seeds demo on empty volume), UI :8501
```
If `docker info` says the daemon is not running in a Linux sandbox, start it with
`(dockerd > /tmp/dockerd.log 2>&1 &) ; sleep 5`. If image pulls fail with `Forbidden`
(registry blocked), do not keep retrying registries; fall back to option C and tell the user.

**C. Local Python (no Docker).**
```bash
pip install -r requirements.txt        # add --break-system-packages on managed system Pythons
python -m forensics.cli demo           # seeds ./data: 50 docs x 3 versions, flags, evals
python -m forensics.cli serve --port 8000 &        # REST API
FORENSICS_API_URL=http://localhost:8000 streamlit run ui/app.py --server.headless true &   # UI :8501
```
Background processes may be stopped after the turn ends. For one-off answers, prefer the
CLI or the Python API (section 3) over starting servers.

`forensics demo` wipes and rebuilds the contents of `FORENSICS_DATA_DIR` (default `./data`).
Use `--no-reset`, or skip it, when the user has real traces they care about. Ask before
resetting non-demo data.

## 2. Core workflows (REST API, base `http://localhost:8000`, docs at `/docs`)

| Goal | Call |
|---|---|
| Run one doc | `POST /pipeline/run` `{"document":{"doc_id":"x","text":"...","source_format":"text"},"pipeline_version":"v1"}` |
| Run many | `POST /pipeline/batch` `{"documents":[...],"pipeline_version":"v2"}` |
| List traces | `GET /traces?status=degraded&flagged=true` |
| Inspect trace | `GET /traces/{trace_id}` (spans: input, output, llm_calls[prompt, raw_response, tokens], confidence, latency, error) |
| Diagnose (dry run) | `POST /traces/{id}/analyze` (not persisted) |
| Flag bad output | `POST /traces/{id}/flag` `{"reason":"...","flagged_by":"..."}` → diagnosis |
| See what should have happened | `GET /traces/{id}/diff/{step}` → received / produced / expected |
| Confirm diagnosis → eval case | `POST /traces/{id}/confirm` `{}` |
| Override | `POST /traces/{id}/confirm` `{"root_cause_step":"classification","category":"misclassification","corrected_output":{"doc_type":"correspondence"}}` |
| Regression | `POST /evals/run?pipeline_version=v3` → resolved / total + per-case detail |
| History | `GET /evals/runs`, `GET /evals/cases` |
| Analytics | `GET /analytics?granularity=day` |

Typical "why is this output wrong?" sequence: find the trace (`GET /traces`, filter by doc
or status) → `POST .../flag` → read `explanation`, `root_cause_step`, `category`,
`evidence_chain` → `GET .../diff/{root_cause_step}` → ask the user to confirm or override →
`POST .../confirm`.

Only confirm or override when the user agrees: a confirmation permanently adds a case to
`eval_dataset.jsonl`. Flagging and analyzing are safe to do on your own.

## 3. CLI and Python API (no server needed)

```bash
python -m forensics.cli run --version v2          # run corpus, prints trace ids + status
python -m forensics.cli analyze <trace_id> [--json]
python -m forensics.cli evals --version v3         # regression over the eval dataset
```
```python
from forensics.services import build_services
s = build_services()                          # reads env: FORENSICS_DATA_DIR, LLM_PROVIDER, ...
tr = s.pipeline("v1").run(raw_document)       # forensics.models.RawDocument
d = s.analyzer.analyze(tr)                    # Diagnosis
from forensics.analytics import compute_analytics; compute_analytics(s.store, "day")
```

## 4. Interpreting results

Trace status: `failure` = a step raised an error; `degraded` = low self-confidence (≤2) or a
broken invariant; `success` = passed the cheap automatic checks. **Success is not the same as
correct.** Confidently wrong outputs (e.g. hallucinated "John Smith" at confidence 4/5) only
surface through human flags, and that is why the feedback loop exists.

Root cause = the earliest step whose judge score is ≤ 2. Each step is judged only on its own
transformation, so a summary that repeats an upstream hallucination is listed under
`affected_downstream` and is not blamed.

| Category | Meaning | Usual fix |
|---|---|---|
| `extraction_hallucination` | entity not in the source | grounding rule in the extraction prompt (v2) |
| `prompt_failure` | explicit instruction ignored (e.g. currency relabeled) | sharper, rule-style instruction (v2) |
| `misclassification` | wrong doc type | primary-purpose rule / examples (v3) |
| `propagation_error` | correct input misread by the next step | copy-verbatim rule, structured hand-off (v3) |
| `context_loss` | facts in the input dropped from the output | chunking or explicit enumeration (open in demo) |
| `pipeline_error` | hard exception (e.g. empty OCR page) | input validation / fallback |
| `none` | no significant drop; lowest-confidence step is the suspect | needs a human look |

Expected demo numbers (mock LLM): v1 flags 12/50; regression v1 0/12 → v2 5/12 → v3 9/12.
Clearly different numbers in mock mode mean something changed. Run `pytest -q` (14 tests).

When reporting to the user, lead with the root-cause sentence, then the evidence (produced
vs expected), then the suggested fix. For analytics, name the top failure type and the
weakest step: that is where engineering effort should go next.

## 5. Configuration (env or `.env`)

`OPENAI_API_KEY` (unset = deterministic `MockLLM`), `OPENAI_MODEL` (gpt-4o-mini),
`LLM_PROVIDER` auto|openai|mock, `JUDGE_PROVIDER` auto|llm|heuristic (llm = LLM-as-judge
combined with heuristics, lower score wins), `PIPELINE_VERSION` v1|v2|v3,
`FORENSICS_DATA_DIR` (./data), `FORENSICS_SEED_DEMO` 1|0, `OTEL_EXPORTER_OTLP_ENDPOINT`
(export spans to Jaeger/Tempo). Never print or commit the API key.

## 6. Extending (code map)

| Task | Where |
|---|---|
| Add a pipeline step | `forensics/pipeline.py`: typed function + `@traced_step("name")`, add to `STEPS` and `Pipeline.run` |
| New prompt version | `forensics/prompts.py`: add to `VERSIONS`, gate rules with `_rules(version, ("v4", RULE_X))` |
| Mock behaviour for a new failure | `forensics/llm.py` `MockLLM`: react only to prompt text |
| New judge check / category | `forensics/judge.py` (+ enum in `models.py`, wording in `analyzer.py::_explain`) |
| Regression comparator | `forensics/feedback.py::compare_step` |
| UI page | `ui/app.py`: talks to the REST API only |

After any change, run `pytest -q`, then `docker compose up --build` and `curl localhost:8000/health`.
Add a test in `tests/test_forensics.py` for each new failure mode.

## 7. Troubleshooting

- `Device or resource busy: '/app/data'`: an old build; the demo must clear the volume's contents, not the directory itself. Rebuild.
- UI says "API unreachable": check `FORENSICS_API_URL` (`http://api:8000` in compose, `http://localhost:8000` locally) and `docker compose ps`.
- Port taken: change the host-side port in `docker-compose.yml` (`"8080:8000"`).
- Empty analytics: no traces yet, so run `forensics demo` or POST documents.
- OpenAI mode errors: the pipeline is JSON-mode; a non-JSON reply raises in that step and shows up as `pipeline_error` in the trace, which is itself a diagnosable failure.
