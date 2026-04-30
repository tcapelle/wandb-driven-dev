---
name: wbagent
description: "Primary skill for querying, analyzing, and launching W&B projects. Covers W&B SDK (training runs, metrics, artifacts, sweeps), Weave SDK (GenAI traces, evaluations, scorers, monitors), and W&B Launch (reproducing runs, submitting training jobs to compute, queue management). Use this for: project overviews, run analysis, trace inspection, eval summaries, monitor setup, launching/relaunching runs, and any broad 'what's going on' questions. For authoring W&B Reports, use the `wandb-reports` skill instead."
---

# W&B Agent Skill

## Environment defaults

- **Python**: run scripts with `python` or `uv run python`, install packages with `uv add`
- Helper scripts live at `${CLAUDE_PLUGIN_ROOT}/skills/wbagent/scripts/`. The
  preamble below adds them to `sys.path`.

```python
import sys, os
sys.path.insert(0, f"{os.environ['CLAUDE_PLUGIN_ROOT']}/skills/wbagent/scripts")
```

---

## Fast recipes — use these first

These cover the most common tasks. Each is a single script. Copy, fill in placeholders, run.

### Count runs (exact, fast)

```python
import wandb, os
api = wandb.Api(timeout=60)
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"
total = len(api.runs(path, per_page=1, include_sweeps=False, lazy=True))
finished = len(api.runs(path, filters={"state": "finished"}, per_page=1, include_sweeps=False, lazy=True))
crashed = len(api.runs(path, filters={"state": "crashed"}, per_page=1, include_sweeps=False, lazy=True))
running = len(api.runs(path, filters={"state": "running"}, per_page=1, include_sweeps=False, lazy=True))
print(f"Total: {total}  |  Finished: {finished}  |  Crashed: {crashed}  |  Running: {running}")
```

### Count traces (fast, server-side)

```python
import weave, os, logging
logging.getLogger("weave").setLevel(logging.ERROR)
from weave.trace_server.trace_server_interface import CallsQueryStatsReq

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
client = weave.init(f"{entity}/{project}")
pid = f"{entity}/{project}"

# Total root traces
stats = client.server.calls_query_stats(CallsQueryStatsReq(
    project_id=pid, filter={"trace_roots_only": True}
))
print(f"Root traces: {stats.count}")

# Count by op name
for op in ["Evaluation.evaluate", "wb_agent.turn"]:
    s = client.server.calls_query_stats(CallsQueryStatsReq(
        project_id=pid,
        filter={"op_names": [f"weave:///{entity}/{project}/op/{op}:*"]},
    ))
    print(f"  {op}: {s.count}")
```

### Summarize project (runs + traces in one script)

```python
import wandb, weave, os, logging
logging.getLogger("weave").setLevel(logging.ERROR)
from weave.trace_server.trace_server_interface import CallsQueryStatsReq

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"

# --- Runs ---
api = wandb.Api(timeout=60)
total_runs = len(api.runs(path, per_page=1, include_sweeps=False, lazy=True))
finished = len(api.runs(path, filters={"state": "finished"}, per_page=1, include_sweeps=False, lazy=True))
recent = api.runs(path, order="-created_at", per_page=5)[:5]

print(f"=== Runs ({total_runs} total, {finished} finished) ===")
for r in recent:
    print(f"  {r.name} [{r.state}] {r.created_at[:10]}")

# --- Weave Traces ---
client = weave.init(path)
pid = f"{entity}/{project}"
root_stats = client.server.calls_query_stats(CallsQueryStatsReq(
    project_id=pid, filter={"trace_roots_only": True}
))
print(f"\n=== Weave Traces ({root_stats.count} root traces) ===")

recent_calls = list(client.get_calls(
    sort_by=[{"field": "started_at", "direction": "desc"}],
    limit=5,
    columns=["op_name", "started_at", "display_name"],
))
for c in recent_calls:
    name = c.display_name or c.op_name.split("/")[-1].split(":")[0]
    started = c.started_at.strftime("%Y-%m-%d %H:%M") if c.started_at else "?"
    print(f"  {name} @ {started}")
```

### Inspect a single run

```python
import wandb, os
api = wandb.Api(timeout=60)
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"

run = api.run(f"{path}/RUN_ID")
print(f"Name: {run.name}")
print(f"State: {run.state}")
print(f"Created: {run.created_at}")
print(f"Tags: {run.tags}")
print(f"Last step: {run.lastHistoryStep}")

# Key metrics (replace with actual keys from probe or user request)
for k in ["loss", "val_loss", "accuracy"]:
    v = run.summary_metrics.get(k)
    if v is not None:
        print(f"  {k}: {v}")
```

### Compare two runs

```python
import wandb, os, sys
sys.path.insert(0, f"{os.environ['CLAUDE_PLUGIN_ROOT']}/skills/wbagent/scripts")
from wandb_helpers import get_api, compare_configs

api = get_api()
path = f"{os.environ['WANDB_ENTITY']}/{os.environ['WANDB_PROJECT']}"

run_a = api.run(f"{path}/RUN_A_ID")
run_b = api.run(f"{path}/RUN_B_ID")

# Config diff
diffs = compare_configs(run_a, run_b)
if diffs:
    print("Config differences:")
    for d in diffs:
        print(f"  {d['key']}: {d[run_a.name]} -> {d[run_b.name]}")
else:
    print("Configs are identical")

# Metric comparison
print("\nMetrics:")
for k in ["loss", "val_loss", "accuracy"]:
    a = run_a.summary_metrics.get(k, "N/A")
    b = run_b.summary_metrics.get(k, "N/A")
    print(f"  {k}: {a} vs {b}")
```

### Summarize latest eval

```python
import weave, os, sys, logging
logging.getLogger("weave").setLevel(logging.ERROR)
from weave.trace.weave_client import CallsFilter
sys.path.insert(0, f"{os.environ['CLAUDE_PLUGIN_ROOT']}/skills/wbagent/scripts")
from weave_helpers import unwrap, eval_results_to_dicts, results_summary

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
client = weave.init(f"{entity}/{project}")

# Get latest eval
op_ref = f"weave:///{entity}/{project}/op/Evaluation.evaluate:*"
evals = list(client.get_calls(
    filter=CallsFilter(op_names=[op_ref]),
    sort_by=[{"field": "started_at", "direction": "desc"}],
    limit=1,
))

if not evals:
    print("No evaluations found")
else:
    ec = evals[0]
    print(f"Eval: {ec.display_name or 'unnamed'} @ {ec.started_at}")

    pas_ref = f"weave:///{entity}/{project}/op/Evaluation.predict_and_score:*"
    pas = list(client.get_calls(
        filter=CallsFilter(op_names=[pas_ref], parent_ids=[ec.id])
    ))
    results = eval_results_to_dicts(pas, agent_name=ec.display_name or "agent")
    print(results_summary(results))
```

### Inspect recent traces

```python
import weave, os, sys, logging
logging.getLogger("weave").setLevel(logging.ERROR)
sys.path.insert(0, f"{os.environ['CLAUDE_PLUGIN_ROOT']}/skills/wbagent/scripts")
from weave_helpers import unwrap, get_token_usage

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
client = weave.init(f"{entity}/{project}")

calls = list(client.get_calls(
    sort_by=[{"field": "started_at", "direction": "desc"}],
    limit=10,
))

for c in calls:
    name = c.display_name or c.op_name.split("/")[-1].split(":")[0]
    started = c.started_at.strftime("%Y-%m-%d %H:%M") if c.started_at else "?"
    duration = ""
    if c.started_at and c.ended_at:
        duration = f" ({(c.ended_at - c.started_at).total_seconds():.1f}s)"
    status = c.summary.get("weave", {}).get("status", "?") if c.summary else "?"
    tokens = get_token_usage(c)
    tok_str = f" [{tokens['total_tokens']} tok]" if tokens['total_tokens'] else ""
    print(f"  {name} [{status}] {started}{duration}{tok_str}")
```

### W&B Reports

For authoring or updating W&B Reports, use the **`wandb-reports`** skill (it
covers the recipes and helpers).

### Set up a Weave Monitor

```python
import weave, os

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
client = weave.init(f"{entity}/{project}")

@weave.op()
def my_scorer(output: dict) -> dict:
    """Score based on output quality."""
    passed = output.get("succeeded", False)
    return {"passed": passed, "score": 1.0 if passed else 0.0}

monitor = weave.Monitor(
    entity=entity,
    project=project,
    name="quality-monitor",
    scorers=[my_scorer],
)
print(f"Monitor created: {monitor.name}")
```

### Relaunch a run (1 command, auto-selects queue)

```bash
python ${CLAUDE_PLUGIN_ROOT}/skills/wbagent/scripts/launch_helpers.py relaunch \
  "https://wandb.ai/entity/project/runs/run_id" \
  --config '{"epochs": 100}'
```

### Relaunch a run (Python)

```python
import sys, os
sys.path.insert(0, f"{os.environ['CLAUDE_PLUGIN_ROOT']}/skills/wbagent/scripts")
from launch_helpers import parse_run_url, list_queues, relaunch_run

entity, project, run_id = parse_run_url("RUN_URL")
run_path = f"{entity}/{project}/{run_id}"

queues = list_queues(entity)
queue = queues[0]

relaunch_run(
    run_path=run_path,
    queue_name=queue["name"],
    namespace=queue["namespace"],
    config={"lr": 0.001, "epochs": 20},
)
```

### Modify code and launch

```python
import sys, os
sys.path.insert(0, f"{os.environ['CLAUDE_PLUGIN_ROOT']}/skills/wbagent/scripts")
from launch_helpers import get_job_artifact, download_code_artifact, list_queues, create_and_launch_modified_job

# Step 1: Download code
info = download_code_artifact("entity/project/job-name:latest")
# Edit files in info["code_dir"]...

# Step 2: Launch modified code
queues = list_queues("ENTITY")
queue = queues[0]
create_and_launch_modified_job(
    code_dir=info["code_dir"],
    entrypoint=info["entrypoint"],
    entity=info["entity"], project=info["project"],
    queue_name=queue["name"], namespace=queue["namespace"],
    job_name="my-modified-job",
    base_image=info["base_image"],
)
```

### Check status of a launched run

```bash
python ${CLAUDE_PLUGIN_ROOT}/skills/wbagent/scripts/launch_helpers.py check \
  "entity" "project" "queue-name" "QUEUE_ITEM_ID"
```

---

## Launch rules

1. **Minimize turns.** For a simple relaunch with config changes, use ONE command: `python ${CLAUDE_PLUGIN_ROOT}/skills/wbagent/scripts/launch_helpers.py relaunch <URL> --config '{"epochs": 100}'`
2. **The CLI auto-handles everything.** `relaunch` auto-discovers queues, selects the best one, finds the job artifact, and submits.
3. **Do NOT read `launch_helpers.py`.** This SKILL.md documents everything you need.
4. **Do NOT check WANDB_ENTITY/PROJECT env vars for launch.** The run URL contains the entity and project.
5. **NEVER fake a launch with `wandb.init()`.** Use `relaunch_run()` or the CLI.
6. **NEVER run training locally in the sandbox.** No GPU. Always use Launch.
7. **Config change vs code change — decide FIRST.**

| Change type | Examples | How to launch |
|---|---|---|
| **Config override** | epochs, lr, batch_size, any value in `wandb.config` | `relaunch_run(..., config={"epochs": 100})` |
| **Code change** | model architecture, loss function, data augmentation | Download code -> edit -> `create_and_launch_modified_job()` |

**Important**: If the user asks to change something that isn't a config field (e.g. "add more conv layers", "change the optimizer"), you MUST modify the code. Passing unknown config keys does nothing — the training script doesn't read them.

### Launch decision tree

| I need to... | Do this |
|---|---|
| Change hyperparameters only | `relaunch_run(run_path, queue_name, namespace, config={"epochs": 100})` |
| Change code (architecture, logic) | `download_code_artifact()` -> edit files -> `create_and_launch_modified_job()` |
| Launch from an artifact path | `launch_job_artifact(artifact_path, queue_name)` |
| Submit new code (no existing run) | `submit_code_artifact_job(code_files, entrypoint, ...)` |
| Check on a launched run | `check_launched_run(entity, project, queue_name, item_id)` |
| Find a queue | `list_queues(entity)` — use the recommended one |
| Create a new queue | `create_queue(name, entity, gpus=1, cpu=8, memory="80Gi")` |

### Step-by-step launch (when CLI one-liner isn't enough)

```python
import sys, os
sys.path.insert(0, f"{os.environ['CLAUDE_PLUGIN_ROOT']}/skills/wbagent/scripts")
from launch_helpers import parse_run_url, list_queues, get_job_artifact, relaunch_run, submit_code_artifact_job

entity, project, run_id = parse_run_url("https://wandb.ai/entity/project/runs/run_id")
run_path = f"{entity}/{project}/{run_id}"

queues = list_queues(entity)
queue = queues[0]

job_artifact = get_job_artifact(run_path)

if job_artifact:
    relaunch_run(run_path, queue["name"], queue["namespace"],
                 config={"lr": 0.001, "epochs": 20})
else:
    submit_code_artifact_job(
        code_files=["train.py"], entrypoint="python train.py",
        entity=entity, project=project,
        queue_name=queue["name"], job_name="my-train-job",
        base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
        requirements=["wandb"],
    )
```

### Launch infrastructure gotchas

- **Always pass `resource_args` explicitly** — queue defaults get double-nested by the server
- **Restart agent after queue delete/recreate** — agent loses its registration
- **`requirements.txt` is read from code dir** — `_create_job` does NOT inspect the venv
- **Keep `requirements.txt` minimal** — only deps not in base image
- **Build base images for `linux/amd64`** — not the default on Mac
- **Inject K8s secrets via `k8s_secrets` param** — not via queue defaults

---

## CRITICAL: Large project performance rules

These rules prevent 502 errors, timeouts, and multi-minute hangs on projects with 10K+ runs or runs with 1K+ metrics. **Violating any of these will cause failures on large projects.**

1. **Always use `wandb.Api(timeout=60)`** — the default 19s timeout causes constant failures
2. **NEVER call `history()` or `scan_history()` without explicit `keys=[...]`** — runs with 1K+ metrics will 502 or timeout when fetching all columns
3. **Use `per_page=min(limit, 1000)`** when calling `api.runs()` for list tasks, and use `per_page=1` for exact count tasks
4. **Prefer server-side filters** (`summary_metrics.X: {$gt: Y}`) over client-side iteration
5. **For exact counts, prefer `len(api.runs(..., per_page=1, include_sweeps=False, lazy=True))`** — never `len(list(runs))`
6. **Use `beta_scan_history`** for runs with 10K+ history steps — reads from parquet, not GraphQL
7. **Never iterate all config keys** unless explicitly needed — access specific keys by name
8. **Default to `include_sweeps=False` for read-only retrieval tasks**
9. **Use `calls_query_stats` for trace counts** — never materialize all calls just to count them

---

## When to use what

| I need to... | Use |
|---|---|
| Query training runs, loss curves, hyperparameters | **W&B SDK** (`wandb.Api()`) — see `references/WANDB_SDK.md` |
| Query GenAI traces, calls, evaluations | **Weave SDK** (`weave.init()`, `client.get_calls()`) — see `references/WEAVE_SDK.md` |
| Convert Weave wrapper types to plain Python | **`weave_helpers.unwrap()`** |
| Build a DataFrame from training runs | **`wandb_helpers.fetch_runs()`** (fast) or **`wandb_helpers.runs_to_dataframe()`** |
| Extract eval results for analysis | **`weave_helpers.eval_results_to_dicts()`** |
| Count traces without fetching them | **`calls_query_stats`** from Weave server API |
| Need low-level Weave filtering (CallsFilter, Query) | **Raw Weave SDK** — see `references/WEAVE_SDK.md` |
| Create or update a report | **`wandb-reports` skill** |
| Set up production monitoring | **`weave.Monitor`** |
| Reproduce/relaunch a run | **`launch_helpers.relaunch_run()`** or CLI |
| Launch a training job on GPU/K8s | **`launch_helpers.submit_code_artifact_job()`** |
| Modify code and launch | **`launch_helpers.download_code_artifact()`** -> edit -> **`create_and_launch_modified_job()`** |
| List or create launch queues | **`launch_helpers.list_queues()`** / **`create_queue()`** |

---

## Bundled files

### Helper libraries

```python
import sys, os
sys.path.insert(0, f"{os.environ['CLAUDE_PLUGIN_ROOT']}/skills/wbagent/scripts")

# Weave helpers (traces, evals, GenAI)
from weave_helpers import (
    unwrap,                  # Recursively convert Weave types -> plain Python
    get_token_usage,         # Extract token counts from a call's summary
    eval_results_to_dicts,   # predict_and_score calls -> list of result dicts
    pivot_solve_rate,        # Build task-level pivot table across agents
    results_summary,         # Print compact eval summary
    eval_health,             # Extract status/counts from Evaluation.evaluate calls
    eval_efficiency,         # Compute tokens-per-success across eval calls
)

# W&B helpers (training runs, metrics) — large-project optimized
from wandb_helpers import (
    get_api,             # Create API with safe timeout (default 60s)
    probe_project,       # Discover project scale, metrics, config BEFORE querying
    fetch_runs,          # FAST: Direct GraphQL with selective metrics (17x faster)
    runs_to_dataframe,   # Legacy: iterate run objects (slower, use fetch_runs instead)
    diagnose_run,        # Quick diagnostic summary (configurable metric keys)
    compare_configs,     # Side-by-side config diff between two runs
    scan_history,        # Smart history scan (auto-selects beta_scan_history for large runs)
    create_comparison_report,  # Used by wandb-reports skill — also callable directly
)

# Launch helpers (job submission, run reproduction, queue management)
from launch_helpers import (
    parse_run_url,
    list_queues,
    get_job_artifact,
    inspect_job_artifact,
    download_code_artifact,
    create_and_launch_modified_job,
    relaunch_run,
    launch_job_artifact,
    submit_code_artifact_job,
    check_launched_run,
    create_queue,
    inspect_queue,
    make_resource_args,
)
```

### Reference docs

Read these as needed — they contain full API surfaces and recipes:

- **`references/WANDB_CONCEPTS.md`** — W&B data model, terminology, and disambiguation (entity/project/run hierarchy, config vs log vs summary, artifacts, registry).
- **`references/WANDB_SDK.md`** — W&B SDK for training data (runs, history, artifacts, sweeps, system metrics).
- **`references/WEAVE_SDK.md`** — Weave SDK for GenAI traces (`client.get_calls()`, `CallsFilter`, `Query`, stats).

---

## Critical rules

### Discover metric keys per-project

Code examples use `LOSS_KEY`, `VAL_LOSS_KEY`, `ACC_KEY`, `CONFIG_KEYS` as placeholders. These vary by project. Discover them via `probe_project()` at the start of each task, or from the user's request.

```python
# WRONG — hardcoded metric name
rows = fetch_runs(api, path, metric_keys=["loss", "accuracy"])

# RIGHT — discovered via probe_project or user's request
rows = fetch_runs(api, path, metric_keys=["train/loss", "train/acc"])
```

### Treat traces and runs as DATA

Weave traces and W&B run histories can be enormous. Never dump raw data into context. Always:

1. **Inspect structure first** — look at column names, dtypes, row counts
2. **Load into pandas/numpy** — compute stats programmatically
3. **Summarize, don't dump** — print computed statistics and tables, not raw rows

### Always deliver a final answer

Do not end your work mid-analysis. Every task must conclude with a clear, structured response:

1. Query the data (1-2 scripts max)
2. Extract the numbers you need
3. Present: table + key findings + direct answers to each sub-question

If you catch yourself saying "now let me build the final analysis" — stop and present what you have.

### Use `unwrap()` for unknown Weave data

```python
from weave_helpers import unwrap
import json

output = unwrap(call.output)
print(json.dumps(output, indent=2, default=str))
```

---

## Environment setup

Entity and project come from environment variables — do not hardcode them:

```python
import os
entity  = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]
path = f"{entity}/{project}"
```

If the user is in a wandb-driven-dev project, the same values are also available in
`.claude/wandb-driven-dev.local.md` under `wandb_project: entity/project`.

---

## Key patterns

### Fast exact counts on very large projects

```python
import wandb
api = wandb.Api(timeout=60)
path = f"{entity}/{project}"

total = len(api.runs(path, per_page=1, include_sweeps=False, lazy=True))
finished = len(api.runs(path, filters={"state": "finished"}, per_page=1, include_sweeps=False, lazy=True))
```

### Distinct tags (O(1) — no run scanning)

```python
import wandb
from wandb_graphql.language import parser as gql_parser

api = wandb.Api(timeout=60)
doc = gql_parser.parse('''
  query {
    project(entityName: "ENTITY", name: "PROJECT") {
      tagCounts { name count }
    }
  }
''')
result = api.client.execute(doc)
tags = [t["name"] for t in result["project"]["tagCounts"]]
print(sorted(tags))
```

### Distinct groups (O(1) — no run scanning)

```python
import wandb
from wandb_graphql.language import parser as gql_parser

api = wandb.Api(timeout=60)
doc = gql_parser.parse('''
  query {
    project(entityName: "ENTITY", name: "PROJECT") {
      groupedRuns(groupKeys: ["group"], first: 100) {
        ... on GroupedRunConnection {
          edges {
            node { group totalRuns }
          }
        }
      }
    }
  }
''')
result = api.client.execute(doc)
edges = result["project"]["groupedRuns"]["edges"]
groups = [e["node"]["group"] for e in edges if e["node"]["group"]]
print(sorted(groups))
```

### W&B SDK — fast run fetching (17x faster on large projects)

```python
import pandas as pd
from wandb_helpers import get_api, fetch_runs

api = get_api()
path = f"{entity}/{project}"

rows = fetch_runs(
    api, path,
    metric_keys=["LOSS_KEY", "ACC_KEY"],
    filters={"state": "finished"},
    limit=100,
)
df = pd.DataFrame(rows)
print(df.describe())
```

### Weave — eval call hierarchy

```
Evaluation.evaluate (root)
  +-- Evaluation.predict_and_score (one per dataset row x trials)
  |     +-- model.predict (the actual model call)
  |     +-- scorer_1.score
  |     +-- scorer_2.score
  +-- Evaluation.summarize
```

### Token usage

```python
from weave_helpers import get_token_usage

usage = get_token_usage(call)
print(f"Tokens: {usage['total_tokens']} (in={usage['input_tokens']}, out={usage['output_tokens']})")
```

---

## Gotchas

### Weave API

| Gotcha | Wrong | Right |
|--------|-------|-------|
| weave.init args | `weave.init(project="x")` | `weave.init("x")` (positional) |
| Parent filter | `filter={'parent_id': 'x'}` | `filter={'parent_ids': ['x']}` (plural, list) |
| WeaveObject access | `rubric.get('passed')` | `getattr(rubric, 'passed', None)` |
| Nested output | `out.get('succeeded')` | `out.get('output').get('succeeded')` (output.output) |
| ObjectRef comparison | `name_ref == "foo"` | `str(name_ref) == "foo"` |
| CallsFilter import | `from weave import CallsFilter` | `from weave.trace.weave_client import CallsFilter` |
| Query import | `from weave import Query` | `from weave.trace_server.interface.query import Query` |
| Eval status path | `summary["status"]` | `summary["weave"]["status"]` |
| Eval success count | `summary["success_count"]` | `summary["weave"]["status_counts"]["success"]` |
| When in doubt | Guess the type | `unwrap()` first, then inspect |

### W&B API

| Gotcha | Wrong | Right |
|--------|-------|-------|
| API timeout | `wandb.Api()` (19s default) | `wandb.Api(timeout=60)` or `get_api()` |
| Summary access | `run.summary["loss"]` | `run.summary_metrics.get("LOSS_KEY")` |
| Loading all runs | `list(api.runs(...))` | `runs[:200]` (always slice) |
| Counting runs | `len(list(api.runs(...)))` | `len(api.runs(..., per_page=1, include_sweeps=False, lazy=True))` |
| Distinct tags | iterate all runs collecting `run.tags` | GraphQL `tagCounts` query |
| Distinct groups | iterate all runs collecting `run.group` | GraphQL `groupedRuns` query |
| `run.config` after lazy fetch | `run.config` returns `{}` | Use `lazy=False` when you need config |
| Pagination | `api.runs(path)` (per_page=50 default) | `api.runs(path, per_page=min(N, 1000))` |
| History — no keys on large run | `run.history(samples=10)` -> 502 | `run.history(samples=10, keys=["LOSS_KEY"])` |
| scan_history — no keys | `scan_history()` -> timeout | `scan_history(keys=["LOSS_KEY"])` |
| Large history (10K+ steps) | `scan_history(keys=[...])` | `beta_scan_history(keys=[...])` (parquet) |
| Cross-run search | iterate all runs client-side | Server-side filter: `{"summary_metrics.X": {"$gt": Y}}` |

### Launch

| Gotcha | Wrong | Right |
|--------|-------|-------|
| List queues | `api.run_queues()` or raw GQL | `list_queues(entity)` from helpers |
| resource_args | Rely on queue defaults | Pass via `make_resource_args()` |
| requirements.txt | `pip freeze` from venv | Write manually — only deps missing from base image |
| Base image arch | `docker build` on Mac | `docker buildx build --platform linux/amd64` |
| Fake launch | `wandb.init()` with config | `relaunch_run()` or `launch_job_artifact()` |
| Unknown config key | `relaunch_run(config={"conv_layers": 4})` | Code change — download, edit, `create_and_launch_modified_job()` |

### Weave logging noise

```python
import logging
logging.getLogger("weave").setLevel(logging.ERROR)
```
