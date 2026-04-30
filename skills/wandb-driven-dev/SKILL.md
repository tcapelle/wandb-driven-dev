---
name: wandb-driven-dev
description: "Enforce a systematic, reproducible approach to empirical questions — hypothesis first, baseline + variant, smoke before full, wandb as source of truth. Use whenever the user asks 'does X work?', 'is A better than B?', 'what's the best N?', or says 'experiment', 'ablation', 'benchmark', 'sweep'. Applies to ML training, ablations, perf benchmarks, correctness tests. Produces wandb runs reviewable and mergeable with confidence."
user-invocable: true
argument-hint: "[setup | reconfigure | review experiment <slug> | <empty>]"
allowed-tools: Read, Write, Edit, Bash, Grep, Glob, Agent
model: opus
---

# Experiment-Driven Development

Any empirical claim is backed by a wandb run, a baseline to compare against, and a falsifier written before the run started. Nothing merges on vibes. If a required piece is missing (hypothesis, baseline, passing smoke), stop and surface the gap — don't work around it.

## When to invoke

Any empirical question. Skip if the answer is already in existing code, logs, or wandb runs — check first.

## Setup mode

If invoked as `setup` (or `reconfigure`), run **Phase 0a** to interview the user and write the project config. Otherwise proceed normally; if config is missing, run Phase 0a first.

## Project config

`.claude/wandb-driven-dev.local.md` holds the per-project preferences gathered once and reused for every experiment. The file is YAML frontmatter (structured fields: wandb project, launcher command, default metrics, GPU counts, training entrypoint) followed by a markdown body with free-form project notes the agents read verbatim. Schema lives in `scripts/wdd_helpers.py:default_config`. Read it first thing on every invocation:

```python
import sys, os
sys.path.insert(0, f"{os.environ['CLAUDE_PLUGIN_ROOT']}/skills/wandb-driven-dev/scripts")
from wdd_helpers import read_config
cfg = read_config()  # None if not yet configured
```

## Wandb work delegated to wbagent

Don't reinvent wandb queries. Every helper used below lives in
`${CLAUDE_PLUGIN_ROOT}/skills/wbagent/scripts/wandb_helpers.py`. Standard preamble:

```python
import sys, os
sys.path.insert(0, f"{os.environ['CLAUDE_PLUGIN_ROOT']}/skills/wbagent/scripts")
from wandb_helpers import (
    get_api, fetch_runs, find_runs_by_config,
    verify_required_metrics, find_run_by_name, runtime_estimate,
    compare_configs, scan_history,
)
```

wandb-driven-dev helpers (config IO, the experiment Reports wrapper) live in `scripts/wdd_helpers.py`.

## Off-thread analysis

For any analysis that touches more than ~5 runs or scans history, delegate to
the `wandb-query` agent via the Agent tool. It runs queries off the main
thread and returns a structured summary. Use it for:

- "What runs match this filter and how do their metrics compare?"
- "Has anyone in this project hit metric X > Y?"
- "Diagnose why this run crashed/diverged."

Don't delegate single-run lookups or anything you can answer in one
`run.summary_metrics.get(...)` call — the round-trip overhead isn't worth it.

## Run identification (mandatory)

Every wandb run launched by this skill must be findable by a single workspace filter:

- **Tag:** every launch passes `--wandb_tags exp/<slug>` (smoke runs additionally tag `smoke`; variants additionally tag `variant`).
- **Name:** wandb run name is `exp-<slug>-<role>`, where `<role>` ∈ `{smoke-baseline, smoke-variant, baseline, variant, variant-<id>}`. If the launcher names the underlying job, use the same string.

If the launcher recorded in `cfg.launcher.command` doesn't thread `--wandb_tags`/`--wandb_name` through to the training script, fix the launcher *before* running experiments — don't bypass identification to ship faster.

## Phases

Each has a gate; do not advance until met.

### 0a. First-run setup (gate: config written)

Skip if `.claude/wandb-driven-dev.local.md` exists. Otherwise interview the user via `AskUserQuestion`, then write the config:

1. **wandb project** (e.g. `entity/project`).
2. **Launch command** — exact string: a local invocation (`uv run python scripts/train.py …`) or a cluster submit (`uv run python k8s/launch.py …`, `sbatch slurm/train.sh …`). The skill does not care about the launcher family beyond what reproduction model it uses.
3. **Reproduction model** — how does the launcher get the code? `working_tree` (local, runs uncommitted edits as-is), `clone` (submitter clones origin/HEAD; needs clean tree + pushed commit), `shared_fs` (compute mounts the working tree), `image` (built/pushed before submit).
4. **Training script + config dir** — for plan.md to cite without guessing.
5. **GPU defaults** — smoke (usually 1) and full (project-typical).
6. **Decision metrics** — exact wandb keys driving pass/fail. Probe the project before asking, so the question is grounded:
   ```python
   from wandb_helpers import get_api, probe_project
   probe_project(get_api(), "<entity>/<project>")
   ```
   If the project has zero finished runs, surface that — usually a typo'd slug.
7. **Health metrics** — keys watched for divergence/NaN.
8. **Project notes (optional)** — anything you'd want future-you to know that doesn't fit the schema (dataset quirks, known-good baseline IDs, launcher gotchas). Goes into the markdown body of `.local.md`.

Write with `wdd_helpers.write_config(cfg, notes=...)` and confirm the file path back to the user.

### 0b. Worktree (gate: on `exp/<slug>` branch with plan.md scaffold)

Experiments never run from the main checkout. Bootstrap:

```bash
WT="$(${CLAUDE_PLUGIN_ROOT}/skills/wandb-driven-dev/scripts/bootstrap_experiment.sh <slug> | tail -1)"
cd "$WT"
```

Slug convention: `YYYYMMDD-<short-kebab>`. Script refuses to clobber an existing branch or worktree, and runs `uv sync` automatically if a `pyproject.toml` is present. If already in a worktree on `exp/*`, keep working there.

### 1. Hypothesis (gate: user approval of plan.md)

Most context comes from the project config. Per-experiment, ask only:

- **Baseline status:** to-be-created (default), fresh re-run on this branch, or pinned existing wandb run. For pinned baselines, verify the chosen decision metrics already exist on it:
  ```python
  run = get_api().run("<entity>/<project>/<pinned-id>")
  {k: run.summary_metrics.get(k) for k in cfg["metrics"]["decision"]}
  ```
- **Per-experiment metric overrides** — only if this experiment cares about something beyond `cfg.metrics`.

Write `experiments/<slug>/plan.md`:

```markdown
# <slug>

**Date:** YYYY-MM-DD
**Question:** one sentence — what are we trying to learn?

## Hypothesis
What we expect to see, stated as a prediction.

## Falsifier
The concrete observation that would prove the hypothesis wrong.
Measurable on wandb: metric key + threshold + which runs.

## Success criteria
Quantitative bar for "variant wins".

## Baseline
- **Status:** to-be-created | fresh re-run | pinned
- **Wandb run:** URL or `TBD — recorded in Phase 4`
- **Why this is the right control:** one line.

## Variant
The change(s) from baseline. Default: one knob, two runs (baseline + variant). For multi-variant benchmarks (≥3 runs comparing one named dimension like architecture or hidden_dim), list each variant; verdict becomes a ranking table. Same data/eval/budget across runs.

## Metrics
Inherited from `.claude/wandb-driven-dev.local.md` unless overridden here.
- **Decision:** ...
- **Health:** ...

## Design        (filled in Phase 2)
## Smoke         (filled in Phase 3)
## Runs          (filled in Phase 4)
## Result        (filled in Phase 5)
```

Show plan.md and wait for explicit approval.

### 2. Design (gate: user approval of budget + commands)

Pick the **smallest step budget** that beats the falsifier with comfortable margin — don't default to the config's `max_steps`.

| Question type | Typical budget |
| --- | --- |
| LR / warmup / optimizer stability | 500–2k |
| Scaling laws | 1k–3k |
| Loss / target scheme | 2k–5k |
| Architecture change | 5k–10k |
| Final paper-number claim | full `max_steps` |

Write into `## Design`:

- launch command for baseline (exact, copy-pasteable, with `--max_steps` override if needed) — uses `cfg.launcher.command`
- launch command(s) for variant(s) — same budget, only the named dimension changes
- step budget + one-line rationale
- expected wall-clock + GPU count

Optionally use `runtime_estimate(api, project, name_pattern, target_steps)` for the wall-clock projection.

### 3. Smoke (gate: clean exit AND every metric in `## Metrics` shows up in wandb)

Shrunk config for both baseline and variant: `max_steps: 50–200`, `num_gpus: cfg.gpus.smoke`, `val_every_n_steps` small enough to fire once. Always pass `--wandb_tags smoke,exp/<slug>` and `--wandb_name exp-<slug>-smoke-{baseline,variant}` through the launcher.

Each smoke must:
1. exit 0
2. log ≥1 train step and ≥1 val step
3. not NaN / diverge on first val

**Verify from training-process logs, not the launcher's success signal** — many launchers report success while the underlying training crashed. Capture stdout/stderr to `experiments/<slug>/logs/` and grep for failure markers (`Traceback`, `CUDA out of memory`, `NaN`, `ChildFailedError`).

**Verify metric logging via wandb.** Use the wbagent helper:

```python
required = cfg["metrics"]["decision"] + cfg["metrics"]["health"]  # plus any per-experiment additions
result = verify_required_metrics(get_api(), [
    "<entity>/<project>/<smoke-baseline-id>",
    "<entity>/<project>/<smoke-variant-id>",
], required)
assert all(not missing for missing in result.values()), f"Phase 3 metric gate failed: {result}"
```

If a required metric is missing: instrument the log call, fix eval cadence, or — only with explicit user acknowledgment — drop the metric and re-smoke.

### 4. Launch (fire-and-forget)

Submit baseline and variant with the approved commands. Use `--wandb_tags exp/<slug>,<role>` and `--wandb_name exp-<slug>-<role>`. If the baseline is pinned, only launch the variant.

Resolve URLs via wandb, not pod logs (the wandb startup banner can lag the launcher by minutes; `wandb.init` registers the run on the server within seconds):

```python
for role in ["baseline", "variant"]:
    run = find_run_by_name(get_api(), cfg["wandb_project"], f"exp-<slug>-{role}", timeout_s=120)
    print(f"{role}: {run.url if run else 'NOT FOUND'}")
```

Create the live dashboard:

```python
from wdd_helpers import create_experiment_report
url = create_experiment_report(
    project=cfg["wandb_project"],
    slug="<slug>",
    decision_metrics=cfg["metrics"]["decision"],
    health_metrics=cfg["metrics"]["health"],
    question="<from plan.md>",
    falsifier="<from plan.md>",
    runs={"baseline": "<url>", "variant": "<url>"},
)
```

Record URLs and the report link in `## Runs`.

**Spawn the watcher** (optional, only when expected runtime fits in ~1 hour). `scripts/watch_runs.py` projects ETA from each run's `train/global_step` + `_runtime`, sleeps until ~70% of the slowest ETA, then enters an **adaptive** poll loop where each subsequent sleep is `0.5 × current max ETA` (floored at 60 s). The wait shrinks naturally as runs converge — typically ~3–5 polls total instead of a constant 2-minute interval. Hard `--max_wait_min` deadline. Burns ~zero LLM tokens (Python loop, not an agent).

```bash
SLUG=<slug>
LOGS=/path/to/worktree/experiments/$SLUG/logs
uv run python ${CLAUDE_PLUGIN_ROOT}/skills/wandb-driven-dev/scripts/watch_runs.py \
    $SLUG --logs_dir $LOGS \
    --target_steps <budget> --max_wait_min 60
```

Use `Bash(run_in_background=True)` so it survives between conversation turns. Watcher writes:
- `logs/05-watch.json` — live status (overwritten each poll).
- `logs/05-staged-result.md` — draft `## Result` block once all runs are terminal. Phase 5 reads this; it never edits plan.md directly.

Skip the watcher if the run budget is multi-hour (it'll hit the deadline) or if you'd rather rely solely on wandb account alerts at https://wandb.ai/settings → Alerts. Both are valid; wandb alerts also cover crash/divergence per-run.

Then exit:

> Launched. Watcher running; say `review experiment <slug>` when you're ready, or you'll see the staged verdict next time we talk.

Don't try to *block* in-conversation — Bash backgrounding and subagent lifetimes both cap below the runtime of a multi-hour training job. The watcher pattern works because it's bounded; multi-hour experiments still need wandb alerts.

### 5. Review (delegate to reviewer agent)

When the user types `review experiment <slug>`, **delegate to the `reviewer` agent** via the Agent tool. The agent:

1. Reads `experiments/<slug>/plan.md`, `experiments/<slug>/logs/05-staged-result.md` (if present), and `.claude/wandb-driven-dev.local.md`.
2. Pulls fresh wandb summaries for the experiment runs.
3. Validates that the only config difference between baseline and variant is the named dimension (`compare_configs`).
4. Drafts the `## Result` block (verdict, key numbers, against-hypothesis, merge recommendation).
5. Returns the proposed `## Result` markdown — does NOT auto-write it.

```python
# Example invocation from the main thread:
Agent(
    description=f"Review experiment {slug}",
    subagent_type="reviewer",
    prompt=(
        f"Review experiment '{slug}'. The plan is at "
        f"experiments/{slug}/plan.md (in the current worktree). "
        f"The staged result, if any, is at experiments/{slug}/logs/05-staged-result.md. "
        f"Project config is at .claude/wandb-driven-dev.local.md. "
        f"Return the proposed ## Result block as markdown — do not write it."
    ),
)
```

Then **show the proposed verdict to the user** and ask for approval before splicing into `plan.md`. After approval:

- **Pass** → splice `## Result`, push branch, `gh pr create`, link plan.md + wandb URLs. After merge, run Phase 6.
- **Fail** → splice `## Result`, run Phase 6. Optionally cherry-pick plan.md into main's `experiments/` as a record.
- **Inconclusive** → splice `## Result`, stay in worktree, propose smallest next run, loop back to Phase 1. Skip Phase 6 until terminal.

### 6. Cleanup (gate: worktree removed, branch resolved)

From the **main checkout**:

```bash
# pass — branch merged via PR
git worktree remove ../<repo>-exp/<slug>
git branch -d exp/<slug>
git push origin --delete exp/<slug>  # only if pushed

# fail / abandoned
git worktree remove ../<repo>-exp/<slug>
git branch -D exp/<slug>
git push origin --delete exp/<slug>  # only if pushed
```

If `git worktree remove` refuses, investigate uncommitted changes before `--force` — they may be unsaved log captures.

## Output capture

Training-process output is ephemeral. Tee every command into `experiments/<slug>/logs/NN-<name>.txt` (numbered for chronological replay): smoke submit, smoke run logs, baseline submit, variant submit, etc. Commit `experiments/<slug>/` at each phase boundary — the commit history is the audit trail.

## Anti-patterns (call them out)

- "I'll just run it and see" — write the falsifier first.
- "Compare to last week's numbers" — re-run the baseline on the current branch unless pinned and justified.
- "Skip smoke, I'm confident" — 10 min of smoke saves 6h of wasted GPU.
- "Change optimizer AND LR AND batch size in one variant" — that's an uncontrolled change. Either name a single dimension or make it an explicit multi-variant benchmark.
- "It looks worse, let it finish" — surface it, let the user decide.
- "Use the config's max_steps for everything" — pick the smallest budget that answers the question.

If the user insists on skipping a gate, state the trade plainly ("I can skip smoke but we won't know if the 6h run crashes on step 1 — proceed?") and require explicit override.

## Iterating on this skill

After a successful invocation, look for patterns worth lifting into `wbagent/scripts/wandb_helpers.py` (generic) or `scripts/wdd_helpers.py` (wandb-driven-dev-specific). Slow paths or missing checks belong in code; phase logic belongs here.
