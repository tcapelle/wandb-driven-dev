# wandb-driven-dev

A Claude Code plugin that enforces **experiment-driven development** for ML
codebases. Every empirical claim is backed by a W&B run, a baseline to compare
against, and a falsifier written before the run started. Nothing merges on vibes.

## What's inside

### Skills

| Skill | Trigger | Purpose |
|---|---|---|
| `wandb-driven-dev` | `/wandb-driven-dev` (also auto-triggers on "experiment", "ablation", "is A better than B", setup/reconfigure, and W&B Report requests) | The methodology — Phases 0–6 from setup to cleanup, with project-local experiment launcher config, training entrypoint, worktree bootstrap, smoke gates, launch, ETA-aware watcher, review, and experiment report helpers. |
| `wbagent` | Auto-triggered on W&B queries | Toolkit for querying W&B runs, summaries, configs, histories, artifacts, sweeps, and reports through `wandb_helpers.py`. |

Experiment report helpers live in `wandb-driven-dev`; generic W&B Reports SDK
details come from the upstream `wbagent` reference at
`skills/wbagent/references/REPORTS.md`.

Fast count, top-k, and at-step comparison workflows use reusable query
primitives in `skills/wbagent/scripts/wandb_helpers.py`, so common W&B questions
do not require one-off Python or broad run iteration.

Project setup discovery is hardcoded in
`skills/wandb-driven-dev/scripts/setup_project.py`. It reads the local config,
uses configured decision/health metrics, samples a few recent finished runs
when run IDs are not supplied, and writes `curves` plus
`wandb_metadata.preflight` back to `.claude/wandb-driven-dev.local.md`. It
stores keys and decisions only, not run summaries or metric values. The default
path is a single selected-summary GraphQL query, so it does not materialize SDK
run objects or scan history. Use `--validate-history` only when summary
coverage is ambiguous; that slower path uses explicit `scan_history(keys=...)`,
bounded scans, and targeted sparse metric fallback.

Plot-reading workflows are hardcoded in
`skills/wandb-driven-dev/scripts/curve_analysis.py`. It assumes setup already
persisted curve step keys, then turns selected W&B curves into pandas-derived
features such as value at step, local slope, trend, and best run by value/slope.
Slope/trend use trailing rolling smoothing by default and report
noise/confidence fields so noisy endpoints do not dominate the verdict. The
analyzer has separate early-training health checks for launch stability and
progress-stage checks for slope shifts and sudden spikes.

### Agents

| Agent | When | Purpose |
|---|---|---|
| `wandb-query` | On-demand | Off-thread analysis of a W&B project or run. Frees the main thread from large query outputs. |
| `reviewer` | Spawned by the `wandb-driven-dev` skill in Phase 5 | Reads the staged result, validates numbers against fresh W&B summaries, drafts the verdict and merge recommendation. |

### Templates

`templates/wandb-driven-dev.local.md.template` — copy to your project as
`.claude/wandb-driven-dev.local.md` for per-project config (W&B entity/project,
repo launcher command, training entrypoint, default metrics, GPU budgets,
free-form notes).

## Install (local plugin)

Drop this directory into a Claude Code plugins location, or run Claude Code
pointing at it directly:

```bash
cc --plugin-dir /path/to/wandb-driven-dev
```

`wbagent` is vendored directly into `skills/wbagent/` as plain files, copied
from the upstream W&B core repository at
`services/wb_agent/src/agent_repository/context_content/production/wbagent/skills/wbagent`.
The original upstream base commit is recorded in
`skills/wbagent/.upstream-commit`, but this plugin intentionally carries local
query-helper extensions in `skills/wbagent/scripts/wandb_helpers.py`. Do not
overwrite `skills/wbagent/` with a blind upstream sync; port local improvements
to upstream manually and then reconcile the vendored copy deliberately.

## Quick start

```
/wandb-driven-dev setup
```

Claude interviews you for the W&B project, repo-specific experiment launcher
command, training entrypoint, reproduction model, GPU budgets, and
decision/health metrics, then writes
`.claude/wandb-driven-dev.local.md`. Subsequent invocations read it.

For a new experiment:

```
/wandb-driven-dev
```

Claude walks you through hypothesis → design → smoke → launch → review →
cleanup, gating each phase. Wandb runs use the `exp/<slug>` tag and
`exp-<slug>-<role>` name convention so they're trivially filterable.

## Prerequisites

- `wandb` and `pandas` Python packages on the Python you're running
- A W&B account with API key configured (`wandb login` or `WANDB_API_KEY`)
- For remote training: access to the runner, scheduler, or cluster used by the
  launcher command recorded in project config

## Project config schema

`.claude/wandb-driven-dev.local.md` (per-project, gitignored):

```markdown
---
wandb_project: entity/project
launcher:
  # Project-specific command to start/submit training; not W&B Launch.
  command: uv run python scripts/train.py
  reproduction: working_tree   # working_tree | clone | shared_fs | image
training:
  # Underlying training entrypoint used for --help flag validation.
  script: scripts/train.py
  config_dir: configs/
gpus:
  smoke: 1
  full: 8
metrics:
  decision: [val/loss, val/accuracy]
  health: [train/loss, train/grad_norm]
curves:
  # W&B default. Override per metric/namespace when the project logs semantic
  # step metrics such as train/global_step or stage_4e/epoch.
  default_step_key: _step
  metric_step_keys:
    train/*: train/global_step
    val/*: train/global_step
  candidate_step_keys: [_step, train/global_step]
wandb_metadata: {}
---

# Free-form project notes

The body of this file is read by the agents as context. Use it for things that
don't fit the structured schema: dataset quirks, known-good baseline run IDs,
project-specific gotchas.
```

## Files

```
wandb-driven-dev/
├── .claude-plugin/plugin.json
├── README.md
├── skills/
│   ├── wandb-driven-dev/
│   │   ├── SKILL.md
│   │   └── scripts/{wdd_helpers.py, setup_project.py, curve_analysis.py, create_report.py, watch_runs.py, bootstrap_experiment.sh}
│   ├── wbagent/         # vendored copy of upstream wbagent; see .upstream-commit
│   │   ├── SKILL.md
│   │   ├── scripts/*.py
│   │   └── references/*.md
├── agents/{wandb-query.md, reviewer.md}
└── templates/wandb-driven-dev.local.md.template
```
