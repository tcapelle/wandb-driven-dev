# wandb-driven-dev

A Claude Code plugin that enforces **experiment-driven development** for ML
codebases. Every empirical claim is backed by a W&B run, a baseline to compare
against, and a falsifier written before the run started. Nothing merges on vibes.

## What's inside

### Skills

| Skill | Trigger | Purpose |
|---|---|---|
| `wandb-driven-dev` | `/wandb-driven-dev` (also auto-triggers on "experiment", "ablation", "is A better than B", and W&B Report requests) | The methodology — Phases 0–6 from setup to cleanup, with config, worktree bootstrap, smoke gates, launch, ETA-aware watcher, review, and experiment report helpers. |
| `wbagent` | Auto-triggered on W&B/Weave queries | Toolkit for querying training runs, traces, evaluations; relaunching runs; submitting new jobs to a Launch queue. |

Experiment report helpers live in `wandb-driven-dev`; generic W&B Reports SDK
details come from the upstream `wbagent` reference at
`skills/wbagent/references/REPORTS.md`.

### Agents

| Agent | When | Purpose |
|---|---|---|
| `wandb-query` | On-demand | Off-thread analysis of a W&B project / run / Weave evaluation. Frees the main thread from large query outputs. |
| `reviewer` | Spawned by the `wandb-driven-dev` skill in Phase 5 | Reads the staged result, validates numbers against fresh W&B summaries, drafts the verdict and merge recommendation. |

### Templates

`templates/wandb-driven-dev.local.md.template` — copy to your project as
`.claude/wandb-driven-dev.local.md` for per-project config (W&B entity/project,
launcher command, default metrics, GPU budgets, free-form notes).

## Install (local plugin)

Drop this directory into a Claude Code plugins location, or run Claude Code
pointing at it directly:

```bash
cc --plugin-dir /path/to/wandb-driven-dev
```

`wbagent` is vendored directly into `skills/wbagent/` as plain files, copied
from the upstream W&B core repository at
`services/wb_agent/src/agent_repository/context_content/production/wbagent/skills/wbagent`.
The upstream commit the files were synced from is recorded in
`skills/wbagent/.upstream-commit`.

`wbagent` updates regularly upstream. To manually pull the latest version from
`wandb/core`:

```bash
scripts/update-wbagent.sh
git diff -- skills/wbagent      # review the change
git add skills/wbagent          # stage when you're happy
```

The script does a shallow sparse clone of `wandb/core`, mirrors the upstream
skill directory into `skills/wbagent/`, and refreshes `.upstream-commit`. The
plugin keeps using the previously committed `wbagent` until you stage and commit
the new files.

## Quick start

```
/wandb-driven-dev setup
```

Claude interviews you for the W&B project, launcher command, training script
location, GPU budgets, and decision/health metrics, then writes
`.claude/wandb-driven-dev.local.md`. Subsequent invocations read it.

For a new experiment:

```
/wandb-driven-dev
```

Claude walks you through hypothesis → design → smoke → launch → review →
cleanup, gating each phase. Wandb runs use the `exp/<slug>` tag and
`exp-<slug>-<role>` name convention so they're trivially filterable.

## Prerequisites

- `wandb` and `weave` Python packages on the Python you're running
- A W&B account with API key configured (`wandb login` or `WANDB_API_KEY`)
- For Launch: at least one queue configured for your entity

## Project config schema

`.claude/wandb-driven-dev.local.md` (per-project, gitignored):

```markdown
---
wandb_project: entity/project
launcher:
  command: uv run python scripts/train.py
  reproduction: working_tree   # working_tree | clone | shared_fs | image
training:
  script: scripts/train.py
  config_dir: configs/
gpus:
  smoke: 1
  full: 8
metrics:
  decision: [val/loss, val/accuracy]
  health: [train/loss, train/grad_norm]
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
│   │   └── scripts/{wdd_helpers.py, create_report.py, watch_runs.py, bootstrap_experiment.sh}
│   ├── wbagent/         # vendored copy of upstream wbagent; see .upstream-commit
│   │   ├── SKILL.md
│   │   ├── scripts/*.py
│   │   └── references/*.md
├── agents/{wandb-query.md, reviewer.md}
├── scripts/update-wbagent.sh  # resync skills/wbagent from wandb/core
└── templates/wandb-driven-dev.local.md.template
```
