---
name: wandb-reports
description: "Author and update W&B Reports — shareable dashboards combining markdown narrative with live charts pulled from W&B runs. Use when the user asks to 'create a wandb report', 'summarize this project as a report', 'build a dashboard for these runs', 'update the experiment report with the new variant', or wants a publishable comparison across runs. For raw run/trace querying without a report, use the `wbagent` skill instead."
user-invocable: true
argument-hint: "[project] [run-ids...]  (optional — agent will ask if missing)"
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# W&B Reports

Author and update W&B Reports — the v2 dashboard format that mixes markdown
prose with live PanelGrids tied to a Runset filter.

## When invoked

If invoked as a slash command, parse the arguments. Common shapes:

- `/wandb-driven-dev:wandb-reports` → ask the user what to build (project, runs, metrics, title).
- `/wandb-driven-dev:wandb-reports entity/project run_id_a run_id_b` → build a comparison report for those specific runs.
- `/wandb-driven-dev:wandb-reports update <existing_report_url>` → append a new run to an existing report.

If invoked via auto-trigger (description match), gather the same info from the conversation; ask only what's missing.

## Required context

Before writing the script, you need:

1. **`entity/project`** — where the report is created and where the runs live.
2. **Run IDs** to include — at least one. For comparisons, ≥2.
3. **Decision metrics** — the metric keys plotted first (largest panels). If
   the project has `.claude/wandb-driven-dev.local.md`, read it; otherwise ask
   the user or `probe_project()` first.
4. **Optional**: health metrics, x-axis key (default `train/global_step`),
   title, description, draft (default True).

Don't fabricate metric keys — discover them.

## Authoring a comparison report (one shot)

Use the `create_comparison_report` helper in wbagent:

```python
import sys, os
sys.path.insert(0, f"{os.environ['CLAUDE_PLUGIN_ROOT']}/skills/wbagent/scripts")
from wandb_helpers import create_comparison_report

url = create_comparison_report(
    project="ENTITY/PROJECT",
    title="Loss-scheme ablation — Apr 2026",
    runs={
        "baseline": "https://wandb.ai/ENTITY/PROJECT/runs/RUN_A",
        "variant":  "https://wandb.ai/ENTITY/PROJECT/runs/RUN_B",
    },
    decision_metrics=["val/loss", "val/accuracy"],
    health_metrics=["train/loss", "train/grad_norm"],
    description="Comparison of baseline vs variant on the loss scheme.",
    x_axis="train/global_step",
    draft=True,
)
print(f"Report: {url}")
```

Notes:
- Pass run **URLs or IDs** in `runs.values()`. The Runset filter pins exact
  IDs, so the report won't drift if tags are added later.
- `header_md` is optional — if you want a custom intro, build it as markdown
  and pass it in.
- Reports save as drafts by default; the URL is still shareable.
- The helper automatically appends a `RunComparer` panel to the PanelGrid so
  reviewers can immediately see what configs/summary values differ across
  the pinned runs. Don't strip it.

## Always include a RunComparer

Every PanelGrid this skill produces must include a `wr.RunComparer(diff_only="on")`
panel alongside the metric plots. The whole point of a comparison report is to
answer "what's different across these runs?" at a glance — the RunComparer
surfaces config + summary diffs without the reviewer having to click into each
run. The `create_comparison_report` helper does this for you; if you build a
PanelGrid by hand (custom blocks or report updates), append a RunComparer
yourself.

## Authoring from scratch (when you need custom blocks)

For non-comparison reports — overviews, post-mortems, multi-section narrative
— use the SDK directly:

> **Install requirement:** Reports use the `wandb_workspaces` package, which
> ships as an extra: `pip install 'wandb[workspaces]'` (or
> `uv pip install 'wandb[workspaces]'`). The legacy `wandb.apis.reports`
> module no longer ships `LinePlot`/`PanelGrid` and will fail at import.

```python
import os
import wandb_workspaces.reports.v2 as wr

entity = os.environ["WANDB_ENTITY"]
project = os.environ["WANDB_PROJECT"]

runset = wr.Runset(entity=entity, project=project, name="All runs")
loss_panel = wr.LinePlot(title="Loss", x="_step", y=["LOSS_KEY"])
acc_panel  = wr.BarPlot(title="Accuracy", metrics=["ACC_KEY"], orientation="v")

report = wr.Report(
    entity=entity,
    project=project,
    title="Project analysis",
    description="Auto-generated summary",
    width="fluid",
    blocks=[
        wr.H1(text="Project analysis"),
        wr.P(text="Summary of recent runs."),
        wr.H2(text="Top-line metrics"),
        wr.PanelGrid(
            runsets=[runset],
            panels=[loss_panel, acc_panel, wr.RunComparer(diff_only=True)],
        ),
        wr.H2(text="Notes"),
        wr.MarkdownBlock(text="- Bullet 1\n- Bullet 2"),
    ],
)
report.save(draft=True)
print(f"Report: {report.url}")
```

Block types worth knowing: `H1`, `H2`, `H3`, `P`, `MarkdownBlock`, `CodeBlock`,
`PanelGrid`, `Image`, `LineBreak`, `HorizontalRule`.

Panel types: `LinePlot`, `BarPlot`, `ScatterPlot`, `ScalarChart`,
`MarkdownPanel`, `CodeComparer`, `RunComparer`.

## Updating an existing report

The Reports SDK supports loading a report by URL, mutating its blocks, and
saving in place:

```python
import wandb
import wandb_workspaces.reports.v2 as wr

api = wandb.Api(timeout=60)
report = api.report("REPORT_URL_OR_PATH")

# Inspect existing blocks
for i, block in enumerate(report.blocks):
    print(i, type(block).__name__)

# Append a new section
report.blocks += [
    wr.H2(text="New variant"),
    wr.MarkdownBlock(text="Added 2026-04-30 — variant `v2` results below."),
    wr.PanelGrid(
        runsets=[wr.Runset(
            entity=report.entity, project=report.project,
            filters=f"ID in {['NEW_RUN_ID']!r}",
        )],
        panels=[
            wr.LinePlot(title="val/loss", x="train/global_step", y=["val/loss"]),
            wr.RunComparer(diff_only="on"),
        ],
    ),
]

report.save()
print(f"Updated: {report.url}")
```

If `api.report(url)` raises (older SDK versions), fall back to listing reports
on the project and matching by ID:

```python
reports = list(api.reports(path=f"{entity}/{project}"))
report = next(r for r in reports if r.id == "REPORT_ID")
```

## Pinning runs reliably

Reports are filtered, not rooted. To guarantee a report shows exactly the runs
you passed (regardless of future tags/groups), filter by run ID:

```python
runset_filter = f"ID in {['run_id_1', 'run_id_2']!r}"
runset = wr.Runset(entity=..., project=..., filters=runset_filter)
```

Tag-based filters (`tags includes 'exp/abc'`) are convenient but drift over
time. Prefer ID pinning for archival reports.

## Width caveat

The Reports SDK accepts `width="readable" | "fixed" | "fluid"` but the backend
currently drops the field — saved reports render at the project default
(typically "readable"). Harmless to set; just don't rely on it.

## Output

After saving, **always print the report URL**. Do not summarize the report
contents back to the user — they'll click through.

## Related

- For querying runs to feed a report, use the **`wbagent`** skill recipes.
- For experiment-driven workflows that auto-generate dashboards per
  experiment, use the **`wandb-driven-dev`** skill (Phase 4 calls
  `wdd_helpers.create_experiment_report` which wraps this).
