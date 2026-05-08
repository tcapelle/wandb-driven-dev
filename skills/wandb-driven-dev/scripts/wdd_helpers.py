"""wandb-driven-dev helpers: project config IO and experiment Reports.

Generic wandb querying lives in `../../wbagent/scripts/wandb_helpers.py`.
This module only contains things tied to the wandb-driven-dev experiment workflow:

- Project config schema + read/write at `.claude/wandb-driven-dev.local.md`.
  The file is YAML frontmatter (structured fields) followed by a markdown body
  (free-form project notes the agents read verbatim).
- Experiment gates for config-based lookup, required metric checks, run
  resolution, runtime estimates, and training flag validation.
- Report builders for experiment dashboards and run comparisons.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(".claude/wandb-driven-dev.local.md")
_FRONTMATTER_DELIM = "---"


# ---------------------------------------------------------------------------
# Project config
# ---------------------------------------------------------------------------

def default_config() -> dict[str, Any]:
    """Schema for the structured frontmatter of `.claude/wandb-driven-dev.local.md`."""
    return {
        "wandb_project": "",
        "launcher": {
            "command": "",
            "reproduction": "working_tree",  # working_tree | clone | shared_fs | image
        },
        "training": {
            "script": "",
            "config_dir": "",
        },
        "gpus": {
            "smoke": 1,
            "full": 1,
        },
        "metrics": {
            "decision": [],
            "health": [],
        },
    }


def _split_frontmatter(text: str) -> tuple[str, str]:
    """Split YAML frontmatter from the markdown body.

    Returns (frontmatter_text, body_text). Raises ValueError if no frontmatter.
    """
    if not text.startswith(_FRONTMATTER_DELIM):
        raise ValueError(
            f"Config file must start with YAML frontmatter delimited by '{_FRONTMATTER_DELIM}'"
        )
    lines = text.splitlines(keepends=True)
    end_idx = None
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == _FRONTMATTER_DELIM:
            end_idx = i
            break
    if end_idx is None:
        raise ValueError("Config file frontmatter is not closed (no trailing '---')")
    fm = "".join(lines[1:end_idx])
    body = "".join(lines[end_idx + 1:])
    return fm, body


def read_config(path: Path | str = CONFIG_PATH) -> dict[str, Any] | None:
    """Read the project config. Returns None if the file doesn't exist.

    The returned dict contains the parsed frontmatter plus a `_notes` key
    holding the markdown body (stripped of leading/trailing whitespace).
    """
    import yaml

    p = Path(path)
    if not p.exists():
        return None
    fm, body = _split_frontmatter(p.read_text())
    cfg = yaml.safe_load(fm) or {}
    cfg["_notes"] = body.strip()
    return cfg


def write_config(
    cfg: dict[str, Any],
    notes: str = "",
    path: Path | str = CONFIG_PATH,
) -> None:
    """Write the project config preserving key order from `default_config`.

    `notes` becomes the markdown body (free-form context for the agents).
    Any `_notes` key in `cfg` is preferred over the `notes` argument.
    """
    import yaml

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    body = cfg.pop("_notes", notes).strip()
    fm = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False).rstrip()
    contents = f"{_FRONTMATTER_DELIM}\n{fm}\n{_FRONTMATTER_DELIM}\n"
    if body:
        contents += "\n" + body + "\n"
    p.write_text(contents)


# ---------------------------------------------------------------------------
# Experiment gate helpers
# ---------------------------------------------------------------------------

def _load_fetch_runs() -> Any:
    """Import upstream wbagent's `fetch_runs` without requiring caller sys.path setup."""
    import sys

    scripts_dir = Path(__file__).resolve().parent.parent.parent / "wbagent" / "scripts"
    scripts_dir_s = str(scripts_dir)
    if scripts_dir_s not in sys.path:
        sys.path.insert(0, scripts_dir_s)
    from wandb_helpers import fetch_runs

    return fetch_runs


def _load_scan_history() -> Any:
    """Import upstream wbagent's `scan_history` without requiring caller sys.path setup."""
    import sys

    scripts_dir = Path(__file__).resolve().parent.parent.parent / "wbagent" / "scripts"
    scripts_dir_s = str(scripts_dir)
    if scripts_dir_s not in sys.path:
        sys.path.insert(0, scripts_dir_s)
    from wandb_helpers import scan_history

    return scan_history


def _config_get(config: dict[str, Any], key: str) -> Any:
    """Read a W&B config key, supporting both flat and dotted nested forms."""
    if key in config:
        return config[key]

    cur: Any = config
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def find_runs_by_config(
    api: Any,
    project: str,
    config_filters: dict[str, Any],
    metric_keys: list[str] | None = None,
    extra_config_keys: list[str] | None = None,
    state: str | None = "finished",
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Find runs matching exact or operator-based config values.

    Wrapper around upstream `fetch_runs` that prefixes config keys with
    `config.` and includes the filtered config keys in the output rows.
    """
    fetch_runs = _load_fetch_runs()

    filters: dict[str, Any] = {}
    for k, v in config_filters.items():
        filters[k if k.startswith("config.") else f"config.{k}"] = v
    if state is not None:
        filters["state"] = state

    config_keys = list({k.removeprefix("config.") for k in config_filters})
    if extra_config_keys:
        config_keys.extend(k for k in extra_config_keys if k not in config_keys)

    rows = fetch_runs(
        api,
        project,
        metric_keys=metric_keys or [],
        config_keys=config_keys,
        filters=filters,
        limit=limit,
    )

    dotted_keys = [key for key in config_keys if "." in key]
    if not dotted_keys:
        return rows

    run_ids = [row.get("name") for row in rows if row.get("name")]
    if not run_ids:
        return rows

    runs_by_id = {
        run.id: run
        for run in api.runs(
            project,
            filters={"name": {"$in": run_ids}},
            per_page=min(len(run_ids), 1000),
            include_sweeps=False,
        )[: len(run_ids)]
    }
    for row in rows:
        run = runs_by_id.get(row.get("name"))
        if not run:
            continue
        for key in dotted_keys:
            row[f"config.{key}"] = _config_get(run.config, key)
    return rows


def verify_required_metrics(
    api: Any,
    run_paths: list[str],
    required: list[str],
) -> dict[str, list[str]]:
    """Confirm every required metric appears on each run summary or history."""
    if not required:
        return {run_path: [] for run_path in run_paths}

    results: dict[str, list[str]] = {}
    scan_history = _load_scan_history()
    for run_path in run_paths:
        run = api.run(run_path)
        missing = [
            metric
            for metric in required
            if metric not in run.summary_metrics or run.summary_metrics[metric] is None
        ]
        if missing:
            rows = scan_history(run, keys=missing, max_rows=10_000)
            seen = {
                key
                for row in rows
                for key in missing
                if row.get(key) is not None
            }
            missing = [metric for metric in missing if metric not in seen]
        results[run_path] = missing
    return results


def find_run_by_name(
    api: Any,
    project: str,
    name: str,
    timeout_s: int = 120,
    poll_interval_s: int = 10,
) -> Any | None:
    """Find a W&B run by exact display name, polling until it appears."""
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        runs = api.runs(
            project,
            filters={"display_name": name},
            per_page=1,
            include_sweeps=False,
        )
        try:
            return next(iter(runs))
        except StopIteration:
            pass
        time.sleep(poll_interval_s)
    return None


def runtime_estimate(
    api: Any,
    project: str,
    name_pattern: str,
    target_steps: int,
    sample: int = 5,
    min_steps: int | None = None,
) -> dict[str, Any] | None:
    """Estimate wall-clock for target steps from prior finished runs."""
    if min_steps is None:
        min_steps = max(target_steps // 10, 1000)

    runs = api.runs(
        project,
        filters={"state": "finished", "display_name": {"$regex": name_pattern}},
        order="-created_at",
        per_page=sample,
        include_sweeps=False,
    )
    samples: list[tuple[float, int]] = []
    excluded = 0
    for run in runs[:sample]:
        runtime = run.summary_metrics.get("_runtime")
        steps = run.summary_metrics.get("train/global_step")
        if not (runtime and steps and steps > 0):
            continue
        if steps < min_steps:
            excluded += 1
            continue
        samples.append((runtime / 3600.0, int(steps)))

    if not samples:
        return None

    hours_per_step = [hours / steps for hours, steps in samples]
    target_low = target_steps * min(hours_per_step)
    target_high = target_steps * max(hours_per_step)

    return {
        "runs_used": len(samples),
        "excluded_short_runs": excluded,
        "min_steps": min_steps,
        "samples": samples,
        "target_hours_low": round(target_low, 2),
        "target_hours_high": round(target_high, 2),
        "notes": "Linear extrapolation from per-step throughput; assumes same GPU count.",
    }


def validate_flags(
    script: str | list[str],
    flags: list[str],
    timeout_s: int = 30,
) -> dict[str, Any]:
    """Check that each flag in `flags` is recognized by `<script> --help`."""
    import re as _re
    import shlex
    import subprocess

    cmd = shlex.split(script) if isinstance(script, str) else list(script)
    cmd.append("--help")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    help_text = (result.stdout or "") + (result.stderr or "")

    present, missing = [], []
    for raw in flags:
        flag = raw.lstrip("-")
        if _re.search(rf"--{_re.escape(flag)}\b", help_text):
            present.append(raw)
        else:
            missing.append(raw)
    return {"present": present, "missing": missing, "help_text": help_text}


# ---------------------------------------------------------------------------
# Experiment reports
# ---------------------------------------------------------------------------

def _run_id(value: str) -> str:
    """Extract a W&B run ID from a run URL, entity/project/run_id path, or raw ID."""
    from urllib.parse import urlparse

    text = value.strip().rstrip("/")
    parsed = urlparse(text)
    path = parsed.path if parsed.scheme and parsed.netloc else text
    return path.rstrip("/").rsplit("/", 1)[-1]


def _run_comparer(wr: Any) -> Any:
    """Construct a RunComparer across SDK versions."""
    if not hasattr(wr, "RunComparer"):
        raise RuntimeError(
            "The installed W&B Reports SDK does not expose RunComparer. "
            "Upgrade wandb with the workspaces extra."
        )
    try:
        return wr.RunComparer(diff_only=True)
    except TypeError:
        return wr.RunComparer(diff_only="on")


def _md_table_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def create_comparison_report(
    project: str,
    title: str,
    runs: dict[str, str],
    decision_metrics: list[str],
    health_metrics: list[str] | None = None,
    description: str | None = None,
    header_md: str | None = None,
    focused_table_md: str | None = None,
    x_axis: str = "train/global_step",
    draft: bool = True,
) -> str:
    """Create a pinned W&B Report comparing experiment runs.

    This lives in wandb-driven-dev because comparison dashboards are part of
    the experiment workflow. wbagent remains responsible for querying run data.

    Args:
        project: "entity/project".
        title: Report title.
        runs: Mapping of label -> wandb run URL, entity/project/run_id path, or run ID.
        decision_metrics: Metric keys plotted first.
        health_metrics: Additional health metric keys.
        description: Optional report description.
        header_md: Optional markdown block before the panel grid.
        focused_table_md: Optional markdown table with selected config/summary
            columns for the runs.
        x_axis: x-axis key for every line plot.
        draft: Save as draft.

    Returns:
        URL of the saved report.
    """
    try:
        from wandb.apis import reports as wr
        import wandb_workspaces.expr as expr
    except ImportError as e:
        raise ImportError(
            "Reports require wandb with the workspaces extra. Install with "
            "`pip install 'wandb[workspaces]'` or `uv pip install 'wandb[workspaces]'`."
        ) from e

    entity, proj = project.split("/", 1)
    run_ids = [_run_id(url_or_id) for url_or_id in runs.values()]

    if header_md is None:
        lines = ["**Decision metrics:** " + ", ".join(f"`{m}`" for m in decision_metrics)]
        if health_metrics:
            lines.append("**Health metrics:** " + ", ".join(f"`{m}`" for m in health_metrics))
        lines.append("**Runs:**")
        for label, url_or_id in runs.items():
            lines.append(f"- `{label}` -> {url_or_id}")
        header_md = "\n\n".join(lines)

    panels: list[Any] = [wr.LinePlot(title=m, x=x_axis, y=[m]) for m in decision_metrics]
    panels += [wr.LinePlot(title=m, x=x_axis, y=[m]) for m in (health_metrics or [])]
    panels.append(_run_comparer(wr))

    runset = wr.Runset(
        entity=entity,
        project=proj,
        name="Selected runs",
        filters=[expr.Metric("name").isin(run_ids)],
    )

    blocks = [
        wr.H1(text=title),
        (getattr(wr, "MarkdownBlock", None) or wr.P)(text=header_md),
    ]
    if focused_table_md:
        blocks.append((getattr(wr, "MarkdownBlock", None) or wr.P)(text=focused_table_md))
    blocks.append(wr.PanelGrid(runsets=[runset], panels=panels))

    report = wr.Report(
        entity=entity,
        project=proj,
        title=title,
        description=description or title,
        width="fluid",
        blocks=blocks,
    )
    report.save(draft=draft)
    return report.url


def create_experiment_report(
    project: str,
    slug: str,
    decision_metrics: list[str],
    runs: dict[str, str],
    health_metrics: list[str] | None = None,
    question: str | None = None,
    falsifier: str | None = None,
    report_columns: list[str] | None = None,
    report_column_values: dict[str, dict[str, Any]] | None = None,
    date: str | None = None,
    x_axis: str = "train/global_step",
    draft: bool = True,
) -> str:
    """Create a wandb Report for a wandb-driven-dev experiment.

    Builds the standard markdown header (slug, date, question, falsifier,
    role-labelled run links) and delegates panel + Runset construction to
    `create_comparison_report`.

    Args:
        project: "entity/project".
        slug: Experiment slug (matches the `exp/<slug>` tag).
        decision_metrics: Decision metric keys from plan.md `## Metrics`.
        runs: Mapping of role -> wandb run URL/id.
        health_metrics: Health metric keys from plan.md `## Metrics`.
        question: One-line experiment question copied from plan.md.
        falsifier: One-line falsifier copied from plan.md.
        report_columns: Selected config/summary columns to show in a focused
            comparison table.
        report_column_values: Mapping of role -> selected column values.
        date: Date string for header (default: today UTC).
        x_axis: x-axis key for every panel.
        draft: Save as draft (default True).

    Returns:
        URL of the saved report.
    """
    date = date or _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")

    lines = [
        f"**Slug:** `{slug}` &nbsp;·&nbsp; **Date:** {date}",
        f"**Filter:** `tag:exp/{slug}` (smokes excluded)",
    ]
    if question:
        lines.append(f"**Question:** {question}")
    if falsifier:
        lines.append(f"**Falsifier:** {falsifier}")
    lines.append("**Decision metrics:** " + ", ".join(f"`{m}`" for m in decision_metrics))
    if health_metrics:
        lines.append("**Health metrics:** " + ", ".join(f"`{m}`" for m in health_metrics))
    lines.append("**Runs:**")
    for role, url in runs.items():
        lines.append(f"- `{role}` -> {url}")

    focused_table_md = None
    if report_columns:
        table_lines = [
            "**Focused columns:** "
            + ", ".join(f"`{column}`" for column in report_columns),
            "",
            "| Run | " + " | ".join(_md_table_cell(column) for column in report_columns) + " |",
            "|---|" + "---|" * len(report_columns),
        ]
        values = report_column_values or {}
        for role in runs:
            row_values = values.get(role, {})
            cells = [role, *[_md_table_cell(row_values.get(column, "-")) for column in report_columns]]
            table_lines.append("| " + " | ".join(cells) + " |")
        focused_table_md = "\n".join(table_lines)

    return create_comparison_report(
        project=project,
        title=f"Experiment {slug}",
        runs=runs,
        decision_metrics=decision_metrics,
        health_metrics=health_metrics,
        description=question or f"Dashboard for experiment `{slug}`.",
        header_md="\n\n".join(lines),
        focused_table_md=focused_table_md,
        x_axis=x_axis,
        draft=draft,
    )
