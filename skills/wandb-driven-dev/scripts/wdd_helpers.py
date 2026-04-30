"""wandb-driven-dev helpers: project config IO and the experiment Reports wrapper.

Generic wandb work lives in `../../wbagent/scripts/wandb_helpers.py`.
This module only contains things tied to the wandb-driven-dev experiment workflow:

- Project config schema + read/write at `.claude/wandb-driven-dev.local.md`.
  The file is YAML frontmatter (structured fields) followed by a markdown body
  (free-form project notes the agents read verbatim).
- `create_experiment_report`: thin wrapper around wbagent's
  `create_comparison_report` that adds slug, falsifier, and per-role run links
  to the dashboard header.
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
# Experiment Report (wandb-driven-dev wrapper around wbagent's generic primitive)
# ---------------------------------------------------------------------------

def create_experiment_report(
    project: str,
    slug: str,
    decision_metrics: list[str],
    runs: dict[str, str],
    health_metrics: list[str] | None = None,
    question: str | None = None,
    falsifier: str | None = None,
    date: str | None = None,
    x_axis: str = "train/global_step",
    draft: bool = True,
) -> str:
    """Create a wandb Report for a wandb-driven-dev experiment.

    Builds the standard markdown header (slug, date, question, falsifier,
    role-labelled run links) and delegates panel + Runset construction to
    wbagent's `create_comparison_report`.

    Args:
        project: "entity/project".
        slug: Experiment slug (matches the `exp/<slug>` tag).
        decision_metrics: Decision metric keys from plan.md `## Metrics`.
        runs: Mapping of role -> wandb run URL/id.
        health_metrics: Health metric keys from plan.md `## Metrics`.
        question: One-line experiment question copied from plan.md.
        falsifier: One-line falsifier copied from plan.md.
        date: Date string for header (default: today UTC).
        x_axis: x-axis key for every panel.
        draft: Save as draft (default True).

    Returns:
        URL of the saved report.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "wbagent" / "scripts"))
    from wandb_helpers import create_comparison_report

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
        lines.append(f"- `{role}` → {url}")

    return create_comparison_report(
        project=project,
        title=f"Experiment {slug}",
        runs=runs,
        decision_metrics=decision_metrics,
        health_metrics=health_metrics,
        description=question or f"Dashboard for experiment `{slug}`.",
        header_md="\n\n".join(lines),
        x_axis=x_axis,
        draft=draft,
    )
