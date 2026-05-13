#!/usr/bin/env python3
"""Fast, bounded W&B queries for wandb-driven-dev workflows.

This script hardcodes the query shapes that should not be left to ad hoc model
code during experiment review:

- exact server-side run counts
- top-k run lookup with selected summary/config columns
- step comparisons for a small set of pinned runs

Every command prints JSON and avoids broad SDK iteration.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_WBAGENT_SCRIPTS = _HERE.parent.parent / "wbagent" / "scripts"
sys.path.insert(0, str(_WBAGENT_SCRIPTS))
from wandb_helpers import (  # noqa: E402
    build_filters,
    compare_runs_at_step,
    count_runs,
    fetch_run_summaries as _fetch_run_summaries,
    fetch_runs as _fetch_runs,
    get_api,
    scan_history_until_step,
)


def fast_fetch_runs(
    api,
    project_path: str,
    metric_keys: list[str],
    filters: dict,
    config_keys: list[str] | None = None,
    order: str = "-created_at",
    limit: int = 50,
    per_page: int = 50,
) -> list[dict]:
    """Backward-compatible wrapper around `wandb_helpers.fetch_runs`."""
    return _fetch_runs(
        api,
        project_path,
        metric_keys=metric_keys,
        filters=filters,
        config_keys=config_keys,
        order=order,
        limit=limit,
        per_page=per_page,
    )


def fast_fetch_run_summaries(
    api,
    project_path: str,
    run_ids: list[str],
    summary_keys: list[str] | None = None,
    order: str = "-created_at",
    per_page: int = 50,
) -> list[dict]:
    """Backward-compatible wrapper around `wandb_helpers.fetch_run_summaries`."""
    return _fetch_run_summaries(
        api,
        project_path,
        run_ids=run_ids,
        summary_keys=summary_keys,
        order=order,
        per_page=per_page,
    )


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _add_common_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--filter", action="append", default=[], help="Filter expression, repeatable: key=value, key<value, key>=value")
    parser.add_argument("--filters-json", help="JSON object of W&B filters")
    parser.add_argument("--state", default="finished", help="Default state filter; use empty string for none")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    top_p = subparsers.add_parser("top", help="Fetch top/bottom runs by a summary metric")
    top_p.add_argument("project")
    top_p.add_argument("--metric", required=True, help="Metric used for sorting")
    top_p.add_argument("--summary", default="", help="Extra summary metrics, comma-separated")
    top_p.add_argument("--config", default="", help="Config keys to include, comma-separated")
    top_p.add_argument("--limit", type=int, default=10)
    top_p.add_argument("--order", choices=["asc", "desc"], default="asc")
    _add_common_filters(top_p)

    count_p = subparsers.add_parser("count", help="Exact count for matching runs")
    count_p.add_argument("project")
    _add_common_filters(count_p)

    compare_p = subparsers.add_parser("compare-step", help="Compare run histories at a target step")
    compare_p.add_argument("project")
    compare_p.add_argument("--runs", required=True, help="Run IDs, comma-separated")
    compare_p.add_argument("--step", required=True, type=int)
    compare_p.add_argument("--metrics", required=True, help="History metrics, comma-separated")
    compare_p.add_argument("--step-key", default="_step")
    compare_p.add_argument("--max-rows", type=int)

    args = parser.parse_args(argv)
    start = time.perf_counter()
    api = get_api()

    if args.command == "top":
        filters = build_filters(args.filter, args.filters_json, args.state or None)
        metric_keys = list(dict.fromkeys([args.metric, *_split_csv(args.summary)]))
        order = f"{'+' if args.order == 'asc' else '-'}summary_metrics.{args.metric}"
        result = fast_fetch_runs(
            api,
            args.project,
            metric_keys=metric_keys,
            filters=filters,
            config_keys=_split_csv(args.config),
            order=order,
            limit=args.limit,
            per_page=args.limit,
        )
    elif args.command == "count":
        filters = build_filters(args.filter, args.filters_json, args.state or None)
        result = count_runs(api, args.project, filters)
    else:
        result = compare_runs_at_step(
            api,
            args.project,
            run_ids=_split_csv(args.runs),
            step=args.step,
            metrics=_split_csv(args.metrics),
            step_key=args.step_key,
            max_rows=args.max_rows,
        )

    print(json.dumps({"latency_s": round(time.perf_counter() - start, 4), "result": result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
