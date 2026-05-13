from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
WANDB_HELPERS_PATH = ROOT / "skills" / "wbagent" / "scripts" / "wandb_helpers.py"

DRIVAERML_PROJECT = "milieu/drivaerml"
DECISION_METRICS = [
    "val/volume_rel_l2",
    "val/surface_rel_l2",
    "val/surface_pressure_rel_l2",
    "val/u_rel_l2",
]
HEALTH_METRICS = [
    "train/loss",
    "train/global_samples_per_sec",
    "val/loss",
]
ALL_METRICS = DECISION_METRICS + HEALTH_METRICS


def load_wandb_helpers():
    spec = importlib.util.spec_from_file_location("wandb_helpers_under_test", WANDB_HELPERS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wandb_helpers = load_wandb_helpers()


class WandbHelpersPureTests(unittest.TestCase):
    def test_nested_get_supports_flat_and_nested_wandb_config_shapes(self):
        config = {
            "training.script": "scripts/train_drivaerml.py",
            "model": {"depth": 16, "width": 256},
            "optimizer": {"lr": 0.001},
        }

        self.assertEqual(
            wandb_helpers.nested_get(config, "training.script"),
            "scripts/train_drivaerml.py",
        )
        self.assertEqual(wandb_helpers.nested_get(config, "model.depth"), 16)
        self.assertEqual(wandb_helpers.nested_get(config, "optimizer.lr"), 0.001)
        self.assertIsNone(wandb_helpers.nested_get(config, "optimizer.weight_decay"))

    def test_build_filters_parses_scalars_and_merges_ranges(self):
        filters = wandb_helpers.build_filters(
            [
                "config.max_steps=20000",
                "created_at>=2026-05-01",
                "created_at<2026-05-08",
                "config.model.model_class=abupt",
            ],
            default_state="finished",
        )

        self.assertEqual(
            filters,
            {
                "state": "finished",
                "config.max_steps": 20_000,
                "created_at": {"$gte": "2026-05-01", "$lt": "2026-05-08"},
                "config.model.model_class": "abupt",
            },
        )


def require_online_wandb():
    if not os.environ.get("WANDB_API_KEY"):
        raise unittest.SkipTest("WANDB_API_KEY is required for online W&B tests")
    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("Install wandb to run online W&B tests") from exc


class OnlineWandbHelpersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_online_wandb()
        cls.api = wandb_helpers.get_api()
        cls.project = DRIVAERML_PROJECT
        cls.finished_runs = cls.api.runs(
            cls.project,
            filters={"state": "finished"},
            order="-created_at",
            per_page=10,
            include_sweeps=False,
        )[:10]
        if not cls.finished_runs:
            raise unittest.SkipTest(f"No finished runs found in {cls.project}")
        cls.latest_run = cls.finished_runs[0]

    def test_get_api_sets_large_project_timeout(self):
        self.assertEqual(getattr(self.api, "_timeout", None), 120)

    def test_probe_project_discovers_drivaerml_metrics_and_config(self):
        probe = wandb_helpers.probe_project(self.api, self.project, sample_size=3)

        self.assertEqual(probe["path"], self.project)
        self.assertGreaterEqual(probe["sample_metric_count"], len(ALL_METRICS))
        for metric in ALL_METRICS:
            self.assertIn(metric, probe["sample_metric_keys"])
        self.assertIn("batch_size", probe["sample_config_keys"])
        self.assertIn("model", probe["sample_config_keys"])
        self.assertTrue(probe["has_step_history"])
        self.assertIn(probe["recommended_per_page"], {10, 50, 100})

    def test_fetch_runs_uses_live_graphql_metric_selection(self):
        import requests

        original_post = requests.post
        captured_payloads: list[dict] = []

        def post_with_wandb_auth(url, **kwargs):
            headers = dict(kwargs.pop("headers", {}) or {})
            headers.pop("Authorization", None)
            headers.setdefault("Content-Type", "application/json")
            kwargs["headers"] = headers
            kwargs["auth"] = ("api", self.api.api_key)
            captured_payloads.append(kwargs["json"])
            return original_post(url, **kwargs)

        with patch.object(requests, "post", side_effect=post_with_wandb_auth):
            rows = wandb_helpers.fetch_runs(
                self.api,
                self.project,
                metric_keys=["val/volume_rel_l2", "train/loss"],
                filters={"state": "finished"},
                config_keys=["batch_size", "model"],
                limit=2,
                per_page=2,
            )

        self.assertTrue(rows)
        payload = captured_payloads[0]
        self.assertIn(
            'summaryMetrics(keys: ["val/volume_rel_l2", "train/loss"])',
            payload["query"],
        )
        self.assertEqual(payload["variables"]["entity"], "milieu")
        self.assertEqual(payload["variables"]["project"], "drivaerml")
        self.assertEqual(payload["variables"]["perPage"], 2)
        for row in rows:
            self.assertEqual(row["state"], "finished")
            self.assertIn("val/volume_rel_l2", row)
            self.assertIn("train/loss", row)
            self.assertIn("config.batch_size", row)
            self.assertIn("config.model", row)

    def test_scan_history_reads_live_rows_for_drivaerml_metric(self):
        metric = next(
            (
                candidate
                for candidate in ALL_METRICS
                if self.latest_run.summary_metrics.get(candidate) is not None
            ),
            None,
        )
        if metric is None:
            raise unittest.SkipTest("Latest run has none of the configured metrics in summary")

        rows = wandb_helpers.scan_history(self.latest_run, keys=[metric], max_rows=5)

        self.assertLessEqual(len(rows), 5)
        self.assertTrue(rows)
        self.assertTrue(any(row.get(metric) is not None for row in rows))

    def test_runs_to_dataframe_extracts_selected_columns_from_live_runs(self):
        rows = wandb_helpers.runs_to_dataframe(
            self.finished_runs,
            limit=3,
            metric_keys=["val/volume_rel_l2", "train/loss"],
            config_keys=["batch_size", "max_steps"],
        )

        self.assertEqual(len(rows), min(3, len(self.finished_runs)))
        for row in rows:
            self.assertEqual(set(row) - {"id", "name", "state", "created_at"}, {
                "config.batch_size",
                "config.max_steps",
                "val/volume_rel_l2",
                "train/loss",
            })
            self.assertEqual(row["state"], "finished")
            self.assertIsNotNone(row["val/volume_rel_l2"])
            self.assertIsNotNone(row["train/loss"])

    def test_diagnose_run_uses_live_history_for_configured_metric(self):
        train_key = None
        for candidate in ALL_METRICS:
            if self.latest_run.summary_metrics.get(candidate) is None:
                continue
            rows = wandb_helpers.scan_history(self.latest_run, keys=[candidate], max_rows=5)
            if any(row.get(candidate) is not None for row in rows):
                train_key = candidate
                break
        if train_key is None:
            raise unittest.SkipTest("Latest run has no configured metric with readable history")

        diagnostics = wandb_helpers.diagnose_run(
            self.latest_run,
            train_key=train_key,
            val_key=None,
            max_steps=200,
        )

        self.assertNotIn("error", diagnostics)
        self.assertGreater(diagnostics["total_steps"], 0)
        self.assertIsNotNone(diagnostics["final_value"])
        self.assertIn("min_value", diagnostics)

    def test_compare_configs_finds_live_model_budget_difference(self):
        baseline = None
        variant = None
        seen: dict[tuple[int | None, int | None], object] = {}
        for run in self.finished_runs:
            model = run.config.get("model") or {}
            key = (
                model.get("surface_anchor_points"),
                model.get("volume_anchor_points"),
            )
            if key in seen:
                continue
            if baseline is None:
                baseline = run
                seen[key] = run
                continue
            variant = run
            break

        if baseline is None or variant is None:
            raise unittest.SkipTest("Need two live runs with different model budgets")

        diffs = wandb_helpers.compare_configs(baseline, variant, keys=["model"])

        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["key"], "model")
        self.assertIn(baseline.name, diffs[0])
        self.assertIn(variant.name, diffs[0])


if __name__ == "__main__":
    unittest.main()
