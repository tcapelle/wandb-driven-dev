from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
WDD_HELPERS_PATH = ROOT / "skills" / "wandb-driven-dev" / "scripts" / "wdd_helpers.py"

DRIVAERML_CONFIG = {
    "wandb_project": "milieu/drivaerml",
    "launcher": {
        "family": "kubernetes",
        "command": "uv run python k8s/launch.py",
        "reproduction": "clone",
    },
    "training": {
        "script": "scripts/train_drivaerml.py",
        "config_dir": "configs/drivaerml",
    },
    "gpus": {
        "smoke": 1,
        "full": 8,
    },
    "metrics": {
        "decision": [
            "val/volume_rel_l2",
            "val/surface_rel_l2",
            "val/surface_pressure_rel_l2",
            "val/u_rel_l2",
        ],
        "health": [
            "train/loss",
            "train/global_samples_per_sec",
            "val/loss",
        ],
    },
    "curves": {
        "default_step_key": "_step",
        "metric_step_keys": {
            "train/*": "train/global_step",
            "val/*": "train/global_step",
        },
        "candidate_step_keys": ["_step", "train/global_step"],
    },
    "_notes": (
        "Smoke locally on 1 GPU; full runs via k8s/launch.py with code cloned "
        "from origin (clean tree + pushed commit required before each launch)."
    ),
}
ALL_METRICS = DRIVAERML_CONFIG["metrics"]["decision"] + DRIVAERML_CONFIG["metrics"]["health"]


def load_wdd_helpers():
    spec = importlib.util.spec_from_file_location("wdd_helpers_under_test", WDD_HELPERS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wdd_helpers = load_wdd_helpers()


def require_online_wandb():
    if not os.environ.get("WANDB_API_KEY"):
        raise unittest.SkipTest("WANDB_API_KEY is required for online W&B tests")
    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("Install wandb to run online W&B tests") from exc


class WddPureHelperTests(unittest.TestCase):
    def test_config_roundtrip_preserves_drivaerml_frontmatter_and_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wandb-driven-dev.local.md"
            wdd_helpers.write_config(DRIVAERML_CONFIG.copy(), path=path)

            loaded = wdd_helpers.read_config(path)

        self.assertEqual(loaded["wandb_project"], "milieu/drivaerml")
        self.assertEqual(loaded["launcher"]["family"], "kubernetes")
        self.assertEqual(loaded["launcher"]["command"], "uv run python k8s/launch.py")
        self.assertEqual(loaded["training"]["script"], "scripts/train_drivaerml.py")
        self.assertEqual(loaded["gpus"]["full"], 8)
        self.assertEqual(loaded["metrics"]["decision"], DRIVAERML_CONFIG["metrics"]["decision"])
        self.assertEqual(loaded["curves"]["default_step_key"], "_step")
        self.assertEqual(loaded["curves"]["metric_step_keys"]["val/*"], "train/global_step")
        self.assertIn("Smoke locally on 1 GPU", loaded["_notes"])

    def test_update_preflight_config_merges_metadata_patch_and_preserves_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wandb-driven-dev.local.md"
            cfg = {
                **DRIVAERML_CONFIG,
                "curves": {
                    "default_step_key": "_step",
                    "metric_step_keys": {"train/*": "train/global_step"},
                    "candidate_step_keys": ["_step"],
                },
            }
            wdd_helpers.write_config(cfg.copy(), path=path)

            loaded = wdd_helpers.update_preflight_config(
                {
                    "curves": {
                        "default_step_key": "_step",
                        "metric_step_keys": {
                            "val/loss": "train/global_step",
                            "val/surface_rel_l2": "train/global_step",
                        },
                        "candidate_step_keys": ["_step", "train/global_step"],
                    },
                    "wandb_metadata": {
                        "preflight": {
                            "project": "milieu/drivaerml",
                            "sampled_run_count": 1,
                            "observed_summary_keys": ["train/global_step", "val/loss"],
                            "observed_step_keys": ["train/global_step"],
                            "observed_metric_keys": ["train/global_step", "val/loss"],
                        }
                    },
                },
                project="milieu/drivaerml",
                path=path,
            )

        self.assertEqual(loaded["wandb_project"], "milieu/drivaerml")
        self.assertEqual(loaded["curves"]["metric_step_keys"]["train/*"], "train/global_step")
        self.assertEqual(loaded["curves"]["metric_step_keys"]["val/loss"], "train/global_step")
        self.assertEqual(loaded["curves"]["candidate_step_keys"], ["_step", "train/global_step"])
        self.assertEqual(
            loaded["wandb_metadata"]["preflight"]["observed_step_keys"],
            ["train/global_step"],
        )
        self.assertNotIn("wandb_cache", loaded)
        self.assertIn("Smoke locally on 1 GPU", loaded["_notes"])

    def test_update_curve_config_refuses_project_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wandb-driven-dev.local.md"
            wdd_helpers.write_config(DRIVAERML_CONFIG.copy(), path=path)

            with self.assertRaises(ValueError):
                wdd_helpers.update_curve_config(
                    {"metric_step_keys": {"val/loss": "global_step"}},
                    project="milieu/asr",
                    path=path,
                )

    def test_update_preflight_config_requires_existing_config_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.local.md"
            with self.assertRaises(FileNotFoundError):
                wdd_helpers.update_preflight_config(
                    {"curves": {"metric_step_keys": {"val/loss": "global_step"}}},
                    project="milieu/drivaerml",
                    path=path,
                )

    def test_curve_step_key_uses_exact_wildcard_and_default_config(self):
        cfg = {
            "curves": {
                "default_step_key": "_step",
                "metric_step_keys": {
                    "train/loss": "train/global_step",
                    "val/*": "eval/step",
                },
            }
        }

        self.assertEqual(wdd_helpers.curve_step_key(cfg, "train/loss"), "train/global_step")
        self.assertEqual(wdd_helpers.curve_step_key(cfg, "val/loss"), "eval/step")
        self.assertEqual(wdd_helpers.curve_step_key(cfg, "custom_metric"), "_step")
        self.assertEqual(wdd_helpers.curve_step_key({}, "train/loss"), "_step")
        self.assertEqual(
            wdd_helpers.curve_step_keys(cfg, ["train/loss", "val/loss"]),
            {"train/loss": "train/global_step", "val/loss": "eval/step"},
        )

    def test_run_id_accepts_urls_paths_and_raw_ids(self):
        self.assertEqual(
            wdd_helpers._run_id("https://wandb.ai/milieu/drivaerml/runs/abc123"),
            "abc123",
        )
        self.assertEqual(wdd_helpers._run_id("milieu/drivaerml/def456"), "def456")
        self.assertEqual(wdd_helpers._run_id("ghi789"), "ghi789")

    def test_create_experiment_report_builds_focused_table_and_delegates(self):
        captured: list[dict] = []

        def fake_create_comparison_report(**kwargs):
            captured.append(kwargs)
            return "https://wandb.ai/milieu/drivaerml/reports/experiment"

        runs = {
            "baseline": "https://wandb.ai/milieu/drivaerml/runs/base123",
            "variant": "https://wandb.ai/milieu/drivaerml/runs/var456",
        }

        with patch.object(
            wdd_helpers,
            "create_comparison_report",
            side_effect=fake_create_comparison_report,
        ):
            url = wdd_helpers.create_experiment_report(
                project=DRIVAERML_CONFIG["wandb_project"],
                slug="drivaerml-batch-size",
                decision_metrics=DRIVAERML_CONFIG["metrics"]["decision"],
                health_metrics=DRIVAERML_CONFIG["metrics"]["health"],
                runs=runs,
                question="Does larger batch improve DrivAerML validation error?",
                falsifier="Variant does not improve val/volume_rel_l2.",
                report_columns=["config.batch_size", "val/volume_rel_l2"],
                report_column_values={
                    "baseline": {"config.batch_size": 8, "val/volume_rel_l2": 0.22},
                    "variant": {"config.batch_size": 16, "val/volume_rel_l2": 0.18},
                },
                date="2026-05-08",
            )

        self.assertEqual(url, "https://wandb.ai/milieu/drivaerml/reports/experiment")
        call = captured[0]
        self.assertEqual(call["project"], "milieu/drivaerml")
        self.assertEqual(call["title"], "Experiment drivaerml-batch-size")
        self.assertEqual(call["decision_metrics"], DRIVAERML_CONFIG["metrics"]["decision"])
        self.assertEqual(call["health_metrics"], DRIVAERML_CONFIG["metrics"]["health"])
        self.assertIn("**Slug:** `drivaerml-batch-size`", call["header_md"])
        self.assertIn("**Falsifier:** Variant does not improve", call["header_md"])
        self.assertIn("| Run | config.batch_size | val/volume_rel_l2 |", call["focused_table_md"])
        self.assertIn("| baseline | 8 | 0.22 |", call["focused_table_md"])
        self.assertIn("| variant | 16 | 0.18 |", call["focused_table_md"])

    def test_md_table_cell_escapes_pipe_and_newline(self):
        self.assertEqual(wdd_helpers._md_table_cell("a|b\nc"), "a\\|b c")


class OnlineWddHelpersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_online_wandb()
        scripts_dir = ROOT / "skills" / "wbagent" / "scripts"
        import sys

        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from wandb_helpers import get_api

        cls.api = get_api()
        cls.project = DRIVAERML_CONFIG["wandb_project"]
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

    def test_find_run_by_name_finds_live_run_without_sweeps(self):
        found = wdd_helpers.find_run_by_name(
            self.api,
            self.project,
            self.latest_run.display_name,
            timeout_s=5,
            poll_interval_s=0,
        )

        self.assertIsNotNone(found)
        self.assertEqual(found.id, self.latest_run.id)
        self.assertEqual(found.display_name, self.latest_run.display_name)

    def test_verify_required_metrics_accepts_live_drivaerml_decision_and_health_metrics(self):
        run_path = f"{self.project}/{self.latest_run.id}"

        result = wdd_helpers.verify_required_metrics(
            self.api,
            [run_path],
            ALL_METRICS,
        )

        self.assertEqual(result, {run_path: []})

    def test_runtime_estimate_uses_live_finished_drivaerml_runs(self):
        estimate = wdd_helpers.runtime_estimate(
            self.api,
            self.project,
            name_pattern=".*",
            target_steps=4000,
            sample=5,
            min_steps=100,
        )

        self.assertIsNotNone(estimate)
        self.assertGreaterEqual(estimate["runs_used"], 1)
        self.assertEqual(estimate["min_steps"], 100)
        self.assertGreater(estimate["target_hours_low"], 0)
        self.assertGreaterEqual(estimate["target_hours_high"], estimate["target_hours_low"])

    def test_find_runs_by_config_filters_live_batch_size_and_backfills_nested_model_column(self):
        batch_size = self.latest_run.config.get("batch_size")
        model = self.latest_run.config.get("model") or {}
        hidden_dim = model.get("hidden_dim")
        if batch_size is None or hidden_dim is None:
            raise unittest.SkipTest("Latest run does not expose batch_size/model.hidden_dim config")

        rows = wdd_helpers.find_runs_by_config(
            self.api,
            self.project,
            config_filters={"batch_size": batch_size, "model.hidden_dim": hidden_dim},
            metric_keys=["val/volume_rel_l2"],
            extra_config_keys=["max_steps"],
            limit=5,
        )

        self.assertTrue(rows)
        self.assertTrue(any(row["name"] == self.latest_run.id for row in rows))
        for row in rows:
            self.assertEqual(row["config.batch_size"], batch_size)
            self.assertEqual(row["config.model.hidden_dim"], hidden_dim)
            self.assertIn("val/volume_rel_l2", row)


if __name__ == "__main__":
    unittest.main()
