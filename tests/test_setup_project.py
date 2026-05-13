from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SETUP_PROJECT_PATH = ROOT / "skills" / "wandb-driven-dev" / "scripts" / "setup_project.py"
WDD_HELPERS_PATH = ROOT / "skills" / "wandb-driven-dev" / "scripts" / "wdd_helpers.py"

PROJECT = "milieu/drivaerml"
TRAIN_LOSS = "train/loss"
VAL_LOSS = "val/loss"
GLOBAL_STEP = "train/global_step"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


setup_project = load_module("setup_project_under_test", SETUP_PROJECT_PATH)
wdd_helpers = load_module("wdd_helpers_for_setup_test", WDD_HELPERS_PATH)


BASE_CONFIG = {
    "wandb_project": PROJECT,
    "launcher": {"command": "uv run python k8s/launch.py", "reproduction": "clone"},
    "training": {"script": "scripts/train_drivaerml.py", "config_dir": "configs/drivaerml"},
    "gpus": {"smoke": 1, "full": 8},
    "metrics": {
        "decision": [VAL_LOSS],
        "health": [TRAIN_LOSS],
    },
    "curves": {
        "default_step_key": "_step",
        "metric_step_keys": {},
        "candidate_step_keys": ["_step", GLOBAL_STEP],
    },
    "_notes": "Smoke locally; full runs via k8s.",
}


class SetupProjectPureTests(unittest.TestCase):
    def test_recommend_metric_step_key_prefers_custom_key_with_coverage(self):
        frames = {
            "run": pd.DataFrame(
                [
                    {"_step": 0, GLOBAL_STEP: 0, TRAIN_LOSS: 1.0},
                    {"_step": 1, GLOBAL_STEP: 50, TRAIN_LOSS: 0.8},
                    {"_step": 2, GLOBAL_STEP: 100, TRAIN_LOSS: 0.7},
                ]
            )
        }

        recommendation = setup_project.recommend_metric_step_key(
            frames,
            TRAIN_LOSS,
            candidates=["_step", GLOBAL_STEP],
        )

        self.assertEqual(recommendation["recommended_step_key"], GLOBAL_STEP)
        self.assertEqual(recommendation["candidates"][GLOBAL_STEP]["points"], 3)

    def test_recommend_metric_step_key_falls_back_to_wandb_step(self):
        frames = {
            "run": pd.DataFrame(
                [
                    {"_step": 0, TRAIN_LOSS: 1.0},
                    {"_step": 1, TRAIN_LOSS: 0.8},
                    {"_step": 2, TRAIN_LOSS: 0.7},
                ]
            )
        }

        recommendation = setup_project.recommend_metric_step_key(
            frames,
            TRAIN_LOSS,
            candidates=["_step", GLOBAL_STEP],
        )

        self.assertEqual(recommendation["recommended_step_key"], "_step")
        self.assertEqual(recommendation["candidates"][GLOBAL_STEP]["points"], 0)

    def test_preflight_uses_selected_summary_graphql_by_default(self):
        calls = []

        def fake_fast_fetch_run_summaries(api, project, run_ids, summary_keys=None, order="-created_at", per_page=50):
            calls.append(
                {
                    "project": project,
                    "run_ids": run_ids,
                    "summary_keys": summary_keys,
                    "per_page": per_page,
                }
            )
            return [
                {
                    "name": "a",
                    "display_name": "run-a",
                    "state": "finished",
                    "created_at": "2026-01-01T00:00:00Z",
                    "summary": {TRAIN_LOSS: 0.7, VAL_LOSS: 0.8, GLOBAL_STEP: 100},
                },
                {
                    "name": "b",
                    "display_name": "run-b",
                    "state": "finished",
                    "created_at": "2026-01-02T00:00:00Z",
                    "summary": {TRAIN_LOSS: 0.6, VAL_LOSS: 0.9, GLOBAL_STEP: 200},
                },
            ]

        with (
            patch.object(setup_project, "fast_fetch_run_summaries", side_effect=fake_fast_fetch_run_summaries),
            patch.object(setup_project, "scan_history", side_effect=AssertionError("history scan should not run")),
        ):
            result = setup_project.preflight_wandb_step_keys(
                api=object(),
                project="entity/project",
                run_ids=["a", "b"],
                metrics=[TRAIN_LOSS, VAL_LOSS],
                candidate_step_keys=[GLOBAL_STEP],
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["run_ids"], ["a", "b"])
        self.assertIn(TRAIN_LOSS, calls[0]["summary_keys"])
        self.assertIn(GLOBAL_STEP, calls[0]["summary_keys"])
        self.assertEqual(result["mode"], "summary")
        self.assertEqual(result["metric_step_keys"][TRAIN_LOSS], GLOBAL_STEP)
        self.assertEqual(result["metric_step_keys"][VAL_LOSS], GLOBAL_STEP)
        self.assertEqual(result["candidate_step_keys"], ["_step", GLOBAL_STEP])

        metadata = result["config_patch"]["wandb_metadata"]["preflight"]
        self.assertEqual(metadata["sampled_run_count"], 2)
        self.assertEqual(metadata["observed_step_keys"], [GLOBAL_STEP])
        self.assertIn(TRAIN_LOSS, metadata["observed_summary_keys"])
        self.assertIn(VAL_LOSS, metadata["observed_metric_keys"])
        self.assertNotIn("run_summaries", metadata)
        self.assertNotIn("source_runs", metadata)
        self.assertNotIn(TRAIN_LOSS, metadata.get("metric_values", {}))

    def test_run_setup_preflight_reads_config_selects_runs_and_writes_metadata(self):
        def fake_fast_fetch_runs(api, project, metric_keys, filters, config_keys=None, order="-created_at", limit=50, per_page=50):
            self.assertEqual(metric_keys, [])
            self.assertEqual(filters, {"state": "finished"})
            self.assertEqual(limit, 3)
            return [{"name": "a"}, {"name": "b"}]

        def fake_fast_fetch_run_summaries(api, project, run_ids, summary_keys=None, order="-created_at", per_page=50):
            self.assertEqual(run_ids, ["a", "b"])
            return [
                {"name": "a", "summary": {TRAIN_LOSS: 0.7, VAL_LOSS: 0.8, GLOBAL_STEP: 100}},
                {"name": "b", "summary": {TRAIN_LOSS: 0.6, VAL_LOSS: 0.9, GLOBAL_STEP: 200}},
            ]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wandb-driven-dev.local.md"
            wdd_helpers.write_config(dict(BASE_CONFIG), path=path)
            with (
                patch.object(setup_project, "fast_fetch_runs", side_effect=fake_fast_fetch_runs),
                patch.object(setup_project, "fast_fetch_run_summaries", side_effect=fake_fast_fetch_run_summaries),
            ):
                result = setup_project.run_setup_preflight(
                    api=object(),
                    config_path=path,
                    write_config=True,
                )
            loaded = wdd_helpers.read_config(path)

        self.assertEqual(result["runs"], ["a", "b"])
        self.assertEqual(result["selected_metrics"], [VAL_LOSS, TRAIN_LOSS])
        self.assertEqual(loaded["curves"]["metric_step_keys"][VAL_LOSS], GLOBAL_STEP)
        self.assertEqual(loaded["curves"]["metric_step_keys"][TRAIN_LOSS], GLOBAL_STEP)
        self.assertEqual(loaded["wandb_metadata"]["preflight"]["observed_step_keys"], [GLOBAL_STEP])
        self.assertIn("Smoke locally", loaded["_notes"])

    def test_run_setup_preflight_requires_existing_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing.local.md"
            with self.assertRaisesRegex(RuntimeError, "missing"):
                setup_project.run_setup_preflight(api=object(), config_path=path)

    def test_preflight_fetch_scans_metrics_and_present_step_keys_once(self):
        class FakeRun:
            summary_metrics = {GLOBAL_STEP: 100, TRAIN_LOSS: 0.7, VAL_LOSS: 0.8}

        class FakeApi:
            def run(self, path):
                self.path = path
                return FakeRun()

        calls = []

        def fake_scan_history(run, keys, max_rows=None):
            calls.append(tuple(keys))
            if keys == ["_step", TRAIN_LOSS]:
                return [
                    {"_step": 0, TRAIN_LOSS: 1.0},
                    {"_step": 1, TRAIN_LOSS: 0.7},
                ]
            if keys == ["_step", VAL_LOSS]:
                return [
                    {"_step": 0, VAL_LOSS: 1.2},
                    {"_step": 1, VAL_LOSS: 0.8},
                ]
            if keys == ["_step", GLOBAL_STEP]:
                return [
                    {"_step": 0, GLOBAL_STEP: 0},
                    {"_step": 1, GLOBAL_STEP: 100},
                ]
            raise AssertionError(f"unexpected scan keys: {keys}")

        with patch.object(setup_project, "scan_history", side_effect=fake_scan_history):
            frame = setup_project.fetch_step_key_preflight_frame(
                FakeApi(),
                "entity/project",
                "run123",
                metrics=[TRAIN_LOSS, VAL_LOSS],
                candidate_step_keys=["_step", GLOBAL_STEP, "missing_step"],
                max_rows=10,
            )

        self.assertEqual(
            calls,
            [
                ("_step", TRAIN_LOSS),
                ("_step", VAL_LOSS),
                ("_step", GLOBAL_STEP),
            ],
        )
        self.assertEqual(frame[TRAIN_LOSS].dropna().tolist(), [1.0, 0.7])
        self.assertEqual(frame[VAL_LOSS].dropna().tolist(), [1.2, 0.8])
        self.assertEqual(frame[GLOBAL_STEP].dropna().tolist(), [0, 100])

    def test_preflight_fetch_uses_pairwise_fallback_for_sparse_metrics(self):
        class FakeRun:
            summary_metrics = {GLOBAL_STEP: 1000, TRAIN_LOSS: 0.7, VAL_LOSS: 0.8}

        class FakeApi:
            def run(self, path):
                self.path = path
                return FakeRun()

        calls = []

        def fake_scan_history(run, keys, max_rows=None):
            calls.append(tuple(keys))
            if keys == ["_step", TRAIN_LOSS]:
                return [
                    {"_step": 10, TRAIN_LOSS: 1.0},
                    {"_step": 20, TRAIN_LOSS: 0.7},
                ]
            if keys == ["_step", VAL_LOSS]:
                return [
                    {"_step": 100, VAL_LOSS: 1.2},
                    {"_step": 200, VAL_LOSS: 0.8},
                ]
            if keys == ["_step", GLOBAL_STEP]:
                return [
                    {"_step": 10, GLOBAL_STEP: 100},
                    {"_step": 20, GLOBAL_STEP: 200},
                ]
            if keys == ["_step", GLOBAL_STEP, VAL_LOSS]:
                return [
                    {"_step": 100, GLOBAL_STEP: 1000, VAL_LOSS: 1.2},
                    {"_step": 200, GLOBAL_STEP: 2000, VAL_LOSS: 0.8},
                ]
            raise AssertionError(f"unexpected scan keys: {keys}")

        with patch.object(setup_project, "scan_history", side_effect=fake_scan_history):
            frame = setup_project.fetch_step_key_preflight_frame(
                FakeApi(),
                "entity/project",
                "run123",
                metrics=[TRAIN_LOSS, VAL_LOSS],
                candidate_step_keys=["_step", GLOBAL_STEP],
                max_rows=10,
            )

        self.assertEqual(
            calls,
            [
                ("_step", TRAIN_LOSS),
                ("_step", VAL_LOSS),
                ("_step", GLOBAL_STEP),
                ("_step", GLOBAL_STEP, VAL_LOSS),
            ],
        )
        val_rows = frame.dropna(subset=[VAL_LOSS, GLOBAL_STEP])
        self.assertEqual(val_rows[GLOBAL_STEP].tolist(), [1000, 2000])
        recommendation = setup_project.recommend_metric_step_key(
            {"run": frame},
            VAL_LOSS,
            candidates=["_step", GLOBAL_STEP],
        )
        self.assertEqual(recommendation["recommended_step_key"], GLOBAL_STEP)


if __name__ == "__main__":
    unittest.main()
