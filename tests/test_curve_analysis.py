from __future__ import annotations

import importlib.util
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CURVE_ANALYSIS_PATH = ROOT / "skills" / "wandb-driven-dev" / "scripts" / "curve_analysis.py"

PROJECT = "milieu/drivaerml"
TRAIN_LOSS = "train/loss"
SURFACE_METRIC = "val/surface_rel_l2"
VOLUME_METRIC = "val/volume_rel_l2"
U_METRIC = "val/u_rel_l2"
VAL_LOSS = "val/loss"
GLOBAL_STEP = "train/global_step"
LONG_200K_RUN_IDS = [
    "llr9m0se",
    "8nugtq3j",
    "3bdnsf6w",
    "qp3abyry",
    "9bwk26k5",
    "iz9hzsob",
    "4b4r1y38",
    "bf00cxqu",
    "whqtivq4",
    "g4ydd899",
    "9em0th6r",
    "1c39t8kk",
    "1zs8rfih",
    "copxn2r0",
    "hsegz7mh",
    "ty2ezne0",
    "187yik7k",
    "8wdbu1su",
    "ronmx306",
    "ctwltfic",
]


def load_curve_analysis():
    spec = importlib.util.spec_from_file_location("curve_analysis_under_test", CURVE_ANALYSIS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


curve_analysis = load_curve_analysis()


def require_online_wandb():
    if not os.environ.get("WANDB_API_KEY"):
        raise unittest.SkipTest("WANDB_API_KEY is required for online W&B tests")
    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("Install wandb to run online W&B tests") from exc


class CurveAnalysisPureTests(unittest.TestCase):
    def test_rows_to_curve_frame_sorts_numeric_rows(self):
        rows = [
            {"_step": 2, GLOBAL_STEP: "100", TRAIN_LOSS: "0.7"},
            {"_step": 0, GLOBAL_STEP: "0", TRAIN_LOSS: "1.0"},
            {"_step": 1, GLOBAL_STEP: "50", TRAIN_LOSS: "0.8"},
            {"_step": 3, GLOBAL_STEP: None, TRAIN_LOSS: "0.1"},
        ]

        frame = curve_analysis.rows_to_curve_frame(rows, [TRAIN_LOSS], step_key=GLOBAL_STEP)

        self.assertEqual(frame[GLOBAL_STEP].tolist(), [0.0, 50.0, 100.0])
        self.assertEqual(frame[TRAIN_LOSS].tolist(), [1.0, 0.8, 0.7])

    def test_rows_to_curve_frame_supports_wandb_step_as_step_key(self):
        rows = [
            {"_step": 2, TRAIN_LOSS: "0.7"},
            {"_step": 0, TRAIN_LOSS: "1.0"},
            {"_step": 1, TRAIN_LOSS: "0.8"},
        ]

        frame = curve_analysis.rows_to_curve_frame(rows, [TRAIN_LOSS], step_key="_step")
        metric = curve_analysis.metric_frame(frame, TRAIN_LOSS, step_key="_step")
        snapshot = curve_analysis.curve_snapshot(
            frame,
            TRAIN_LOSS,
            target_step=2,
            window_steps=2,
            step_key="_step",
            smoothing_points=1,
        )

        self.assertEqual(frame.columns.tolist(), ["_step", TRAIN_LOSS])
        self.assertEqual(metric.columns.tolist(), ["_step", TRAIN_LOSS])
        self.assertEqual(snapshot["step"], 2.0)
        self.assertEqual(snapshot["value"], 0.7)

    def test_curve_snapshot_computes_latest_point_slope_and_trend(self):
        frame = pd.DataFrame(
            [
                {"_step": 0, GLOBAL_STEP: 0, TRAIN_LOSS: 1.0},
                {"_step": 1, GLOBAL_STEP: 50, TRAIN_LOSS: 0.8},
                {"_step": 2, GLOBAL_STEP: 100, TRAIN_LOSS: 0.7},
                {"_step": 3, GLOBAL_STEP: 150, TRAIN_LOSS: 0.65},
            ]
        )

        snapshot = curve_analysis.curve_snapshot(
            frame,
            TRAIN_LOSS,
            target_step=125,
            window_steps=100,
            step_key=GLOBAL_STEP,
            lower_is_better=True,
            smoothing_points=1,
        )

        self.assertEqual(snapshot["step"], 100)
        self.assertEqual(snapshot["value"], 0.7)
        self.assertEqual(snapshot["window_start_step"], 0)
        self.assertAlmostEqual(snapshot["window_delta"], -0.3)
        self.assertAlmostEqual(snapshot["slope_per_step"], -0.003)
        self.assertAlmostEqual(snapshot["slope_per_1k"], -3.0)
        self.assertEqual(snapshot["trend"], "improving")
        self.assertIn(snapshot["trend_confidence"], {"high", "medium"})

    def test_curve_snapshot_smoothing_handles_noisy_endpoint(self):
        frame = pd.DataFrame(
            [
                {"_step": 0, GLOBAL_STEP: 0, TRAIN_LOSS: 1.0},
                {"_step": 1, GLOBAL_STEP: 25, TRAIN_LOSS: 0.8},
                {"_step": 2, GLOBAL_STEP: 50, TRAIN_LOSS: 0.6},
                {"_step": 3, GLOBAL_STEP: 75, TRAIN_LOSS: 0.4},
                {"_step": 4, GLOBAL_STEP: 100, TRAIN_LOSS: 2.0},
            ]
        )

        snapshot = curve_analysis.curve_snapshot(
            frame,
            TRAIN_LOSS,
            target_step=100,
            window_steps=100,
            step_key=GLOBAL_STEP,
            lower_is_better=True,
            smoothing_points=3,
        )

        self.assertEqual(snapshot["value"], 2.0)
        self.assertEqual(snapshot["smoothed_value"], 0.6)
        self.assertLess(snapshot["slope_per_1k"], 0)
        self.assertEqual(snapshot["trend"], "improving")
        self.assertGreater(snapshot["noise_per_1k"], 0)

    def test_early_training_health_accepts_stable_descent(self):
        frame = pd.DataFrame(
            [
                {"_step": 0, GLOBAL_STEP: 0, TRAIN_LOSS: 1.0},
                {"_step": 1, GLOBAL_STEP: 25, TRAIN_LOSS: 0.9},
                {"_step": 2, GLOBAL_STEP: 50, TRAIN_LOSS: 0.8},
                {"_step": 3, GLOBAL_STEP: 75, TRAIN_LOSS: 0.72},
            ]
        )

        health = curve_analysis.early_training_health(
            frame,
            TRAIN_LOSS,
            target_step=75,
            step_key=GLOBAL_STEP,
            lower_is_better=True,
            smoothing_points=1,
        )

        self.assertEqual(health["status"], "healthy")
        self.assertEqual(health["directional_fraction"], 1.0)
        self.assertGreater(health["relative_improvement"], 0)
        self.assertEqual(health["snapshot"]["trend"], "improving")

    def test_early_training_health_flags_noisy_launch(self):
        frame = pd.DataFrame(
            [
                {"_step": 0, GLOBAL_STEP: 0, TRAIN_LOSS: 1.0},
                {"_step": 1, GLOBAL_STEP: 25, TRAIN_LOSS: 0.9},
                {"_step": 2, GLOBAL_STEP: 50, TRAIN_LOSS: 0.85},
                {"_step": 3, GLOBAL_STEP: 75, TRAIN_LOSS: 1.8},
            ]
        )

        health = curve_analysis.early_training_health(
            frame,
            TRAIN_LOSS,
            target_step=75,
            step_key=GLOBAL_STEP,
            lower_is_better=True,
            max_bad_spikes=0,
        )

        self.assertIn(health["status"], {"watch", "fail"})
        self.assertGreaterEqual(health["spikes"]["bad_spike_count"], 1)
        self.assertIn("bad-direction spikes exceed early tolerance", health["reasons"])

    def test_progress_curve_insight_reports_recent_spike(self):
        frame = pd.DataFrame(
            [
                {"_step": 0, GLOBAL_STEP: 0, TRAIN_LOSS: 1.0},
                {"_step": 1, GLOBAL_STEP: 50, TRAIN_LOSS: 0.9},
                {"_step": 2, GLOBAL_STEP: 100, TRAIN_LOSS: 0.8},
                {"_step": 3, GLOBAL_STEP: 150, TRAIN_LOSS: 0.7},
                {"_step": 4, GLOBAL_STEP: 200, TRAIN_LOSS: 0.6},
                {"_step": 5, GLOBAL_STEP: 250, TRAIN_LOSS: 0.55},
                {"_step": 6, GLOBAL_STEP: 300, TRAIN_LOSS: 1.3},
            ]
        )

        insight = curve_analysis.progress_curve_insight(
            frame,
            TRAIN_LOSS,
            target_step=300,
            window_steps=150,
            step_key=GLOBAL_STEP,
            lower_is_better=True,
            smoothing_points=3,
        )

        self.assertEqual(insight["stage"], "progress")
        self.assertGreaterEqual(insight["spikes"]["bad_spike_count"], 1)
        self.assertIn("recent bad-direction spikes detected", insight["reasons"])
        self.assertEqual(insight["current"]["value"], 1.3)

    def test_analyze_curve_frames_ranks_lower_and_higher_metrics(self):
        frames = {
            "baseline": pd.DataFrame(
                [
                    {"_step": 0, GLOBAL_STEP: 0, TRAIN_LOSS: 1.0, "val/accuracy": 0.4},
                    {"_step": 1, GLOBAL_STEP: 100, TRAIN_LOSS: 0.8, "val/accuracy": 0.5},
                ]
            ),
            "variant": pd.DataFrame(
                [
                    {"_step": 0, GLOBAL_STEP: 0, TRAIN_LOSS: 1.0, "val/accuracy": 0.4},
                    {"_step": 1, GLOBAL_STEP: 100, TRAIN_LOSS: 0.7, "val/accuracy": 0.6},
                ]
            ),
        }

        result = curve_analysis.analyze_curve_frames(
            frames,
            metrics=[TRAIN_LOSS, "val/accuracy"],
            steps=[100],
            window_steps=100,
            step_key=GLOBAL_STEP,
        )

        self.assertTrue(result["metrics"][TRAIN_LOSS]["lower_is_better"])
        self.assertEqual(result["metrics"][TRAIN_LOSS]["points"][0]["best_run"], "variant")
        self.assertFalse(result["metrics"]["val/accuracy"]["lower_is_better"])
        self.assertEqual(result["metrics"]["val/accuracy"]["points"][0]["best_run"], "variant")

    def test_analyze_curve_frames_auto_selects_stage_by_step(self):
        frames = {
            "run": pd.DataFrame(
                [
                    {"_step": 0, GLOBAL_STEP: 0, TRAIN_LOSS: 1.0},
                    {"_step": 1, GLOBAL_STEP: 100, TRAIN_LOSS: 0.8},
                    {"_step": 2, GLOBAL_STEP: 200, TRAIN_LOSS: 0.7},
                    {"_step": 3, GLOBAL_STEP: 2000, TRAIN_LOSS: 0.5},
                ]
            )
        }

        result = curve_analysis.analyze_curve_frames(
            frames,
            metrics=[TRAIN_LOSS],
            steps=[200, 2000],
            window_steps=100,
            step_key=GLOBAL_STEP,
            stage="auto",
            early_step_threshold=500,
        )

        points = result["metrics"][TRAIN_LOSS]["points"]
        self.assertEqual(points[0]["resolved_stage"], "early")
        self.assertEqual(points[0]["stage_analysis"]["run"]["stage"], "early")
        self.assertEqual(points[1]["resolved_stage"], "progress")
        self.assertEqual(points[1]["stage_analysis"]["run"]["stage"], "progress")

    def test_compare_curve_frames_uses_opinionated_defaults(self):
        frames = {
            "baseline": pd.DataFrame(
                [
                    {"_step": 0, GLOBAL_STEP: 0, TRAIN_LOSS: 1.0},
                    {"_step": 1, GLOBAL_STEP: 100, TRAIN_LOSS: 0.8},
                ]
            ),
            "variant": pd.DataFrame(
                [
                    {"_step": 0, GLOBAL_STEP: 0, TRAIN_LOSS: 1.0},
                    {"_step": 1, GLOBAL_STEP: 100, TRAIN_LOSS: 0.7},
                ]
            ),
        }

        result = curve_analysis.compare_curve_frames(
            frames,
            metrics=[TRAIN_LOSS],
            steps=[100],
            step_key=GLOBAL_STEP,
        )

        self.assertEqual(result["window_steps"], 1000)
        self.assertEqual(result["smoothing_method"], "median")
        self.assertEqual(result["stage"], "auto")
        self.assertEqual(result["metrics"][TRAIN_LOSS]["points"][0]["best_run"], "variant")

    def test_compare_wandb_curves_from_config_groups_metrics_by_step_key(self):
        calls = []

        def fake_compare(api, project, run_ids, metrics, steps, step_key):
            calls.append(
                {
                    "project": project,
                    "run_ids": run_ids,
                    "metrics": metrics,
                    "steps": steps,
                    "step_key": step_key,
                }
            )
            return {
                "metrics": {
                    metric: {"points": [{"best_run": "run-a"}]}
                    for metric in metrics
                },
                "fetched_through_step": max(steps),
                "workers": 2,
            }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "wandb-driven-dev.local.md"
            path.write_text(
                """---
wandb_project: entity/project
curves:
  default_step_key: _step
  metric_step_keys:
    train/*: train/global_step
    val/*: eval/global_step
  candidate_step_keys: [_step, train/global_step, eval/global_step]
---
"""
            )
            with patch.object(curve_analysis, "compare_wandb_curves", side_effect=fake_compare):
                result = curve_analysis.compare_wandb_curves_from_config(
                    api=object(),
                    run_ids=["run-a", "run-b"],
                    metrics=[TRAIN_LOSS, VAL_LOSS],
                    steps=[100],
                    config_path=path,
                )

        self.assertEqual(
            calls,
            [
                {
                    "project": "entity/project",
                    "run_ids": ["run-a", "run-b"],
                    "metrics": [TRAIN_LOSS],
                    "steps": [100],
                    "step_key": "train/global_step",
                },
                {
                    "project": "entity/project",
                    "run_ids": ["run-a", "run-b"],
                    "metrics": [VAL_LOSS],
                    "steps": [100],
                    "step_key": "eval/global_step",
                },
            ],
        )
        self.assertEqual(
            result["metric_step_keys"],
            {TRAIN_LOSS: "train/global_step", VAL_LOSS: "eval/global_step"},
        )
        self.assertEqual(set(result["metrics"]), {TRAIN_LOSS, VAL_LOSS})

    def test_analyze_wandb_runs_fetches_runs_in_parallel(self):
        def fake_fetch(api, project, run_id, metrics, step_key, max_step, max_rows):
            time.sleep(0.1)
            return pd.DataFrame(
                [
                    {"_step": 0, GLOBAL_STEP: 0, TRAIN_LOSS: 1.0},
                    {"_step": 1, GLOBAL_STEP: 100, TRAIN_LOSS: 0.5 if run_id == "b" else 0.8},
                ]
            )

        start = time.perf_counter()
        with patch.object(curve_analysis, "fetch_run_curve_frame", side_effect=fake_fetch):
            result = curve_analysis.analyze_wandb_runs(
                api=object(),
                project="entity/project",
                run_ids=["a", "b", "c"],
                metrics=[TRAIN_LOSS],
                steps=[100],
                window_steps=100,
                step_key=GLOBAL_STEP,
                workers=3,
                api_factory=lambda: object(),
            )
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 0.25)
        self.assertEqual(result["workers"], 3)
        self.assertEqual(result["metrics"][TRAIN_LOSS]["points"][0]["best_run"], "b")


class OnlineCurveAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_online_wandb()
        cls.api = curve_analysis.get_api()

    def test_analyze_wandb_runs_compares_drivaerml_curves_at_step_15305(self):
        result = curve_analysis.analyze_wandb_runs(
            self.api,
            PROJECT,
            run_ids=["srehoxzc", "bo2blqjb"],
            metrics=[TRAIN_LOSS, SURFACE_METRIC, VOLUME_METRIC, U_METRIC, VAL_LOSS],
            steps=[15305],
            window_steps=1000,
            step_key=GLOBAL_STEP,
        )

        train_point = result["metrics"][TRAIN_LOSS]["points"][0]
        surface_point = result["metrics"][SURFACE_METRIC]["points"][0]
        self.assertEqual(train_point["best_run"], "srehoxzc")
        self.assertEqual(surface_point["best_run"], "srehoxzc")

        train_srehoxzc = train_point["runs"]["srehoxzc"]
        train_bo2blqjb = train_point["runs"]["bo2blqjb"]
        self.assertEqual(train_point["resolved_stage"], "progress")
        self.assertEqual(train_point["stage_analysis"]["srehoxzc"]["stage"], "progress")
        self.assertEqual(train_srehoxzc["step"], 15300)
        self.assertEqual(train_bo2blqjb["step"], 15300)
        self.assertAlmostEqual(train_srehoxzc["value"], 0.04521029070019722)
        self.assertAlmostEqual(train_bo2blqjb["value"], 0.9314912557601929)
        self.assertLess(train_srehoxzc["slope_per_1k"], 0)

        surface_srehoxzc = surface_point["runs"]["srehoxzc"]
        surface_bo2blqjb = surface_point["runs"]["bo2blqjb"]
        self.assertEqual(surface_srehoxzc["step"], 15000)
        self.assertEqual(surface_bo2blqjb["step"], 15000)
        self.assertAlmostEqual(surface_srehoxzc["value"], 0.07945780126933602)
        self.assertAlmostEqual(surface_bo2blqjb["value"], 0.3133448169707625)

    def test_analyze_20_long_drivaerml_curves_stays_under_budget(self):
        start = time.perf_counter()
        result = curve_analysis.analyze_wandb_runs(
            self.api,
            PROJECT,
            run_ids=LONG_200K_RUN_IDS,
            metrics=[TRAIN_LOSS, SURFACE_METRIC],
            steps=[150000],
            window_steps=10000,
            step_key=GLOBAL_STEP,
            workers=12,
        )
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 30.0)
        self.assertEqual(result["workers"], 12)
        self.assertEqual(result["metrics"][TRAIN_LOSS]["points"][0]["best_run"], "bf00cxqu")
        self.assertEqual(result["metrics"][SURFACE_METRIC]["points"][0]["best_run"], "bf00cxqu")
        self.assertAlmostEqual(
            result["metrics"][TRAIN_LOSS]["points"][0]["runs"]["bf00cxqu"]["value"],
            0.014377381652593613,
        )
        self.assertAlmostEqual(
            result["metrics"][SURFACE_METRIC]["points"][0]["runs"]["bf00cxqu"]["value"],
            0.05642211868934791,
        )


if __name__ == "__main__":
    unittest.main()
