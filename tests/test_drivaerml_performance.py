from __future__ import annotations

import importlib.util
import os
import time
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
WANDB_HELPERS_PATH = ROOT / "skills" / "wbagent" / "scripts" / "wandb_helpers.py"

PROJECT = "milieu/drivaerml"
AS_OF_FILTER = {"created_at": {"$lt": "2026-05-08"}}

SURFACE_METRIC = "val/surface_rel_l2"
VOLUME_METRIC = "val/volume_rel_l2"
U_METRIC = "val/u_rel_l2"
VAL_LOSS = "val/loss"
TRAIN_LOSS = "train/loss"
GLOBAL_STEP = "train/global_step"

GOLDEN_BEST_ABUPT_20K = {
    "id": "srehoxzc",
    "display_name": "exp-20260429-3model-bench-variant-abupt",
    "max_steps": 20_000,
    "train/global_step": 20_000,
    "model_class": "abupt",
    "surface_anchor_points": 8_000,
    SURFACE_METRIC: 0.07429194545928113,
}
GOLDEN_SECOND_ABUPT_20K_ID = "bo2blqjb"
GOLDEN_100K_GLOBAL_STEP_COUNT = 52
GOLDEN_COMPARE_AT_STEP = {
    "target_step": 15_305,
    "runs": {
        "srehoxzc": {
            "train_floor_step": 15_300,
            TRAIN_LOSS: 0.04521029070019722,
            "val_floor_step": 15_000,
            SURFACE_METRIC: 0.07945780126933602,
            VOLUME_METRIC: 0.15117401261296323,
            U_METRIC: 0.13385340545687657,
            VAL_LOSS: 0.04099014331586659,
        },
        "bo2blqjb": {
            "train_floor_step": 15_300,
            TRAIN_LOSS: 0.9314912557601929,
            "val_floor_step": 15_000,
            SURFACE_METRIC: 0.3133448169707625,
            VOLUME_METRIC: 0.5791143328250141,
            U_METRIC: 0.21636501578114703,
            VAL_LOSS: 0.9659046530723572,
        },
    },
}

LATENCY_BUDGETS_S = {
    "best_abupt_20k": 5.0,
    "count_100k": 2.0,
    "find_two_and_compare_step": 30.0,
}


def load_wandb_helpers():
    spec = importlib.util.spec_from_file_location("wandb_helpers_perf_under_test", WANDB_HELPERS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wandb_helpers = load_wandb_helpers()


def require_online_wandb():
    if not os.environ.get("WANDB_API_KEY"):
        raise unittest.SkipTest("WANDB_API_KEY is required for online W&B performance tests")
    try:
        import wandb  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("Install wandb to run online W&B performance tests") from exc


def timed(callable_):
    start = time.perf_counter()
    result = callable_()
    return result, time.perf_counter() - start


def assert_close(testcase: unittest.TestCase, actual: float, expected: float) -> None:
    testcase.assertAlmostEqual(actual, expected, places=12)


class FakeHistoryRun:
    lastHistoryStep = 9

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def scan_history(self, *, keys: list[str]):
        self.calls.append({"keys": keys})
        return iter(
            [
                {"_step": 0, GLOBAL_STEP: 0, TRAIN_LOSS: 1.0},
                {"_step": 1, GLOBAL_STEP: 50, TRAIN_LOSS: 0.9},
                {"_step": 2, GLOBAL_STEP: 100, TRAIN_LOSS: 0.8},
                {"_step": 3, GLOBAL_STEP: 150, TRAIN_LOSS: 0.7},
            ]
        )


class WandbHelpersPerformancePureTests(unittest.TestCase):
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

    def test_scan_until_step_stops_after_target_is_passed(self):
        run = FakeHistoryRun()

        rows = wandb_helpers.scan_history_until_step(
            run,
            keys=["_step", GLOBAL_STEP, TRAIN_LOSS],
            step_key=GLOBAL_STEP,
            target_step=75,
        )

        self.assertEqual(run.calls, [{"keys": ["_step", GLOBAL_STEP, TRAIN_LOSS]}])
        self.assertEqual([row[GLOBAL_STEP] for row in rows], [0, 50])

    def test_fetch_runs_omits_summary_metrics_when_no_metric_keys_requested(self):
        captured = {}

        def fake_post_graphql(api, query, variables):
            captured["query"] = query
            captured["variables"] = variables
            return {
                "data": {
                    "project": {
                        "runs": {
                            "edges": [
                                {
                                    "node": {
                                        "id": "Run:abc",
                                        "name": "abc",
                                        "state": "finished",
                                        "createdAt": "2026-01-01T00:00:00Z",
                                        "displayName": "abc",
                                    }
                                }
                            ],
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                        }
                    }
                }
            }

        with patch.object(wandb_helpers, "_post_graphql", side_effect=fake_post_graphql):
            rows = wandb_helpers.fetch_runs(
                api=object(),
                path="entity/project",
                metric_keys=[],
                filters={"state": "finished"},
                limit=1,
                per_page=1,
            )

        self.assertNotIn("summaryMetrics", captured["query"])
        self.assertEqual(rows[0]["name"], "abc")


class DrivaerMlPerformanceReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_online_wandb()
        cls.api = wandb_helpers.get_api()

    def fetch_abupt_20k(self, limit: int) -> list[dict]:
        filters = {
            "state": "finished",
            **AS_OF_FILTER,
            "config.model.model_class": "abupt",
            "config.max_steps": 20_000,
        }
        return wandb_helpers.fetch_runs(
            self.api,
            PROJECT,
            metric_keys=[SURFACE_METRIC, VOLUME_METRIC, GLOBAL_STEP],
            filters=filters,
            config_keys=["max_steps", "model"],
            order=f"+summary_metrics.{SURFACE_METRIC}",
            limit=limit,
            per_page=limit,
        )

    def test_best_abupt_20k_by_surface_metric_is_fast_and_matches_golden(self):
        rows, latency_s = timed(lambda: self.fetch_abupt_20k(limit=1))

        self.assertLess(latency_s, LATENCY_BUDGETS_S["best_abupt_20k"])
        self.assertEqual(len(rows), 1)
        best = rows[0]
        model = best["config.model"]
        self.assertEqual(best["name"], GOLDEN_BEST_ABUPT_20K["id"])
        self.assertEqual(best["config.max_steps"], GOLDEN_BEST_ABUPT_20K["max_steps"])
        self.assertEqual(best[GLOBAL_STEP], GOLDEN_BEST_ABUPT_20K[GLOBAL_STEP])
        self.assertEqual(model["model_class"], GOLDEN_BEST_ABUPT_20K["model_class"])
        self.assertEqual(
            model["surface_anchor_points"],
            GOLDEN_BEST_ABUPT_20K["surface_anchor_points"],
        )
        assert_close(self, best[SURFACE_METRIC], GOLDEN_BEST_ABUPT_20K[SURFACE_METRIC])

    def test_count_100k_step_runs_is_exact_and_fast(self):
        def count_runs():
            return wandb_helpers.count_runs(
                self.api,
                PROJECT,
                filters={
                    "state": "finished",
                    **AS_OF_FILTER,
                    f"summary_metrics.{GLOBAL_STEP}": 100_000,
                },
            )

        count, latency_s = timed(count_runs)

        self.assertLess(latency_s, LATENCY_BUDGETS_S["count_100k"])
        self.assertEqual(count, GOLDEN_100K_GLOBAL_STEP_COUNT)

    def test_find_two_abupt_runs_and_compare_at_step_15305_is_fast_and_golden(self):
        target_step = GOLDEN_COMPARE_AT_STEP["target_step"]

        def find_and_compare():
            rows = self.fetch_abupt_20k(limit=2)
            raw = wandb_helpers.compare_runs_at_step(
                self.api,
                PROJECT,
                run_ids=[row["name"] for row in rows],
                step=target_step,
                metrics=[TRAIN_LOSS, SURFACE_METRIC, VOLUME_METRIC, U_METRIC, VAL_LOSS],
                step_key=GLOBAL_STEP,
            )
            comparison = {
                run_id: {
                    "train_floor_step": values["metrics"][TRAIN_LOSS][GLOBAL_STEP],
                    TRAIN_LOSS: values["metrics"][TRAIN_LOSS]["value"],
                    "val_floor_step": values["metrics"][SURFACE_METRIC][GLOBAL_STEP],
                    SURFACE_METRIC: values["metrics"][SURFACE_METRIC]["value"],
                    VOLUME_METRIC: values["metrics"][VOLUME_METRIC]["value"],
                    U_METRIC: values["metrics"][U_METRIC]["value"],
                    VAL_LOSS: values["metrics"][VAL_LOSS]["value"],
                }
                for run_id, values in raw.items()
            }
            return rows, comparison

        (rows, comparison), latency_s = timed(find_and_compare)

        self.assertLess(latency_s, LATENCY_BUDGETS_S["find_two_and_compare_step"])
        self.assertEqual([row["name"] for row in rows], [
            GOLDEN_BEST_ABUPT_20K["id"],
            GOLDEN_SECOND_ABUPT_20K_ID,
        ])
        for run_id, expected in GOLDEN_COMPARE_AT_STEP["runs"].items():
            actual = comparison[run_id]
            self.assertEqual(actual["train_floor_step"], expected["train_floor_step"])
            self.assertEqual(actual["val_floor_step"], expected["val_floor_step"])
            assert_close(self, actual[TRAIN_LOSS], expected[TRAIN_LOSS])
            assert_close(self, actual[SURFACE_METRIC], expected[SURFACE_METRIC])
            assert_close(self, actual[VOLUME_METRIC], expected[VOLUME_METRIC])
            assert_close(self, actual[U_METRIC], expected[U_METRIC])
            assert_close(self, actual[VAL_LOSS], expected[VAL_LOSS])
        self.assertLess(comparison["srehoxzc"][SURFACE_METRIC], comparison["bo2blqjb"][SURFACE_METRIC])


if __name__ == "__main__":
    unittest.main()
