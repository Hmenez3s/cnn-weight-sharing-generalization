"""Synthetic checks for the paired posthoc diagnostic calculations."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "width-depth-20261017"
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_endpoint_dynamics import _check_selected, _first_crossing, _paired  # noqa: E402


class EndpointDynamicsTests(unittest.TestCase):
    def test_paired_mean_interval_and_tie_counts(self) -> None:
        left = {11: 1.0, 23: 3.0, 37: 2.0, 53: 2.0, 71: 5.0}
        right = {11: 1.0, 23: 1.0, 37: 2.0, 53: 0.0, 71: 5.0}
        result = _paired(left, right)
        self.assertAlmostEqual(result["mean_difference"], 0.8)
        self.assertAlmostEqual(result["sd"], math.sqrt(1.2))
        margin = 2.7764451051977987 * math.sqrt(1.2) / math.sqrt(5)
        self.assertAlmostEqual(result["ci95"][0], 0.8 - margin)
        self.assertAlmostEqual(result["ci95"][1], 0.8 + margin)
        self.assertEqual((result["positive_seeds"], result["negative_seeds"], result["ties"]), (2, 0, 3))

    def test_never_attained_threshold_is_none_not_zero(self) -> None:
        rows = {200: {"validation_accuracy": 0.17}, 400: {"validation_accuracy": None},
                600: {"validation_accuracy": 0.179}}
        self.assertIsNone(_first_crossing(rows, 18.0))
        rows[800] = {"validation_accuracy": 0.18}
        self.assertEqual(_first_crossing(rows, 18.0), 800)

    def test_historical_ce_replay_tolerance_keeps_accuracy_guard_strict(self) -> None:
        logged = {"validation_accuracy_percent": 21.25, "validation_loss": 3.5,
                  "clean_train_accuracy_percent": 40.0, "gap_pp": 18.75}
        archived = {"validation_accuracy_percent": 21.25, "validation_loss": 3.5000001,
                    "clean_train_accuracy_percent": 40.0}
        self.assertAlmostEqual(_check_selected(archived, logged, "C8", 11, "selected"), 3.5)
        self.assertAlmostEqual(_check_selected(archived, logged, "H8", 11, "selected"), 3.5)
        with self.assertRaises(ValueError):
            _check_selected({**archived, "validation_loss": 3.501}, logged, "C8", 11, "selected")
        with self.assertRaises(ValueError):
            _check_selected(archived, {**logged, "validation_accuracy_percent": 21.2500001},
                            "C8", 11, "selected")
        with self.assertRaises(ValueError):
            _check_selected(archived, logged, "C7B1", 11, "selected")


if __name__ == "__main__":
    unittest.main()
