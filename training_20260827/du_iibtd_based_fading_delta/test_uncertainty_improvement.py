import unittest
from types import SimpleNamespace

import numpy as np

from du_iibtd_based_fading_delta.shared.noquant.config import Config as NoquantConfig
from du_iibtd_based_fading_delta.shared.noquant.environment import (
    UAVUGVEnvironment as NoquantEnvironment,
    compute_uncertainty_update_metrics as compute_noquant_metrics,
)
from du_iibtd_based_fading_delta.shared.noquant.runner import parse_args as parse_noquant_args
from du_iibtd_based_fading_delta.shared.quant.config import Config as QuantConfig
from du_iibtd_based_fading_delta.shared.quant.environment import (
    UAVUGVEnvironment as QuantEnvironment,
    compute_uncertainty_update_metrics as compute_quant_metrics,
)
from du_iibtd_based_fading_delta.shared.quant.runner import parse_args as parse_quant_args


class UncertaintyImprovementTests(unittest.TestCase):
    def test_improvement_is_signed_while_delta_is_unsigned(self):
        previous = np.ones((2, 3, 4), dtype=float)

        for compute_metrics in (compute_quant_metrics, compute_noquant_metrics):
            with self.subTest(compute_metrics=compute_metrics.__module__):
                down_delta, down_improvement = compute_metrics(previous, 0.99 * previous)
                up_delta, up_improvement = compute_metrics(previous, 1.01 * previous)

                self.assertAlmostEqual(down_delta, 0.01)
                self.assertAlmostEqual(up_delta, 0.01)
                self.assertAlmostEqual(down_improvement, 0.01)
                self.assertAlmostEqual(up_improvement, -0.01)

    def test_metrics_reject_shape_mismatch(self):
        for compute_metrics in (compute_quant_metrics, compute_noquant_metrics):
            with self.subTest(compute_metrics=compute_metrics.__module__):
                with self.assertRaisesRegex(ValueError, "matching shapes"):
                    compute_metrics(np.ones((2, 2, 2)), np.ones((2, 2, 3)))

    def test_default_threshold_is_002_for_both_variants(self):
        self.assertAlmostEqual(
            QuantConfig().planner.hybrid_uncertainty_improvement_threshold,
            0.02,
        )
        self.assertAlmostEqual(
            NoquantConfig().planner.hybrid_uncertainty_improvement_threshold,
            0.02,
        )

    def test_worsening_uncertainty_counts_as_stalled_despite_large_delta(self):
        for environment_class in (QuantEnvironment, NoquantEnvironment):
            with self.subTest(environment_class=environment_class.__module__):
                state = SimpleNamespace(
                    hybrid_enabled=True,
                    planner_initialized=True,
                    planner_submode="local",
                    latest_var_map=1.02 * np.ones((2, 2, 2), dtype=float),
                    previous_switch_uncertainty_map=np.ones((2, 2, 2), dtype=float),
                    last_uncertainty_map_delta=float("nan"),
                    last_uncertainty_map_improvement=float("nan"),
                    hybrid_uncertainty_improvement_threshold=0.02,
                    hybrid_uncertainty_stall_steps=2,
                    planner_stall_count=0,
                )

                environment_class._update_hybrid_planner_submode(
                    state,
                    planner_submode_before_step="local",
                    map_updated=True,
                )

                self.assertAlmostEqual(state.last_uncertainty_map_delta, 0.02)
                self.assertAlmostEqual(state.last_uncertainty_map_improvement, -0.02)
                self.assertEqual(state.planner_stall_count, 1)

    def test_improvement_threshold_flag_uses_canonical_name(self):
        for parse_args in (parse_quant_args, parse_noquant_args):
            with self.subTest(parse_args=parse_args.__module__):
                args = parse_args(
                    ["--hybrid_uncertainty_improvement_threshold", "0.021"]
                )

                self.assertAlmostEqual(
                    args.hybrid_uncertainty_improvement_threshold,
                    0.021,
                )


if __name__ == "__main__":
    unittest.main()
