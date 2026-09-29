"""Focused contracts for the reward mini-screen information terms."""

import unittest

from du_iibtd_based_fading_delta.shared.noquant.environment import (
    compute_information_shaping_rewards as compute_noquant_rewards,
)
from du_iibtd_based_fading_delta.shared.quant.environment import (
    compute_information_shaping_rewards as compute_quant_rewards,
)


class InformationShapingRewardTest(unittest.TestCase):
    reward_functions = (compute_noquant_rewards, compute_quant_rewards)

    def test_partial_novelty_gets_frequency_aware_reward(self):
        for reward_fn in self.reward_functions:
            with self.subTest(reward_fn=reward_fn.__module__):
                r_novel, r_repeat, full_repeat = reward_fn(
                    novelty_ratio=0.5,
                    observed_band_count=6,
                    sampling_blocked=False,
                    lambda_novel_info=0.05,
                    lambda_full_repeat=0.05,
                )
                self.assertAlmostEqual(r_novel, 0.025)
                self.assertEqual(r_repeat, 0.0)
                self.assertEqual(full_repeat, 0)

    def test_valid_zero_novelty_sample_is_full_repeat(self):
        for reward_fn in self.reward_functions:
            with self.subTest(reward_fn=reward_fn.__module__):
                r_novel, r_repeat, full_repeat = reward_fn(
                    novelty_ratio=0.0,
                    observed_band_count=3,
                    sampling_blocked=False,
                    lambda_novel_info=0.05,
                    lambda_full_repeat=0.05,
                )
                self.assertEqual(r_novel, 0.0)
                self.assertAlmostEqual(r_repeat, -0.05)
                self.assertEqual(full_repeat, 1)

    def test_blocked_or_empty_sample_is_not_repeat(self):
        for reward_fn in self.reward_functions:
            for observed_band_count, sampling_blocked in ((4, True), (0, False)):
                with self.subTest(
                    reward_fn=reward_fn.__module__,
                    observed_band_count=observed_band_count,
                    sampling_blocked=sampling_blocked,
                ):
                    result = reward_fn(
                        novelty_ratio=0.0,
                        observed_band_count=observed_band_count,
                        sampling_blocked=sampling_blocked,
                        lambda_novel_info=0.05,
                        lambda_full_repeat=0.05,
                    )
                    self.assertEqual(result, (0.0, 0.0, 0))

    def test_novelty_ratio_is_clipped_to_physical_range(self):
        for reward_fn in self.reward_functions:
            with self.subTest(reward_fn=reward_fn.__module__):
                r_novel, r_repeat, full_repeat = reward_fn(
                    novelty_ratio=1.5,
                    observed_band_count=3,
                    sampling_blocked=False,
                    lambda_novel_info=0.05,
                    lambda_full_repeat=0.05,
                )
                self.assertAlmostEqual(r_novel, 0.05)
                self.assertEqual(r_repeat, 0.0)
                self.assertEqual(full_repeat, 0)


if __name__ == "__main__":
    unittest.main()
