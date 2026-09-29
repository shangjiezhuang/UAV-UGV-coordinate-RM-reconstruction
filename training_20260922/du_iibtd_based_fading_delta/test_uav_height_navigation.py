"""Regression tests for resource accounting and fixed-altitude building avoidance."""

from importlib import import_module
import unittest

import numpy as np


class UAVHeightNavigationTests(unittest.TestCase):
    def make_scene(self, variant, height=55.0, closed=False):
        config = import_module(f'du_iibtd_based_fading_delta.shared.{variant}.config').Config()
        config.scene.grid_size = (9, 9)
        occupancy = np.zeros((9, 9), dtype=bool)
        occupancy[4, :9 if closed else 7] = True
        scene_class = import_module(f'du_iibtd_based_fading_delta.shared.{variant}.sim_models').GridScene
        scene = scene_class(config, occupancy_grid=occupancy, building_heights=occupancy * height)
        return config, scene

    def make_env(self, variant, height=55.0, closed=False):
        config, scene = self.make_scene(variant, height, closed)
        cls = import_module(f'du_iibtd_based_fading_delta.shared.{variant}.environment').UAVUGVEnvironment
        env = cls.__new__(cls)
        env.config, env.scene = config, scene
        env.Nx = env.Ny = 9
        env.K = 12
        env.uav_pos = np.array([1., 3.])
        env.uav_energy = 8500.
        env.uav_step_count = 4
        env._get_motion_target_grid = lambda: (7, 3)
        env.uav_direction_ids = np.arange(5)
        env.uav_direction_choices = 5
        env.bandwidth_ratios = np.asarray(config.uav.bandwidth_ratios)
        env.num_bw_choices = len(env.bandwidth_ratios)
        env._init_transmit_power()
        env.uav_action_size = 5 * env.num_bw_choices * env.num_power_choices
        env.source_measurement_bits = config.comm.source_measurement_bits
        if variant == 'quant':
            env.quant_bits = np.asarray(config.uav.quant_bits)
            env.num_quant_choices = len(env.quant_bits)
            env.current_quant_bits = 8
            env.uav_action_size *= env.num_quant_choices
        env._set_bandwidth_info(0.3)
        env._uav_action_mask_cache = {}
        env._mask_cache_max_entries = 128
        env.sampling_valid_mask = ~scene.occupancy
        return env

    def test_low_roofs_are_flyable_but_equal_and_higher_roofs_block(self):
        for variant in ('quant', 'noquant'):
            for height, allowed in ((49., True), (50., False), (55., False)):
                with self.subTest(variant=variant, height=height):
                    _, scene = self.make_scene(variant, height)
                    self.assertEqual(scene.is_uav_position_valid((4, 3)), allowed)
                    self.assertFalse(scene.is_ugv_position_valid((4, 3)))
                    self.assertTrue(scene.is_uav_position_valid((3, 3)))
                    self.assertFalse(scene.is_uav_position_valid((-1, 3)))

    def test_macro_move_checks_intermediate_cells_and_action_mask_blocks_wall(self):
        for variant in ('quant', 'noquant'):
            with self.subTest(variant=variant):
                env = self.make_env(variant)
                _, moved = env._move_uav(1)
                self.assertEqual(moved, 2)
                np.testing.assert_array_equal(env.uav_pos, [3, 3])
                mask = env._build_uav_action_mask().reshape(5, -1)
                self.assertFalse(mask[1].any())
                self.assertTrue(mask[2].all())
                _, moved = env._move_uav(1)  # Even a forced invalid action cannot cross.
                self.assertEqual(moved, 0)
                np.testing.assert_array_equal(env.uav_pos, [3, 3])
                low = self.make_env(variant, height=49.)
                _, moved = low._move_uav(1)
                self.assertEqual(moved, 4)
                np.testing.assert_array_equal(low.uav_pos, [5, 3])

    def test_forecast_detours_are_executable_and_reach_goal(self):
        for variant in ('quant', 'noquant'):
            env = self.make_env(variant)
            for axis in (0, 1):
                with self.subTest(variant=variant, axis=axis):
                    path = env.scene.uav_path_prefix((1, 3), (7, 3), macro_step=4, horizon=64, first_axis=axis)
                    self.assertTrue(path)
                    self.assertEqual(path[-1], (7, 3))
                    self.assertTrue(any(y >= 7 for x, y in path))
                    current = np.array([1., 3.])
                    for endpoint in path:
                        delta = np.asarray(endpoint) - current
                        self.assertEqual(np.count_nonzero(delta), 1)
                        direction = (1 if delta[0] > 0 else 3) if delta[0] else (2 if delta[1] > 0 else 4)
                        actual, _ = env._rollout_direction(current, direction, 4, env.scene.is_uav_position_valid, True)
                        np.testing.assert_array_equal(actual, endpoint)
                        current = actual

    def test_detour_progress_uses_flyable_distance(self):
        for variant in ('quant', 'noquant'):
            env = self.make_env(variant)
            before = env._uav_shortest_path_distance((3, 3), (7, 3))
            after = env._uav_shortest_path_distance((3, 7), (7, 3))
            self.assertEqual(before, 12.)
            self.assertEqual(after, 8.)  # Moving away in Manhattan distance is necessary progress.

    def test_unreachable_targets_are_excluded_even_in_fallback(self):
        for variant in ('quant', 'noquant'):
            env = self.make_env(variant, closed=True)
            self.assertTrue(np.isinf(env.scene.uav_shortest_path_distance((1, 3), (7, 3))))
            self.assertEqual(env.scene.uav_path_prefix((1, 3), (7, 3), macro_step=4, horizon=8, first_axis=0), ())
            self.assertFalse(env._uav_reachable_sampling_mask()[7, 3])
            env.sampled_mask = np.zeros((9, 9, 12), dtype=bool)
            env.latest_var_map = np.ones((9, 9, 12))
            env.target_count = 1
            env._select_center_freq_for_grid = lambda *args: 0
            candidates = np.zeros((9, 9), dtype=bool)
            candidates[7, 3] = True
            targets = env._build_fallback_targets(candidates)
            self.assertTrue(targets)
            self.assertTrue(all(t.gx < 4 for t in targets))

    def test_ref32_keeps_eight_mbit_raw_payload_and_scales_quantized_bits(self):
        for variant in ('quant', 'noquant'):
            env = self.make_env(variant)
            self.assertEqual(env.config.uav.total_bandwidth, 50e6)
            self.assertEqual(env.source_measurement_bits, 32)
            self.assertEqual(env.config.comm.data_per_sample / env.source_measurement_bits, 250000)
            for ratio, count, comm_mhz in ((.3, 4, 100/3), (.5, 6, 25), (.6, 8, 50/3)):
                env._set_bandwidth_info(ratio)
                self.assertEqual(env.sensing_band_num, count)
                self.assertAlmostEqual(env.current_comm_units * env.config.uav.unit_bandwidth_hz / 1e6, comm_mhz)
                if variant == 'quant':
                    for bits, per_band_mbit in ((10, 2.5), (8, 2), (6, 1.5)):
                        env.current_quant_bits = bits
                        self.assertEqual(env._current_sample_packet_bits(), count * per_band_mbit * 1e6)
                else:
                    self.assertEqual(count * env.config.comm.data_per_sample, count * 8e6)


if __name__ == '__main__':
    unittest.main()
