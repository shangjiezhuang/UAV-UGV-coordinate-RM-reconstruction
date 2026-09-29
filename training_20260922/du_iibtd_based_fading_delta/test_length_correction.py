"""Geometric, link-budget, configuration and caching checks for length loss."""
from dataclasses import asdict
from importlib import import_module
import math
import json
import unittest
from unittest.mock import Mock

import numpy as np

from du_iibtd_based_fading_delta.channel_loss import excess_loss_db, link_blocked_length_m
from du_iibtd_based_fading_delta.evaluation_common import apply_saved_config
from du_iibtd_based_fading_delta.PPO_AStar_2PathSupport.controller import _large_scale_capacity
from du_iibtd_based_fading_delta.test_transmit_power import stub


class LengthCorrectionTests(unittest.TestCase):
    def scene(self, variant, roofs=((10, 25, 60.0),)):
        module = 'du_iibtd_based_fading_delta.shared.' + variant
        config = import_module(module + '.config').Config()
        config.scene.grid_size = (80, 3)
        heights = np.zeros((80, 3), dtype=float)
        for start, end, height in roofs:
            heights[start:end, 1] = height
        scene = import_module(module + '.sim_models').GridScene(
            config, occupancy_grid=heights > 0, building_heights=heights)
        return config, scene

    def test_hand_calculated_axis_length_disjoint_roofs_and_height_clearance(self):
        for variant in ('quant', 'noquant'):
            for roofs, expected in [(((10, 25, 60.),), 30.),
                                    (((10, 15, 60.), (25, 35, 60.)), 30.),
                                    (((10, 25, 10.),), 0.), ((), 0.)]:
                with self.subTest(variant=variant, roofs=roofs):
                    _, scene = self.scene(variant, roofs)
                    a, b = np.array([0., 1.]), np.array([79., 1.])
                    self.assertAlmostEqual(scene.get_blocked_length_m(a, b), expected)
                    self.assertEqual(scene.has_line_of_sight(a, b), expected == 0)

    def test_identical_roof_split_does_not_change_length(self):
        for variant in ('quant', 'noquant'):
            _, whole = self.scene(variant)
            _, split = self.scene(variant, ((10, 17, 60.), (17, 25, 60.)))
            a, b = np.array([0., 1.]), np.array([79., 1.])
            self.assertEqual(whole.get_blocked_length_m(a, b), split.get_blocked_length_m(a, b))

    def test_geometry_cache_reused_and_bounded(self):
        for variant in ('quant', 'noquant'):
            _, scene = self.scene(variant)
            a, b = np.array([0., 1.]), np.array([79., 1.])
            scene._get_supercover_line_cells = Mock(wraps=scene._get_supercover_line_cells)
            self.assertAlmostEqual(scene.get_blocked_length_m(a, b), 30.)
            self.assertAlmostEqual(scene.get_blocked_length_m(a, b), 30.)
            self.assertEqual(scene._get_supercover_line_cells.call_count, 1)
            scene._cache_max_entries = 2
            for x in (70., 71., 72., 73.):
                scene.get_blocked_length_m(a, np.array([x, 1.]))
                self.assertLessEqual(len(scene._blocked_length_cache), 2)
            self.assertEqual(scene.get_blocked_length_m(a, a), 0.)

    def test_los_and_disabled_profile_do_not_scan_roofs(self):
        for variant in ('quant', 'noquant'):
            config, scene = self.scene(variant)
            scene.get_blocked_length_m = Mock(side_effect=AssertionError('unexpected geometry scan'))
            a, b = np.array([0., 1.]), np.array([79., 1.])
            self.assertEqual(link_blocked_length_m(scene, config.comm, a, b, True), 0.)
            config.comm.nlos_length_loss_db_per_m = 0.
            self.assertEqual(link_blocked_length_m(scene, config.comm, a, b, False), 0.)

    def test_loss_cap_los_and_disabled_parameters(self):
        for variant in ('quant', 'noquant'):
            config, _ = self.scene(variant)
            self.assertAlmostEqual(excess_loss_db(config.comm, False, 30.), 29.6)
            self.assertAlmostEqual(excess_loss_db(config.comm, False, 200.), 67.)
            self.assertEqual(excess_loss_db(config.comm, True, 200.), 1.6)
            config.comm.nlos_length_loss_cap_db = 12.
            self.assertEqual(excess_loss_db(config.comm, False, 200.), 35.)
            config.comm.nlos_length_loss_cap_db = 0.
            self.assertEqual(excess_loss_db(config.comm, False, 200.), 23.)

    def test_actual_link_planner_and_diagnostics_agree_on_real_geometry(self):
        for variant in ('quant', 'noquant'):
            _, scene = self.scene(variant)
            env = stub(variant)
            env.scene = scene
            env.uav_pos, env.ugv_pos = np.array([0., 1.]), np.array([79., 1.])
            env.current_tx_power_dbm = 15.
            info = env._get_channel_info()
            self.assertFalse(info.los)
            self.assertAlmostEqual(info.blocked_length_m, 30.)
            self.assertAlmostEqual(info.length_correction_db, 6.6)
            self.assertAlmostEqual(info.excess_loss_db, 29.6)
            d3 = math.hypot(158., 50.)
            expected_loss = 20 * math.log10(d3) + 20 * math.log10(3.5e9) - 147.55 + 29.6
            self.assertAlmostEqual(info.path_loss_db, expected_loss)
            self.assertAlmostEqual(info.capacity_bps / 1e6,
                _large_scale_capacity(env, (0, 1), (79, 1)) / 1e6)
            env.config.comm.nlos_length_loss_db_per_m = 0.
            before = env._get_channel_info()
            self.assertAlmostEqual(before.snr_db - info.snr_db, 6.6)

    def test_legacy_fixed_twenty_does_not_silently_enable_length_loss(self):
        for variant in ('quant', 'noquant'):
            config, _ = self.scene(variant)
            raw = asdict(config)
            raw['comm']['los_excess_db'] = 1.0
            raw['comm']['nlos_excess_db'] = 20.0
            del raw['comm']['nlos_length_loss_db_per_m']
            del raw['comm']['nlos_length_loss_cap_db']
            with self.assertWarnsRegex(UserWarning, 'Legacy length-loss config'):
                apply_saved_config(config, raw)
            self.assertEqual(config.comm.nlos_length_loss_db_per_m, 0.)
            self.assertEqual(excess_loss_db(config.comm, False, 100.), 20.)

    def test_uncapped_cli_json_and_long_roof_remain_uncapped(self):
        for variant in ('quant','noquant'):
            runner=import_module('du_iibtd_based_fading_delta.shared.'+variant+'.runner')
            config=runner.build_config(runner.parse_args(['--nlos_length_loss_cap_db','none']))
            raw=json.loads(json.dumps(asdict(config),allow_nan=False))
            self.assertIsNone(raw['comm']['nlos_length_loss_cap_db'])
            restored=type(config)();apply_saved_config(restored,raw)
            self.assertIsNone(restored.comm.nlos_length_loss_cap_db)
            self.assertAlmostEqual(excess_loss_db(restored.comm,False,200),67.)
            _,scene=self.scene(variant,((10,75,60.),))
            length=link_blocked_length_m(scene,restored.comm,np.array([0.,1.]),np.array([79.,1.]),False)
            self.assertAlmostEqual(length,130.)
            self.assertAlmostEqual(excess_loss_db(restored.comm,False,length),51.6)

    def test_saved_cli_and_invalid_coefficients(self):
        for variant in ('quant', 'noquant'):
            runner = import_module('du_iibtd_based_fading_delta.shared.' + variant + '.runner')
            config = runner.build_config(runner.parse_args([
                '--nlos_length_loss_db_per_m', '0.1', '--nlos_length_loss_cap_db', '12']))
            restored = type(config)()
            apply_saved_config(restored, asdict(config))
            self.assertEqual(restored.comm.nlos_length_loss_db_per_m, .1)
            self.assertEqual(restored.comm.nlos_length_loss_cap_db, 12.)
            self.assertEqual((restored.comm.shadow_std_los_db, restored.comm.shadow_std_nlos_db), (2., 6.))
            for key in ('nlos_length_loss_db_per_m', 'nlos_length_loss_cap_db'):
                for value in (-1., float('nan'), float('inf')):
                    invalid = type(config)()
                    setattr(invalid.comm, key, value)
                    with self.assertRaisesRegex(ValueError, key):
                        invalid.__post_init__()


if __name__ == '__main__':
    unittest.main()
