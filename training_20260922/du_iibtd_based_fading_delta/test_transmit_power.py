"""Power actions, channel consistency and battery accounting for both physical variants."""
from dataclasses import asdict
from importlib import import_module
from itertools import product
from types import SimpleNamespace
import unittest

import numpy as np

from du_iibtd_based_fading_delta.evaluation_common import apply_saved_config
from du_iibtd_based_fading_delta.power_control import dbm_to_watts, validate_power_config
from du_iibtd_based_fading_delta.PPO_AStar_2PathSupport.controller import _large_scale_capacity
from du_iibtd_based_fading_delta.Greedy_heuristic_quant.greedy_policy import GreedyPathPolicy


def stub(variant):
    package = 'du_iibtd_based_fading_delta.shared.' + variant
    config = import_module(package + '.config').Config()
    env_type = import_module(package + '.environment').UAVUGVEnvironment
    env = env_type.__new__(env_type)
    env.config = config
    env._init_transmit_power()
    env.uav_direction_ids = [0, 1, 2, 3, 4]
    env.uav_direction_choices = 5
    env.bandwidth_ratios = np.asarray(config.uav.bandwidth_ratios)
    env.num_bw_choices = len(env.bandwidth_ratios)
    env.ugv_action_size = 5
    env.uav_action_size = 5 * env.num_bw_choices * env.num_power_choices
    if variant == 'quant':
        env.quant_bits = np.asarray(config.uav.quant_bits)
        env.num_quant_choices = len(env.quant_bits)
        env.uav_action_size *= env.num_quant_choices
    env.uav_pos = np.array([1., 1.])
    env.ugv_pos = np.array([50., 50.])
    env.uav_step_count = 4
    env.uav_energy = config.uav.max_energy
    env.scene = SimpleNamespace(is_uav_position_valid=lambda pos: True, has_line_of_sight=lambda **kwargs: False, get_blocked_length_m=lambda **kwargs: 0.0)
    env._uav_action_mask_cache = {}
    env._uav_action_mask_cache_key = lambda: (1, 1)
    env._cache_store = lambda cache, key, value: cache.__setitem__(key, value)
    env._can_follow_direction = lambda **kwargs: True
    env._rollout_direction = lambda **kwargs: (kwargs['position'].copy(), 0 if kwargs['direction_idx'] == 0 else 4)
    env._set_bandwidth_info(.3)
    env.source_measurement_bits = config.comm.source_measurement_bits
    env.queue_capacity_bits = config.uav.queue_capacity_bits
    env.uav_data_queue = []
    sim_type = import_module(package + '.sim_models').SimDataGen
    env.sim_data = sim_type.__new__(sim_type)
    env.sim_data.config = config
    env.sim_data.rng = np.random.RandomState(42)
    config.comm.shadow_std_los_db = config.comm.shadow_std_nlos_db = 0
    return env


class TransmitPowerTests(unittest.TestCase):
    def test_default_action_product_and_greedy_encoding(self):
        policy = GreedyPathPolicy()
        for variant, count in [('quant', 135), ('noquant', 45)]:
            env = stub(variant)
            self.assertEqual(env.uav_action_size, count)
            self.assertEqual(env.get_action_dims()['uav_tx_power'], 3)
            expected = list(product(range(5), range(3), *([range(3)] if variant == 'quant' else []), range(3)))
            self.assertEqual([env._decode_uav_action(a) for a in range(count)], expected)
            for a, choices in enumerate(expected):
                d, bw = choices[:2]
                q = choices[2] if variant == 'quant' else None
                self.assertEqual(policy._encode_action(env, d, bw, q, choices[-1]), a)
                env._set_uav_action_controls(a)
                self.assertEqual(env.current_tx_power_dbm, [9., 12., 15.][choices[-1]])
            self.assertEqual(env.config.comm.tx_power_dbm, 12.)
            for invalid in (-1, count):
                with self.assertRaises(ValueError):
                    env._decode_uav_action(invalid)

    def test_exact_power_mask_boundary_and_geometric_cache(self):
        for variant in ('quant', 'noquant'):
            env = stub(variant)
            fixed_cost = env.config.uav.hover_power + env.config.uav.sensing_power_for_units(4)
            env.uav_energy = fixed_cost + dbm_to_watts(12)
            np.testing.assert_array_equal(env._build_uav_action_mask()[:3], [True, True, False])
            env.uav_energy -= 1e-6
            np.testing.assert_array_equal(env._build_uav_action_mask()[:3], [True, False, False])
            env.uav_energy = fixed_cost + dbm_to_watts(9) - 1e-6
            with self.assertRaisesRegex(RuntimeError, 'no feasible action'):
                env._build_uav_action_mask()
            env.uav_energy = 100
            self.assertTrue(env._build_uav_action_mask().all())

    def test_channel_and_planner_follow_power_without_stale_cache(self):
        for variant in ('quant', 'noquant'):
            env = stub(variant)
            # Isolate power/cache behavior from the outage gate. Under the
            # current dense-city loss, two low-power cases otherwise both
            # have zero capacity; outage boundaries have separate tests.
            env.config.comm.snr_outage_threshold_db = -100.0
            capacities = []
            snrs = []
            for power_index in (0, 2, 1, 0):
                env._set_transmit_power(power_index)
                channel = env._get_channel_info()
                capacity = _large_scale_capacity(env, (1, 1), (50, 50))
                self.assertAlmostEqual(channel.capacity_bps / 1e6, capacity / 1e6)
                capacities.append(capacity)
                snrs.append(channel.snr_db)
            self.assertGreater(capacities[1], capacities[2])
            self.assertGreater(capacities[2], capacities[0])
            self.assertEqual(capacities[0], capacities[3])
            self.assertAlmostEqual(snrs[1] - snrs[0], 6.)

    def test_energy_empty_partial_full_slot_and_outage(self):
        for variant in ('quant', 'noquant'):
            env = stub(variant)
            env.config.uav.step_duration = 2.
            for power in range(3):
                for queue, capacity, duration in [(0., 4., 0.), (2., 4., .5), (20., 4., 2.), (2., 0., 2.)]:
                    env.uav_energy = 100.
                    env.last_uav_step_energy = 0.
                    env._set_transmit_power(power)
                    env.ugv_channel_info = SimpleNamespace(capacity_bps=capacity)
                    env._charge_transmit_energy(queue)
                    expected = dbm_to_watts([9, 12, 15][power]) * duration
                    self.assertAlmostEqual(env.last_uav_tx_duration, duration)
                    self.assertAlmostEqual(env.last_uav_tx_energy, expected)
                    self.assertAlmostEqual(env.last_uav_step_energy, expected)
                    self.assertAlmostEqual(env.uav_energy + expected, 100.)
                    self.assertLessEqual(expected, env._transmit_energy_bounds()[power])
            env._reset_transmit_power()
            self.assertEqual(env.current_tx_power_dbm, 12.)
            self.assertEqual(env.last_uav_tx_energy, 0.)

    def test_legacy_and_new_saved_config_are_distinct(self):
        for variant in ('quant', 'noquant'):
            env = stub(variant)
            config = env.config
            raw = asdict(config)
            raw['comm']['tx_power_dbm'] = 1.
            del raw['comm']['tx_power_choices_dbm']
            del raw['comm']['tx_energy_enabled']
            with self.assertWarnsRegex(UserWarning, 'Legacy fixed-power'):
                apply_saved_config(config, raw)
            validate_power_config(config.comm)
            self.assertEqual(config.comm.tx_power_choices_dbm, [1.])
            self.assertFalse(config.comm.tx_energy_enabled)
            fresh = stub(variant).config
            apply_saved_config(config, asdict(fresh))
            self.assertEqual(config.comm.tx_power_choices_dbm, [9., 12., 15.])
            self.assertTrue(config.comm.tx_energy_enabled)

    def test_invalid_power_config_fails(self):
        for choices, default in [([], 27), ([20, 20], 20), ([30, 20], 20), ([float('nan')], 27), ([20, 30], 27)]:
            with self.assertRaises(ValueError):
                validate_power_config(SimpleNamespace(tx_power_choices_dbm=choices, tx_power_dbm=default, tx_energy_enabled=True))


if __name__ == '__main__':
    unittest.main()
