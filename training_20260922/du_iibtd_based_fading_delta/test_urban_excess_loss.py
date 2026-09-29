"""Urban link budgets, planning consistency, and saved-run compatibility."""
from dataclasses import asdict
from importlib import import_module
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from du_iibtd_based_fading_delta.evaluation_common import apply_saved_config
from du_iibtd_based_fading_delta.PPO_AStar_2PathSupport.controller import _large_scale_capacity
from du_iibtd_based_fading_delta.test_transmit_power import stub
from du_iibtd_based_fading_delta import test_optimization_contract as contract_tests


class UrbanExcessLossTests(unittest.TestCase):
    def link_env(self, variant, los):
        env = stub(variant)
        env.uav_pos = np.array([10.0, 0.0])
        env.ugv_pos = np.array([0.0, 0.0])
        env.current_tx_power_dbm = 9.0
        env.scene.has_line_of_sight = lambda **kwargs: los
        return env

    def test_hand_calculated_los_and_nlos_budget_matches_planner(self):
        # 20 m horizontal / 50 m vertical; 3.5 GHz; 33.333 MHz; NF=8 dB.
        for variant in ('quant', 'noquant'):
            for los, expected_snr in ((True, 20.2158716812), (False, -1.1841283188)):
                with self.subTest(variant=variant, los=los):
                    env = self.link_env(variant, los)
                    actual = env._get_channel_info()
                    self.assertAlmostEqual(actual.snr_db, expected_snr, places=7)
                    self.assertAlmostEqual(actual.path_loss_db, 79.555340866 if los else 100.955340866, places=7)
                    planned = _large_scale_capacity(env, (10, 0), (0, 0))
                    self.assertAlmostEqual(planned / 1e6, actual.capacity_bps / 1e6, places=7)

    def test_communication_bandwidth_sets_integrated_noise(self):
        expected_noise = {0.3: -90.7712125472, 0.5: -92.0205999133, 0.6: -93.7815125038}
        for variant in ('quant', 'noquant'):
            for ratio, noise_dbm in expected_noise.items():
                env = self.link_env(variant, False)
                env._set_bandwidth_info(ratio)
                actual = env._get_channel_info()
                recovered_noise = env.current_tx_power_dbm - actual.path_loss_db - actual.snr_db
                self.assertAlmostEqual(recovered_noise, noise_dbm, places=7)

    def test_recovery_corridor_uses_the_same_nominal_link_budget(self):
        for variant in ('quant', 'noquant'):
            for los in (True, False):
                source = self.link_env(variant, los)
                source.config.comm.tx_power_dbm = 9.0
                corridor = contract_tests.OptimizationContractTest._comm_target_stub(
                    type(source), source.config, nx=11, ny=1,
                    scene=SimpleNamespace(has_line_of_sight=lambda **kwargs: los, get_blocked_length_m=lambda **kwargs: 0.0))
                corridor.current_comm_units = 8
                selected = corridor._select_ugv_path_corridor_target(source.uav_pos)
                self.assertEqual(selected, ((0, 0), los, True))
                cached_los, cached_snr = next(iter(corridor._ugv_service_link_cache.values()))
                self.assertEqual(cached_los, los)
                self.assertAlmostEqual(cached_snr, source._get_channel_info().large_scale_snr_db)

    def test_added_loss_can_change_actual_outage_while_planner_stays_nominal(self):
        for variant in ('quant', 'noquant'):
            env = self.link_env(variant, False)
            env.sim_data.rng = Mock(normal=Mock(return_value=8.0))
            actual = env._get_channel_info()
            self.assertTrue(actual.outage)
            self.assertEqual(actual.capacity_bps, 0.0)
            self.assertGreater(_large_scale_capacity(env, (10, 0), (0, 0)), 0.0)
            env.config.comm.nlos_excess_db = 15.0
            self.assertFalse(env._get_channel_info().outage)

    def test_old_saved_config_reproduces_zero_los_and_fifteen_nlos_loss(self):
        for variant in ('quant', 'noquant'):
            for los, old_snr in ((True, 21.8158716812), (False, 6.8158716812)):
                env = self.link_env(variant, los)
                raw = asdict(env.config)
                del raw['comm']['los_excess_db']
                raw['comm']['nlos_excess_db'] = 15.0
                with self.assertWarnsRegex(UserWarning, 'Legacy excess-loss config'):
                    apply_saved_config(env.config, raw)
                self.assertAlmostEqual(env._get_channel_info().snr_db, old_snr, places=7)

    def test_explicit_saved_profiles_and_cli_round_trip(self):
        for variant in ('quant', 'noquant'):
            runner = import_module('du_iibtd_based_fading_delta.shared.' + variant + '.runner')
            for los, nlos in ((1.0, 20.0), (1.6, 23.0), (0.0, 15.0)):
                config = runner.build_config(runner.parse_args([
                    '--los_excess_db', str(los), '--nlos_excess_db', str(nlos)]))
                restored = type(config)()
                apply_saved_config(restored, asdict(config))
                self.assertEqual((restored.comm.los_excess_db, restored.comm.nlos_excess_db), (los, nlos))

    def test_invalid_loss_is_rejected(self):
        for variant in ('quant', 'noquant'):
            config_type = import_module('du_iibtd_based_fading_delta.shared.' + variant + '.config').Config
            for name in ('los_excess_db', 'nlos_excess_db'):
                for value in (-1.0, float('nan'), float('inf')):
                    config = config_type()
                    setattr(config.comm, name, value)
                    with self.assertRaisesRegex(ValueError, name):
                        config.__post_init__()


if __name__ == '__main__':
    unittest.main()
