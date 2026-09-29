"""Transmission outage uses realized SNR; candidate planning stays nominal."""
from dataclasses import asdict
from importlib import import_module
import unittest
from unittest.mock import Mock

import numpy as np

from du_iibtd_based_fading_delta.evaluation_common import apply_saved_config
from du_iibtd_based_fading_delta.PPO_AStar_2PathSupport.controller import _large_scale_capacity
from du_iibtd_based_fading_delta.test_transmit_power import stub


class ReceivedSNRTests(unittest.TestCase):
    def test_received_gate_and_rate_share_the_same_shadow_draw(self):
        for variant in ('quant', 'noquant'):
            for shadow_db, outage in ((6., True), (-6., False)):
                with self.subTest(variant=variant, shadow=shadow_db):
                    env = stub(variant)
                    nominal = env._get_channel_info().large_scale_snr_db
                    env.config.comm.snr_outage_threshold_db = nominal + (-3. if shadow_db > 0 else 3.)
                    env.config.comm.shadow_std_nlos_db = 6.
                    rng = Mock()
                    rng.normal.return_value = shadow_db
                    env.sim_data.rng = rng
                    channel = env._get_channel_info()
                    rng.normal.assert_called_once_with(0, 6.)
                    self.assertAlmostEqual(channel.large_scale_snr_db, nominal)
                    self.assertAlmostEqual(channel.snr_db, nominal - shadow_db)
                    self.assertEqual(channel.outage, outage)
                    bandwidth = env.current_comm_units * env.config.uav.unit_bandwidth_hz
                    rate = bandwidth * np.log2(1 + 10 ** (channel.snr_db / 10))
                    self.assertAlmostEqual(channel.shannon_capacity_bps / 1e6, rate / 1e6)
                    self.assertAlmostEqual(channel.capacity_bps / 1e6, 0. if outage else rate / 1e6)

    def test_threshold_equality_is_service_not_outage(self):
        for variant in ('quant', 'noquant'):
            env = stub(variant)
            env.sim_data.rng = Mock(normal=Mock(return_value=6.))
            received = env._get_channel_info().snr_db
            for offset, outage in ((-1e-8, False), (0., False), (1e-8, True)):
                env.config.comm.snr_outage_threshold_db = received + offset
                self.assertEqual(env._get_channel_info().outage, outage)

    def test_legacy_nominal_gate_can_disagree_with_received_snr(self):
        for variant in ('quant', 'noquant'):
            for shadow_db, expected_outage in ((6., False), (-6., True)):
                env = stub(variant)
                env.config.comm.outage_snr_mode = 'nominal'
                nominal = env._get_channel_info().large_scale_snr_db
                env.config.comm.snr_outage_threshold_db = nominal + (-3. if shadow_db > 0 else 3.)
                env.sim_data.rng = Mock(normal=Mock(return_value=shadow_db))
                channel = env._get_channel_info()
                self.assertEqual(channel.outage, expected_outage)
                self.assertNotEqual(channel.outage, channel.snr_db < env.config.comm.snr_outage_threshold_db)

    def test_planning_remains_nominal_without_drawing_channel_noise(self):
        for variant in ('quant', 'noquant'):
            for shadow_db, actual_outage in ((6., True), (-6., False)):
                env = stub(variant)
                nominal = env._get_channel_info().large_scale_snr_db
                env.config.comm.snr_outage_threshold_db = nominal + (-3. if shadow_db > 0 else 3.)
                env.sim_data.rng = Mock(normal=Mock(return_value=shadow_db))
                actual = env._get_channel_info()
                self.assertEqual(actual.outage, actual_outage)
                env.sim_data.rng.normal.reset_mock()
                planned = _large_scale_capacity(env, (1, 1), (50, 50))
                env.sim_data.rng.normal.assert_not_called()
                self.assertEqual(planned > 0, actual_outage)

    def test_seeded_channel_sequence_is_unchanged_by_gate_selection(self):
        for variant in ('quant', 'noquant'):
            received, nominal = stub(variant), stub(variant)
            nominal.config.comm.outage_snr_mode = 'nominal'
            for env in (received, nominal):
                env.config.comm.shadow_std_nlos_db = 6.
                env.sim_data.rng = np.random.RandomState(321)
            for _ in range(12):
                a, b = received._get_channel_info(), nominal._get_channel_info()
                self.assertEqual(a.snr_db, b.snr_db)
                self.assertEqual(a.shannon_capacity_bps, b.shannon_capacity_bps)
                self.assertEqual(a.large_scale_snr_db, b.large_scale_snr_db)
            np.testing.assert_array_equal(received.sim_data.rng.normal(size=8), nominal.sim_data.rng.normal(size=8))

    def test_saved_configs_preserve_historical_outage_semantics(self):
        for variant in ('quant', 'noquant'):
            config = stub(variant).config
            self.assertEqual(config.comm.outage_snr_mode, 'received')
            raw = asdict(config)
            del raw['comm']['outage_snr_mode']
            with self.assertWarnsRegex(UserWarning, 'Legacy outage config'):
                apply_saved_config(config, raw)
            self.assertEqual(config.comm.outage_snr_mode, 'nominal')
            raw['comm']['outage_snr_mode'] = 'received'
            apply_saved_config(config, raw)
            self.assertEqual(config.comm.outage_snr_mode, 'received')
            raw['comm']['outage_snr_mode'] = 'nominal'
            apply_saved_config(config, raw)
            self.assertEqual(config.comm.outage_snr_mode, 'nominal')

    def test_cli_and_invalid_mode_validation(self):
        for variant in ('quant', 'noquant'):
            runner = import_module('du_iibtd_based_fading_delta.shared.' + variant + '.runner')
            for mode in ('received', 'nominal'):
                config = runner.build_config(runner.parse_args(['--outage_snr_mode', mode]))
                self.assertEqual(config.comm.outage_snr_mode, mode)
            config.comm.outage_snr_mode = 'unknown'
            with self.assertRaisesRegex(ValueError, 'outage_snr_mode'):
                config.__post_init__()


if __name__ == '__main__':
    unittest.main()
