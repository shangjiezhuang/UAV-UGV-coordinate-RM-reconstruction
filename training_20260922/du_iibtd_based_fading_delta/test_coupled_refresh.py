import importlib
import contextlib
import io
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from du_iibtd_based_fading_delta.uncertainty_refresh import (
    append_uncertainty_window, update_hybrid_planner_submode,
)
from du_iibtd_based_fading_delta.evaluation_common import apply_saved_config
from du_iibtd_based_fading_delta.shared import reconstruction as sm


class CoupledRefreshTests(unittest.TestCase):
    def test_signed_sum_boundary_and_rolling_window(self):
        # Exact 3% is sufficient; negative improvement must not be clipped.
        for norms, triggered in [([100, 99, 97.02], False), ([100,99,98], True),
                                  ([100,90,94.5], False), ([100,95,99.75], True),
                                  ([100,100,100], True), ([100,105,110], True)]:
            h, total, decision = append_uncertainty_window([], norms[0], norms[1])
            self.assertFalse(decision);self.assertTrue(np.isnan(total))
            h, total, decision = append_uncertainty_window(h, norms[1], norms[2])
            self.assertEqual(decision, triggered, (norms,total))
        h, total, decision = append_uncertainty_window([100,95,90.25],90.25,89.8)
        self.assertEqual(h,[95.,90.25,89.8]);self.assertFalse(decision)
        h,total,decision = append_uncertainty_window(h,89.8,89.7)
        self.assertTrue(decision)
        _,total,decision=append_uncertainty_window([0,0],0,0)
        self.assertEqual(total,0);self.assertTrue(decision)

    def test_defaults_saved_protocol_and_legacy_rejection(self):
        for variant in ['quant','noquant']:
            cm=importlib.import_module(f'du_iibtd_based_fading_delta.shared.{variant}.config')
            cfg=cm.Config()
            self.assertEqual(cfg.planner.hybrid_uncertainty_improvement_threshold,.03)
            self.assertEqual(cfg.planner.hybrid_uncertainty_window_updates,2)
            self.assertEqual(cfg.planner.hybrid_global_hold_intervals*cfg.planner.ensemble_refresh_interval,15)
            loaded=cm.Config();apply_saved_config(loaded,asdict(cfg));loaded.__post_init__()
            self.assertEqual(loaded.planner.reconstruction_refresh_mode,'local_to_global')
            utils=importlib.import_module(f'du_iibtd_based_fading_delta.shared.{variant}.utils')
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                utils.print_config_summary(cfg)
            self.assertIn('3.0%',stream.getvalue())
            with self.assertRaisesRegex(ValueError,'Legacy'):
                apply_saved_config(cfg,{'planner':{'hybrid_uncertainty_stall_steps':2}})
            cfg.planner.hybrid_uncertainty_window_updates=1
            with self.assertRaises(ValueError):cfg.__post_init__()

    def run_refits(self, truth):
        cm=importlib.import_module('du_iibtd_based_fading_delta.shared.quant.config')
        cfg=cm.Config();cfg.scene.grid_size=(2,2);cfg.scene.total_freq_bands_nums=2
        cfg.training.device='cpu';cfg.planner.iibtd_device='cpu'
        td=sm.IIBTD_opt(cfg,np.array([[0,0],[0,1],[1,0],[1,1]]),((0,1),(0,1)),np.ones((2,2),bool))
        if truth is not None:td.set_ground_truth(np.full((2,2,2),truth))
        model=SimpleNamespace();calls=[]
        def result(variance):
            return np.ones((2,2,2)),np.full((2,2,2),variance),None,{
                'member_models':[model],'member_observation_counts':np.array([1])}
        def full(**kwargs):calls.append(('full',kwargs['seed']));return result(1.)
        def incremental(**kwargs):calls.append(('incremental',None));return result(10.)
        sample=lambda:SimpleNamespace(position=np.array([0,0]),gamma=np.ones(2),omega=np.ones(2,int))
        with patch.object(sm,'ensemble_reconstruct_maps',side_effect=full), \
             patch.object(sm,'incremental_refresh_ensemble_models',side_effect=incremental), \
             patch.object(sm,'release_reconstruction_model',return_value=False):
            for _ in range(3):td.add_samples([sample()]);td.reconstruct()
            # Even a large rise never invokes an independent refit.
            self.assertEqual([x[0] for x in calls],['full','incremental','incremental'])
            before=(len(td._all_samples),td._reconstruct_round)
            state=td.full_refit_for_mode_switch()
            self.assertEqual(before,(len(td._all_samples),td._reconstruct_round))
            self.assertFalse(td._pending_samples)
            self.assertTrue(td.get_latest_ensemble_diagnostics()['mode_switch_refresh_triggered'])
            with self.assertRaisesRegex(RuntimeError,'Duplicate'):td.full_refit_for_mode_switch()
            td.add_samples([sample()])
            with self.assertRaisesRegex(RuntimeError,'completed'):td.full_refit_for_mode_switch()
            td.reset();self.assertEqual(td._last_switch_refit_round,-1)
        return calls,state

    def test_refit_uses_received_data_without_truth_decision_or_double_counting(self):
        a,sa=self.run_refits(1);b,sb=self.run_refits(100);c,sc=self.run_refits(None)
        self.assertEqual(a,b);self.assertEqual(a,c)
        np.testing.assert_array_equal(sa.spectrum_map,sb.spectrum_map)
        self.assertNotEqual(sa.nmse,sb.nmse)

    def test_two_valid_updates_switch_once_then_hold_and_reentry(self):
        calls=[]
        env=SimpleNamespace(hybrid_enabled=True,planner_initialized=True,planner_submode='local',
            previous_switch_uncertainty_map=np.ones((2,2,2)),latest_var_map=np.ones((2,2,2)),
            _local_uncertainty_norm_window=[],hybrid_uncertainty_improvement_threshold=.03,
            current_step=1,ensemble_events=[],radio_map_state=sm.RadioMapState(np.ones((2,2,2)),.5,3),
            global_steps_remaining=0,planner_stall_count=0)
        def switch(mode,hold_steps=0):
            env.planner_submode=mode;env.global_steps_remaining=hold_steps
            env._local_uncertainty_norm_window=[];calls.append(mode);return True
        env._switch_planner_submode=switch
        env._get_hybrid_global_hold_steps=lambda:15
        env._count_local_reentry_candidates=lambda:2
        env._get_hybrid_local_reentry_min_targets=lambda:2
        env._start_new_grid_plan=lambda:calls.append('plan')
        env._snapshot_global_switch_top_targets=lambda top_k:calls.append('targets')
        env._sync_cached_ensemble_state=lambda:setattr(env,'latest_var_map',np.full((2,2,2),.5))
        def refit():calls.append('refit');return sm.RadioMapState(np.ones((2,2,2)),.2,3)
        env.td=SimpleNamespace(full_refit_for_mode_switch=refit)
        update_hybrid_planner_submode(env,'local',True)
        for _ in range(4):update_hybrid_planner_submode(env,'local',False)
        self.assertEqual(calls,[])
        env.current_step=6;event={'step':6,'nmse':.5,'nmse_delta':.1,'reconstruction_data_bits':8e6}
        env.ensemble_events=[event]
        update_hybrid_planner_submode(env,'local',True)
        self.assertEqual(calls,['global','refit','targets','plan'])
        self.assertEqual(event['reconstruction_data_bits'],8e6)
        self.assertEqual(event['nmse'],.2);self.assertAlmostEqual(event['nmse_delta'],-.2)
        self.assertEqual(env.global_steps_remaining,15)
        np.testing.assert_array_equal(env.previous_switch_uncertainty_map,env.latest_var_map)
        for _ in range(14):update_hybrid_planner_submode(env,'global',False)
        self.assertEqual(env.planner_submode,'global')
        env._count_local_reentry_candidates=lambda:1
        update_hybrid_planner_submode(env,'global',False)
        self.assertEqual(env.planner_submode,'global')
        env._count_local_reentry_candidates=lambda:2
        update_hybrid_planner_submode(env,'global',False)
        self.assertEqual(env.planner_submode,'local');self.assertEqual(env._local_uncertainty_norm_window,[])

    def test_quant_and_noquant_share_reconstruction_implementation(self):
        q=importlib.import_module('du_iibtd_based_fading_delta.shared.quant.sim_models')
        n=importlib.import_module('du_iibtd_based_fading_delta.shared.noquant.sim_models')
        self.assertIs(q.IIBTD_opt,n.IIBTD_opt)


if __name__=='__main__':unittest.main()
