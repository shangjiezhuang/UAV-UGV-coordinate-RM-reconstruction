"""Changing hidden uncertainty must not influence a fixed-target greedy decision."""
from types import SimpleNamespace
import unittest

import numpy as np

from .Greedy_heuristic_quant.greedy_policy import GreedyPathPolicy


def environment():
    n = 41
    rng = np.random.RandomState(123)
    return SimpleNamespace(
        Nx=n, Ny=n, uav_pos=np.array([20., 20.]), uav_step_count=4,
        uncertainty=SimpleNamespace(spatial_uncertainty=rng.uniform(1, 5, (n, n))),
        sampling_valid_mask=np.ones((n, n), dtype=bool),
        local_spatial_visit=np.zeros((n, n)),
        uav_direction_ids=[1, 2, 3, 4, 0], num_bw_choices=3,
        bandwidth_ratios=np.array([.3, .5, .6]), num_quant_choices=3,
        quant_bits=np.array([10, 8, 6]), default_quant_bits=10,
        num_power_choices=1, uav_action_size=45,
        config=SimpleNamespace(uav=SimpleNamespace(step_size=4, default_bw_ratio=.6)),
        scene=SimpleNamespace(is_uav_position_valid=lambda p: 0 <= p[0] < n and 0 <= p[1] < n),
        _get_motion_target_grid=lambda: (36, 20),
        _build_uav_action_mask=lambda: np.ones(45, dtype=bool))


class GreedyLocalUncertaintyTests(unittest.TestCase):
    def test_manhattan_boundary_and_uav_center(self):
        env=environment();p=GreedyPathPolicy();m=p._local_uncertainty_mask(env)
        self.assertEqual(int(m.sum()),481)
        self.assertTrue(m[35,20]);self.assertFalse(m[36,20]);self.assertFalse(m[30,30])
        env.uav_pos=np.array([0.,0.]);m=p._local_uncertainty_mask(env)
        self.assertTrue(m[15,0]);self.assertFalse(m[16,0]);self.assertEqual(int(m.sum()),136)

    def test_hidden_values_do_not_change_normalization_or_actions(self):
        env=environment();p=GreedyPathPolicy();visible=p._local_uncertainty_mask(env)
        before=p._build_spatial_uncertainty(env);a=p.select_action(env);plan=dict(p.last_plan)
        env.uncertainty.spatial_uncertainty[~visible]=1e30
        env.local_spatial_visit[~visible]=1e9
        np.testing.assert_array_equal(before,p._build_spatial_uncertainty(env))
        self.assertEqual(a,p.select_action(env))
        self.assertEqual(plan['planned_score'],p.last_plan['planned_score'])
        self.assertEqual(p.last_plan['uncertainty_radius_cells'],15)
        self.assertTrue(np.all(before[~visible]==0))
        self.assertGreater(np.ptp(before[visible]),.9)

    def test_hidden_values_do_not_change_fallback_resource_rank(self):
        env=environment();p=GreedyPathPolicy();visible=p._local_uncertainty_mask(env)
        x=p._build_spatial_uncertainty(env)
        a=p._adaptive_uncertainty_rank(env,.5,x,None,None)
        x[~visible]=1e20
        self.assertEqual(a,p._adaptive_uncertainty_rank(env,.5,x,None,None))

    def test_variance_tensor_fallback_reads_only_local_values(self):
        env=environment();p=GreedyPathPolicy();visible=p._local_uncertainty_mask(env)
        env.latest_var_map=np.repeat(env.uncertainty.spatial_uncertainty[:,:,None],3,axis=2)
        expected=p._build_spatial_uncertainty(env);del env.uncertainty
        before=p._build_spatial_uncertainty(env)
        env.latest_var_map[~visible]=np.nan
        np.testing.assert_array_equal(before,p._build_spatial_uncertainty(env))
        np.testing.assert_allclose(expected,before,rtol=1e-14,atol=1e-14)

    def test_empty_valid_window_does_not_use_outside_statistics(self):
        env=environment();p=GreedyPathPolicy();visible=p._local_uncertainty_mask(env)
        env.sampling_valid_mask[visible]=False
        a=p._build_spatial_uncertainty(env)
        env.uncertainty.spatial_uncertainty[~visible]=-1e30
        np.testing.assert_array_equal(a,p._build_spatial_uncertainty(env))
        self.assertTrue(np.all(a==0))

    def test_visible_uncertainty_still_changes_direction_without_target(self):
        env=environment();env._get_motion_target_grid=lambda: None
        p=GreedyPathPolicy();u=env.uncertainty.spatial_uncertainty;u[:]=1
        u[21:25,20]=10
        p.select_action(env);self.assertEqual(p.last_plan['selected_direction'],1)
        u[:]=1;u[16:20,20]=10
        p.select_action(env);self.assertEqual(p.last_plan['selected_direction'],3)


if __name__=='__main__':
    unittest.main()
