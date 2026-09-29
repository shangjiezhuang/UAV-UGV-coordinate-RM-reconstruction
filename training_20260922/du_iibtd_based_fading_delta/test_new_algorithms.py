"""Focused implementation tests for PPO+A*, HAPPO, and MAPPO-CF."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from du_iibtd_based_fading_delta.HAPPO.buffer import RolloutBuffer
from du_iibtd_based_fading_delta.HAPPO.happo import HAPPO, update_compound_factor
from du_iibtd_based_fading_delta.HAPPO.networks import HAPPOPolicy
from du_iibtd_based_fading_delta.MAPPO_CF.mappo_cf import MAPPOCF, counterfactual_advantage
from du_iibtd_based_fading_delta.MAPPO_CF.networks import MAPPOCFPolicy
from du_iibtd_based_fading_delta.PPO_AStar.controller import (
    astar_grid_path,
    select_astar_ugv_action,
)
from du_iibtd_based_fading_delta.PPO_AStar_2PathSupport.controller import (
    select_two_path_support_target,
    two_axis_path_prefixes,
)
from du_iibtd_based_fading_delta.PPO_AStar_2PathSupportRecovery.controller import (
    update_recovery_state,
)
from du_iibtd_based_fading_delta.shared.quant.environment import (
    UAVUGVEnvironment as QuantEnvironment,
)
from du_iibtd_based_fading_delta.shared.noquant.environment import (
    UAVUGVEnvironment as NoQuantEnvironment,
)
from du_iibtd_based_fading_delta.fair_training import (
    training_reward_for_trainer,
    ugv_control_mode_for_trainer,
)
from du_iibtd_based_fading_delta.ugv_control import resolve_ugv_action, uses_service_action_mask


def _tiny_config():
    return SimpleNamespace(
        device="cpu",
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        use_feature_norm=False,
        use_orthogonal_init=False,
        lr_actor=1e-3,
        lr_critic=1e-3,
        num_epochs=1,
        num_minibatches=2,
        entropy_coef=0.0,
        value_loss_coef=0.5,
        clip_epsilon=0.2,
        max_grad_norm=0.5,
    )


def _rollout(policy, obs_dims, action_dims):
    buffer = RolloutBuffer(
        rollout_length=4,
        num_envs=1,
        obs_dims=obs_dims,
        action_dims=action_dims,
        gamma=0.99,
        gae_lambda=0.95,
    )
    rng = np.random.RandomState(7)
    for step in range(4):
        uav_obs = rng.normal(size=(1, obs_dims["uav_obs"])).astype(np.float32)
        ugv_obs = rng.normal(size=(1, obs_dims["ugv_obs"])).astype(np.float32)
        critic_state = rng.normal(size=(1, obs_dims["critic_state"])).astype(
            np.float32
        )
        uav_mask = np.ones((1, action_dims["uav_action"]), dtype=bool)
        ugv_mask = np.ones((1, action_dims["ugv_action"]), dtype=bool)
        action = policy.get_actions(
            uav_obs,
            ugv_obs,
            critic_state,
            uav_mask,
            ugv_mask,
        )
        terminal = np.asarray([step == 3], dtype=np.float32)
        buffer.add(
            uav_obs=uav_obs,
            ugv_obs=ugv_obs,
            critic_state=critic_state,
            uav_action=action["uav_action"],
            ugv_action=action["ugv_action"],
            uav_log_prob=action["uav_log_prob"],
            ugv_log_prob=action["ugv_log_prob"],
            uav_action_mask=uav_mask,
            ugv_action_mask=ugv_mask,
            reward=np.asarray([1.0 + step], dtype=np.float32),
            value=action["value"],
            done=terminal,
            terminated=terminal,
            truncated=np.zeros(1, dtype=np.float32),
            timeout_value=np.zeros(1, dtype=np.float32),
        )
    buffer.compute_returns_and_advantages(np.zeros(1, dtype=np.float32))
    return buffer


class _AStarEnv:
    def __init__(self):
        self.walkable_mask = np.ones((5, 5), dtype=bool)
        self.walkable_mask[1, 0] = False
        self._ugv_component_labels = np.zeros((5, 5), dtype=np.int32)
        self._ugv_component_labels[1, 0] = -1
        self.ugv_pos = np.asarray([0.0, 0.0])
        self.current_step = 1

    @staticmethod
    def _nearest_component_road_cell_to_reference(reference, component):
        del component
        return tuple(np.rint(reference).astype(int).tolist())


class NewAlgorithmTest(unittest.TestCase):
    def test_clean_observation_dimensions_match_across_variants(self):
        obs_config = SimpleNamespace(
            include_remaining_time=True,
            include_quant_context=True,
            num_planner_features=5,
        )
        for env_class in (QuantEnvironment, NoQuantEnvironment):
            env = object.__new__(env_class)
            env.config = SimpleNamespace(obs=obs_config)
            env_class._setup_observation_spaces(env)
            self.assertEqual(
                env.get_obs_dims(),
                {"uav_obs": 14, "ugv_obs": 14, "critic_state": 19},
            )
            self.assertEqual(env._extract_auxiliary_obs_features().size, 0)

    def test_uav_action_mask_has_no_episode_horizon_dependency(self):
        for env_class, action_size in ((QuantEnvironment, 1), (NoQuantEnvironment, 1)):
            env = object.__new__(env_class)
            env.config = SimpleNamespace(
                uav=SimpleNamespace(
                    sensing_units_for_ratio=lambda ratio: 1,
                    sensing_power_for_units=lambda units: 1.0,
                    flight_power=3.0,
                    hover_power=2.0,
                    step_duration=1.0,
                ),
                comm=SimpleNamespace(tx_power_choices_dbm=[30.0], tx_power_dbm=30.0, tx_energy_enabled=True),
            )
            env._init_transmit_power()
            env.bandwidth_ratios = np.asarray([0.5])
            env.num_bw_choices = 1
            env.num_quant_choices = 1
            env.uav_action_size = action_size
            env.uav_direction_choices = 1
            env.uav_direction_ids = [0]
            env.uav_step_count = 1
            env.uav_energy = 4.0
            env.uav_pos = np.zeros(2, dtype=float)
            env.scene = SimpleNamespace(is_uav_position_valid=lambda pos: True)
            env._uav_action_mask_cache = {}
            env._uav_action_mask_cache_key = lambda: (0,)
            env._can_follow_direction = lambda **kwargs: True
            env._rollout_direction = lambda **kwargs: (env.uav_pos, 0)
            env._cache_store = lambda cache, key, value: cache.__setitem__(key, value)
            mask = env_class._build_uav_action_mask(env)
            np.testing.assert_array_equal(mask, np.asarray([True]))

    def test_astar_path_avoids_obstacle_and_replans_on_target_change(self):
        env = _AStarEnv()
        path = astar_grid_path(env.walkable_mask, (0, 0), (2, 0))
        self.assertEqual(path[0], (0, 0))
        self.assertEqual(path[-1], (2, 0))
        self.assertNotIn((1, 0), path)

        self.assertEqual(select_astar_ugv_action(env, (2, 0)), 2)
        state = env._ppo_astar_plan_state
        self.assertEqual(state.replans, 1)
        env.ugv_pos = np.asarray([0.0, 1.0])
        env.current_step = 2
        self.assertEqual(select_astar_ugv_action(env, (2, 0)), 1)
        self.assertEqual(state.replans, 1)
        env.current_step = 3
        self.assertEqual(select_astar_ugv_action(env, (0, 4)), 2)
        self.assertEqual(state.replans, 2)

    def test_new_trainer_contracts(self):
        self.assertEqual(ugv_control_mode_for_trainer("happo"), "policy")
        self.assertEqual(ugv_control_mode_for_trainer("mappo_cf"), "policy")
        self.assertEqual(
            ugv_control_mode_for_trainer("ppo_astar"),
            "astar_target",
        )
        self.assertEqual(
            ugv_control_mode_for_trainer("ppo_astar_support"),
            "astar_support",
        )
        self.assertEqual(
            ugv_control_mode_for_trainer("ppo_astar_2path_support"),
            "astar_2path_support",
        )
        self.assertEqual(
            ugv_control_mode_for_trainer("ppo_astar_2path_support_recovery"),
            "astar_2path_support_recovery",
        )
        config = SimpleNamespace(
            planner=SimpleNamespace(
                ugv_control_mode="astar_target",
                ugv_service_action_mask=False,
            )
        )
        self.assertFalse(uses_service_action_mask(config))
        reward = np.asarray([4.0], dtype=np.float32)
        adjusted = training_reward_for_trainer(
            "ppo_astar", reward, [{"r_ugv_progress": 1.5}]
        )
        np.testing.assert_allclose(adjusted, [2.5])

    def test_two_path_forecast_uses_two_axis_priorities_and_one_batch(self):
        paths = two_axis_path_prefixes(
            (0, 0),
            (6, 5),
            macro_step=4,
            horizon=3,
        )
        self.assertEqual(
            paths,
            (
                ((4, 0), (6, 0), (6, 4)),
                ((0, 4), (0, 5), (4, 5)),
            ),
        )
        self.assertEqual(sum(len(path) for path in paths), 2 * 3)

        merged = two_axis_path_prefixes(
            (0, 0),
            (4, 0),
            macro_step=4,
            horizon=3,
        )
        self.assertEqual(merged, (((4, 0),),))

    def test_target_and_current_support_astar_receive_different_goals(self):
        target_env = _AStarEnv()
        target_env.walkable_mask[1, 0] = True
        target_env._ugv_component_labels[1, 0] = 0
        target_env.config = SimpleNamespace(
            planner=SimpleNamespace(ugv_control_mode="astar_target")
        )
        action, mode = resolve_ugv_action(
            target_env,
            0,
            controller_target_grid=(4, 0),
            support_target_grid=(0, 4),
        )
        self.assertEqual((action, mode), (1, "astar_target"))
        self.assertEqual(target_env._ppo_astar_plan_state.goal, (4, 0))

        support_env = _AStarEnv()
        support_env.walkable_mask[1, 0] = True
        support_env._ugv_component_labels[1, 0] = 0
        support_env.config = SimpleNamespace(
            planner=SimpleNamespace(ugv_control_mode="astar_support")
        )
        action, mode = resolve_ugv_action(
            support_env,
            0,
            controller_target_grid=(4, 0),
            support_target_grid=(0, 4),
        )
        self.assertEqual((action, mode), (2, "astar_support"))
        self.assertEqual(support_env._ppo_astar_plan_state.goal, (0, 4))

    def test_two_path_support_scores_only_two_batch_sized_road_candidate_sets(self):
        env = _AStarEnv()
        env.walkable_mask = np.ones((8, 8), dtype=bool)
        env._ugv_component_labels = np.zeros((8, 8), dtype=np.int32)
        env.uav_pos = np.asarray([0.0, 0.0])
        env.uav_step_count = 4
        env.current_comm_units = 4
        env._get_motion_target_grid = lambda: (6, 5)
        env.scene = SimpleNamespace(
            has_line_of_sight=lambda uav_position, ugv_position: True
        )
        env.config = SimpleNamespace(
            planner=SimpleNamespace(
                ensemble_refresh_interval=3,
                ugv_comm_expanded_path_horizon=20,
            ),
            uav=SimpleNamespace(unit_bandwidth_hz=1e6),
            comm=SimpleNamespace(
                noise_figure_db=7.0,
                carrier_freq=2.4e9,
                los_excess_db=1.0,
                nlos_excess_db=20.0,
                tx_power_dbm=23.0,
                snr_outage_threshold_db=-5.0,
            ),
            scene=SimpleNamespace(
                grid_spacing=10.0,
                uav_height=50.0,
                ugv_height=1.5,
            ),
        )

        selected = select_two_path_support_target(env)

        self.assertIn(selected, env._ppo_astar_2path_candidate_goals)
        self.assertEqual(len(env._ppo_astar_2path_candidate_goals), 6)
        self.assertEqual(
            sum(len(path) for path in env._ppo_astar_2path_forecast_paths),
            6,
        )
        cached_paths = env._ppo_astar_2path_forecast_paths
        env.uav_pos = np.asarray([1.0, 0.0])
        self.assertEqual(select_two_path_support_target(env), selected)
        self.assertEqual(env._ppo_astar_2path_forecast_paths, cached_paths)

        env.map_update_count = 1
        select_two_path_support_target(env)
        self.assertNotEqual(env._ppo_astar_2path_forecast_paths, cached_paths)

    def test_two_path_recovery_enters_on_outage_and_exits_with_hysteresis(self):
        env = SimpleNamespace(
            current_step=1,
            queue_bits=0.0,
            ugv_channel_info=SimpleNamespace(outage=False, capacity_bps=100.0),
            config=SimpleNamespace(
                planner=SimpleNamespace(
                    ugv_comm_backlog_threshold=0.5,
                    ugv_recovery_exit_backlog_threshold=0.2,
                    ugv_recovery_poor_service_steps=2,
                    ugv_recovery_min_hold_steps=2,
                    ugv_recovery_good_link_steps=2,
                    ugv_recovery_service_margin=1.0,
                ),
                uav=SimpleNamespace(step_duration=1.0),
            ),
        )
        env._queue_remaining_bits = lambda: env.queue_bits
        env._queue_backlog_norm = lambda: env.queue_bits / 1000.0
        env._current_sample_packet_bits = lambda: 10.0

        state = update_recovery_state(env)
        self.assertEqual(state.submode, "two_path")

        env.current_step = 2
        env.queue_bits = 100.0
        env.ugv_channel_info.outage = True
        env.ugv_channel_info.capacity_bps = 0.0
        state = update_recovery_state(env)
        self.assertEqual(state.submode, "support_recovery")
        self.assertEqual(state.switch_reason, "outage_queue")
        self.assertEqual(state.mode_switch, "two_path->support_recovery")

        env._ppo_astar_2path_target_cache = ("stale",)
        env.queue_bits = 0.0
        env.ugv_channel_info.outage = False
        env.ugv_channel_info.capacity_bps = 100.0
        env.current_step = 3
        self.assertEqual(update_recovery_state(env).submode, "support_recovery")
        env.current_step = 4
        state = update_recovery_state(env)
        self.assertEqual(state.submode, "two_path")
        self.assertEqual(state.mode_switch, "support_recovery->two_path")
        self.assertFalse(hasattr(env, "_ppo_astar_2path_target_cache"))

    def test_recovery_requires_two_poor_service_steps_at_high_backlog(self):
        env = SimpleNamespace(
            current_step=1,
            queue_bits=600.0,
            ugv_channel_info=SimpleNamespace(outage=False, capacity_bps=5.0),
            config=SimpleNamespace(
                planner=SimpleNamespace(
                    ugv_comm_backlog_threshold=0.5,
                    ugv_recovery_exit_backlog_threshold=0.2,
                    ugv_recovery_poor_service_steps=2,
                    ugv_recovery_min_hold_steps=2,
                    ugv_recovery_good_link_steps=2,
                    ugv_recovery_service_margin=1.0,
                ),
                uav=SimpleNamespace(step_duration=1.0),
            ),
        )
        env._queue_remaining_bits = lambda: env.queue_bits
        env._queue_backlog_norm = lambda: env.queue_bits / 1000.0
        env._current_sample_packet_bits = lambda: 10.0

        self.assertEqual(update_recovery_state(env).submode, "two_path")
        env.current_step = 2
        state = update_recovery_state(env)
        self.assertEqual(state.submode, "support_recovery")
        self.assertEqual(state.switch_reason, "backlog_poor_service")

    def test_recovery_mode_routes_to_current_support_goal(self):
        env = _AStarEnv()
        env.walkable_mask[1, 0] = True
        env._ugv_component_labels[1, 0] = 0
        env.current_step = 1
        env.config = SimpleNamespace(
            planner=SimpleNamespace(
                ugv_control_mode="astar_2path_support_recovery",
                ugv_comm_backlog_threshold=0.5,
                ugv_recovery_exit_backlog_threshold=0.2,
                ugv_recovery_poor_service_steps=2,
                ugv_recovery_min_hold_steps=2,
                ugv_recovery_good_link_steps=2,
                ugv_recovery_service_margin=1.0,
            ),
            uav=SimpleNamespace(step_duration=1.0),
        )
        env.ugv_channel_info = SimpleNamespace(outage=True, capacity_bps=0.0)
        env._queue_remaining_bits = lambda: 100.0
        env._queue_backlog_norm = lambda: 0.1

        action, mode = resolve_ugv_action(
            env,
            0,
            support_target_grid=(0, 4),
        )

        self.assertEqual((action, mode), (2, "astar_2path_support_recovery"))
        self.assertEqual(env._ppo_astar_plan_state.goal, (0, 4))
        self.assertEqual(
            env._ppo_astar_2path_recovery_submode,
            "support_recovery",
        )

    def test_happo_compound_factor_uses_new_old_ratio(self):
        factor = update_compound_factor(
            torch.tensor([1.0, 2.0]),
            torch.log(torch.tensor([0.4, 0.25])),
            torch.log(torch.tensor([0.2, 0.5])),
        )
        torch.testing.assert_close(factor, torch.tensor([2.0, 1.0]))

    def test_counterfactual_advantage_is_not_independent_gae(self):
        advantage = counterfactual_advantage(
            q_taken=torch.tensor([3.0]),
            all_action_q=torch.tensor([[1.0, 3.0]]),
            action_probabilities=torch.tensor([[0.25, 0.75]]),
        )
        torch.testing.assert_close(advantage, torch.tensor([0.5]))

    def test_happo_and_mappo_cf_complete_one_tiny_update(self):
        torch.manual_seed(3)
        obs_dims = {"uav_obs": 3, "ugv_obs": 4, "critic_state": 5}
        action_dims = {"uav_action": 3, "ugv_action": 5}
        config = _tiny_config()

        happo_policy = HAPPOPolicy(obs_dims, action_dims, config)
        happo_buffer = _rollout(happo_policy, obs_dims, action_dims)
        happo = HAPPO(happo_policy, config)
        happo._sample_agent_order = lambda: ("uav", "ugv")
        happo_metrics = happo.update(happo_buffer)
        self.assertTrue(np.isfinite(happo_metrics["value_loss"]))
        self.assertEqual(happo_metrics["happo_uav_first"], 1.0)

        cf_policy = MAPPOCFPolicy(obs_dims, action_dims, config)
        cf_buffer = _rollout(cf_policy, obs_dims, action_dims)
        cf_metrics = MAPPOCF(cf_policy, config).update(cf_buffer)
        for key in (
            "value_loss",
            "q_value_loss",
            "cf_uav_advantage_std",
            "cf_ugv_advantage_std",
        ):
            self.assertTrue(np.isfinite(cf_metrics[key]), key)


if __name__ == "__main__":
    unittest.main()
