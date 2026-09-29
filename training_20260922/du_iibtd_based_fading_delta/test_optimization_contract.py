import json
import os
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from DU_IIBTD_res_Sr_learn_nu.DU_IIBTD import _stable_batched_ridge_solve
from du_iibtd_based_fading_delta.analyze_observation_replays import summarize as summarize_replays
from du_iibtd_based_fading_delta.analyze_pareto_run import summarize_run
from du_iibtd_based_fading_delta.shared.evaluate_checkpoint import _optional_config_bool
from du_iibtd_based_fading_delta.shared.noquant.config import Config as NoQuantConfig
from du_iibtd_based_fading_delta.shared.noquant.environment import (
    UAVUGVEnvironment as NoQuantEnvironment,
)
from du_iibtd_based_fading_delta.MAPPO_noquant.networks import MAPPOPolicy as NoQuantPolicy
from du_iibtd_based_fading_delta.shared.noquant.sim_models import SimDataGen as NoQuantSimDataGen
from du_iibtd_based_fading_delta.shared.noquant.runner import update_pareto_eval_checkpoints
from du_iibtd_based_fading_delta.shared.quant.config import Config as QuantConfig
from du_iibtd_based_fading_delta.shared.quant.environment import UAVUGVEnvironment as QuantEnvironment
from du_iibtd_based_fading_delta.MAPPO_quant.networks import MAPPOPolicy as QuantPolicy
from du_iibtd_based_fading_delta.shared.quant.sim_models import SimDataGen as QuantSimDataGen
from du_iibtd_based_fading_delta.IPPO_quant.networks import IPPOPolicy as QuantIPPOPolicy
from du_iibtd_based_fading_delta.fair_baseline_policies import (
    GuidanceGreedyUGVOverridePolicy,
    RandomValidPolicy,
    make_baseline_policy,
    ugv_control_mode_for_baseline,
)
from du_iibtd_based_fading_delta.energy_tradeoff import summarize_energy_episode_records
from du_iibtd_based_fading_delta.evaluation_common import (
    add_cumulative_step_metrics,
    compact_eval_result,
    scene_artifacts_payload,
)
from du_iibtd_based_fading_delta.fair_training import (
    training_reward_for_trainer,
    ugv_control_mode_for_trainer,
)
from du_iibtd_based_fading_delta.uav_ppo import UAVPPOPolicy
from du_iibtd_based_fading_delta.ugv_control import resolve_ugv_action
from du_iibtd_based_fading_delta.scene_suite import (
    aggregate_eval_results,
    data_accounting_violation_bits,
    dominates_nmse_link,
    is_strict_pareto_feasible,
)


class _RecordingPolicy:
    def __init__(self):
        self.saved_paths = []

    def save(self, path: str) -> None:
        self.saved_paths.append(path)


class _StayPolicy:
    def get_single_action(self, *args, **kwargs):
        return {"uav_action": 0, "ugv_action": 0}


class _OpenScene:
    @staticmethod
    def get_blocked_length_m(uav_position, ugv_position) -> float:
        return 0.0

    @staticmethod
    def is_ugv_position_valid(position) -> bool:
        return True

    @staticmethod
    def is_uav_position_valid(position) -> bool:
        return True

    @staticmethod
    def has_line_of_sight(uav_position, ugv_position) -> bool:
        return True


class _GuidanceEnv:
    ugv_action_size = 5
    ugv_step_count = 1
    ugv_pos = np.array([0.0, 0.0])
    scene = _OpenScene()
    _offsets = {
        0: np.array([0.0, 0.0]),
        1: np.array([1.0, 0.0]),
        2: np.array([0.0, 1.0]),
        3: np.array([-1.0, 0.0]),
        4: np.array([0.0, -1.0]),
    }

    @staticmethod
    def _get_ugv_guidance_target_grid():
        return (2, 0), "communication_predictive", True, True

    def _rollout_direction(self, position, direction_idx, **kwargs):
        return np.asarray(position) + self._offsets[int(direction_idx)], int(direction_idx != 0)

    @staticmethod
    def _occupancy_shortest_path_distance(position, target) -> float:
        return float(np.abs(np.asarray(position) - np.asarray(target)).sum())


def _eval_point(nmse: float, link_bits: float) -> dict:
    return {
        "eval_mean_nmse": nmse,
        "eval_mean_link_transmitted_bits": link_bits,
        "eval_energy_failure_rate": 0.0,
        "eval_full_horizon_rate": 1.0,
        "eval_mean_service_completion_ratio": 1.0,
        "eval_min_scene_service_completion_ratio": 1.0,
        "eval_max_data_accounting_violation_bits": 0.0,
        "eval_mean_prefill_observed_band_units": 80.0,
        "eval_mean_prefill_equivalent_data_bits": 640e6,
    }


class OptimizationContractTest(unittest.TestCase):
    def test_frozen_step_artifacts_cover_all_episodes_and_cumulative_metrics(
        self,
    ) -> None:
        details = add_cumulative_step_metrics(
            {
                "data_produced_bits": [10.0, 10.0],
                "data_delivered_bits": [4.0, 8.0],
                "completed_packet_bits": [0.0, 10.0],
                "link_transmitted_bits": [5.0, 9.0],
                "channel_outage": [1.0, 0.0],
                "r_nmse": [1.0, 2.0],
                "r_queue": [-0.5, -0.25],
                "r_progress": [3.0, 4.0],
                "r_novel_info": [0.0, 0.0],
                "r_full_repeat": [-0.1, 0.0],
            }
        )
        np.testing.assert_allclose(details["team_reward"], [3.4, 5.75])
        np.testing.assert_allclose(
            details["cumulative_data_produced_bits"],
            [10.0, 20.0],
        )
        np.testing.assert_allclose(
            details["cumulative_outage_ratio"],
            [1.0, 0.5],
        )
        np.testing.assert_allclose(
            details["cumulative_service_completion_ratio"],
            [0.0, 0.5],
        )

        episode_artifacts = [
            {"episode_index": index, "step_details": details}
            for index in range(3)
        ]
        result = {
            "eval_scene_source": "radioseerselect",
            "eval_scene_sample_index": 8513,
            "eval_scene_sample_tag": "",
            "eval_reset_seed_base": 200042,
            "eval_episode_records": [{"episode_index": 0}],
            "eval_episode_artifacts": episode_artifacts,
        }
        artifacts = scene_artifacts_payload([result])
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(len(artifacts[0]["episodes"]), 3)
        self.assertNotIn("eval_episode_artifacts", compact_eval_result(result))

    def test_shared_runtime_is_canonical_and_legacy_mappo_paths_are_shims(self) -> None:
        from du_iibtd_based_fading_delta.MAPPO_noquant.config import Config as LegacyNoQuantConfig
        from du_iibtd_based_fading_delta.MAPPO_noquant.environment import (
            UAVUGVEnvironment as LegacyNoQuantEnvironment,
        )
        from du_iibtd_based_fading_delta.MAPPO_quant.config import Config as LegacyQuantConfig
        from du_iibtd_based_fading_delta.MAPPO_quant.environment import (
            UAVUGVEnvironment as LegacyQuantEnvironment,
        )

        self.assertIs(LegacyNoQuantConfig, NoQuantConfig)
        self.assertIs(LegacyQuantConfig, QuantConfig)
        self.assertIs(LegacyNoQuantEnvironment, NoQuantEnvironment)
        self.assertIs(LegacyQuantEnvironment, QuantEnvironment)
        for config in (NoQuantConfig(), QuantConfig()):
            self.assertIs(config.training, config.mappo)
            serialized = asdict(config)
            self.assertIn("mappo", serialized)
            self.assertNotIn("training", serialized)

    def test_noquant_observation_config_does_not_require_quant_context(self) -> None:
        config = NoQuantConfig()
        self.assertFalse(_optional_config_bool(config.obs, "include_quant_context"))
        self.assertFalse(_optional_config_bool(config.obs, "missing_flag"))

    def test_source_sampler_never_applies_hidden_dataset_quantization(self) -> None:
        source = np.asarray([[[0.1234567, 0.7654321, 0.3333333]]], dtype=float)
        for sim_type in (NoQuantSimDataGen, QuantSimDataGen):
            with self.subTest(sim=sim_type.__module__):
                sim = sim_type.__new__(sim_type)
                sim.ground_truth = source.copy()
                sim.Nx = 1
                sim.Ny = 1
                sampled = sim.get_data_at_newpos(
                    np.asarray([0.0, 0.0]),
                    add_noise=False,
                )
                np.testing.assert_array_equal(sampled, source[0, 0])
                with self.assertRaises(TypeError):
                    sim.get_data_at_newpos(
                        np.asarray([0.0, 0.0]),
                        quantized=True,
                    )

    def test_explicit_cuda_request_never_falls_back_to_cpu(self) -> None:
        config = SimpleNamespace(device="cuda:0")
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            for policy_cls in (QuantPolicy, NoQuantPolicy, UAVPPOPolicy):
                with self.subTest(policy=policy_cls.__module__):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "Refusing to fall back to CPU",
                    ):
                        policy_cls({}, {}, config)

    def test_trainer_to_ugv_controller_mapping_is_unambiguous(self) -> None:
        self.assertEqual(ugv_control_mode_for_trainer("mappo"), "policy")
        self.assertEqual(ugv_control_mode_for_trainer("ippo"), "policy")
        self.assertEqual(ugv_control_mode_for_trainer("ppo_fixed"), "fixed")
        self.assertEqual(
            ugv_control_mode_for_trainer("ppo_heuristic"),
            "legacy_heuristic",
        )
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

    def test_greedy_2path_baseline_reuses_astar_2path_controller(self) -> None:
        self.assertEqual(
            ugv_control_mode_for_baseline("greedy_astar_2path_support"),
            "astar_2path_support",
        )
        self.assertEqual(
            ugv_control_mode_for_baseline("greedy_2path"),
            "astar_2path_support",
        )

    def test_legacy_astar_alias_is_canonicalized_in_saved_config(self) -> None:
        for config in (NoQuantConfig(), QuantConfig()):
            config.planner.ugv_control_mode = "astar"
            config.__post_init__()
            self.assertEqual(config.planner.ugv_control_mode, "astar_target")
        env = SimpleNamespace(
            config=SimpleNamespace(
                planner=SimpleNamespace(ugv_control_mode="fixed")
            ),
            ugv_action_size=5,
        )
        self.assertEqual(resolve_ugv_action(env, 4), (0, "fixed"))

    def test_uav_only_controllers_exclude_environment_owned_ugv_progress(self) -> None:
        team_reward = np.asarray([2.5, -1.0], dtype=np.float32)
        infos = [
            {"r_ugv_progress": 0.5},
            {"r_ugv_progress": -2.0},
        ]

        np.testing.assert_allclose(
            training_reward_for_trainer("ppo_heuristic", team_reward, infos),
            np.asarray([2.0, 1.0], dtype=np.float32),
        )
        for trainer in (
            "ppo_astar",
            "ppo_astar_support",
            "ppo_astar_2path_support",
        ):
            with self.subTest(trainer=trainer):
                np.testing.assert_allclose(
                    training_reward_for_trainer(trainer, team_reward, infos),
                    np.asarray([2.0, 1.0], dtype=np.float32),
                )
        for trainer in ("mappo", "ippo", "ppo_fixed"):
            with self.subTest(trainer=trainer):
                self.assertIs(
                    training_reward_for_trainer(trainer, team_reward, infos),
                    team_reward,
                )

    def test_energy_tradeoff_extrema_keep_same_episode_bits(self) -> None:
        records = [
            {"energy_budget_j": 9000, "nmse": 0.20, "link_transmitted_bits": 2.0},
            {"energy_budget_j": 9000, "nmse": 0.10, "link_transmitted_bits": 7.0},
            {"energy_budget_j": 9000, "nmse": 0.30, "link_transmitted_bits": 1.0},
        ]
        summary = summarize_energy_episode_records(records)[0]
        np.testing.assert_allclose(
            summary["mean_nmse_pair"], [9000.0, 0.2, 10.0 / 3.0]
        )
        np.testing.assert_allclose(summary["min_nmse_pair"], [9000.0, 0.1, 7.0])
        np.testing.assert_allclose(summary["max_nmse_pair"], [9000.0, 0.3, 1.0])

    def test_du_iibtd_stable_solve_preserves_normal_path_and_recovers_singular_float32(self) -> None:
        normal_ata = torch.tensor(
            [[[2.0, 0.25], [0.25, 1.0]]],
            dtype=torch.float32,
        )
        normal_atb = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
        ridge = 1e-5
        expected = torch.linalg.solve(
            normal_ata + torch.eye(2).unsqueeze(0) * ridge,
            normal_atb.unsqueeze(-1),
        ).squeeze(-1)
        actual = _stable_batched_ridge_solve(normal_ata, normal_atb, ridge)
        torch.testing.assert_close(actual, expected)

        singular_float32 = torch.tensor(
            [[[1.0e8, 1.0e8], [1.0e8, 1.0e8]]],
            dtype=torch.float32,
        )
        singular_rhs = torch.tensor([[2.0e8, 2.0e8]], dtype=torch.float32)
        recovered = _stable_batched_ridge_solve(
            singular_float32,
            singular_rhs,
            ridge,
        )
        self.assertTrue(bool(torch.isfinite(recovered).all()))
        torch.testing.assert_close(
            recovered,
            torch.ones_like(recovered),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_based_defaults_freeze_prefill_and_path_corridor_support(self) -> None:
        for config in (NoQuantConfig(), QuantConfig()):
            self.assertEqual(config.planner.prefill_percent, 5.0)
            self.assertEqual(config.planner.ugv_comm_target_mode, "path_corridor")
            self.assertEqual(config.planner.ugv_comm_backlog_threshold, 0.5)
            self.assertEqual(config.planner.ugv_comm_local_path_horizon, 15)
            self.assertEqual(config.planner.ugv_comm_expanded_path_horizon, 20)
            self.assertEqual(config.planner.ugv_comm_corridor_width, 1)
            self.assertFalse(config.planner.ugv_service_action_mask)
            self.assertEqual(
                config.uav.sensing_units_for_ratio(config.uav.default_bw_ratio),
                8,
            )

    def test_pareto_archive_rejects_dominated_checkpoint(self) -> None:
        policy = _RecordingPolicy()
        with tempfile.TemporaryDirectory() as model_dir:
            frontier, saved = update_pareto_eval_checkpoints(
                policy, model_dir, [], 1, _eval_point(0.10, 1.0e9)
            )
            self.assertIsNotNone(saved)
            frontier, saved = update_pareto_eval_checkpoints(
                policy, model_dir, frontier, 2, _eval_point(0.12, 1.2e9)
            )
            self.assertIsNone(saved)
            self.assertEqual(len(frontier), 1)
            self.assertEqual(len(policy.saved_paths), 1)
            self.assertTrue(os.path.isfile(os.path.join(model_dir, "pareto_frontier.json")))

    def test_pareto_archive_keeps_real_tradeoff(self) -> None:
        policy = _RecordingPolicy()
        with tempfile.TemporaryDirectory() as model_dir:
            frontier, _ = update_pareto_eval_checkpoints(
                policy, model_dir, [], 1, _eval_point(0.10, 1.0e9)
            )
            frontier, saved = update_pareto_eval_checkpoints(
                policy, model_dir, frontier, 2, _eval_point(0.08, 1.4e9)
            )
            self.assertIsNotNone(saved)
            self.assertEqual(len(frontier), 2)

    def test_pareto_archive_rejects_energy_failure(self) -> None:
        policy = _RecordingPolicy()
        failed = _eval_point(0.05, 0.5e9)
        failed["eval_energy_failure_rate"] = 0.1
        with tempfile.TemporaryDirectory() as model_dir:
            frontier, saved = update_pareto_eval_checkpoints(
                policy, model_dir, [], 1, failed
            )
            self.assertIsNone(saved)
            self.assertEqual(frontier, [])
            self.assertEqual(policy.saved_paths, [])

    def test_pareto_archive_treats_service_completion_as_diagnostic(self) -> None:
        policy = _RecordingPolicy()
        incomplete = _eval_point(0.05, 0.5e9)
        incomplete["eval_mean_service_completion_ratio"] = 0.6
        with tempfile.TemporaryDirectory() as model_dir:
            frontier, saved = update_pareto_eval_checkpoints(
                policy, model_dir, [], 1, incomplete
            )
            self.assertIsNotNone(saved)
            self.assertEqual(len(frontier), 1)
            self.assertEqual(len(policy.saved_paths), 1)

    def test_pareto_archive_accepts_low_weakest_scene_completion(self) -> None:
        policy = _RecordingPolicy()
        incomplete = _eval_point(0.05, 0.5e9)
        incomplete["eval_min_scene_service_completion_ratio"] = 0.9
        with tempfile.TemporaryDirectory() as model_dir:
            frontier, saved = update_pareto_eval_checkpoints(
                policy, model_dir, [], 1, incomplete
            )
            self.assertIsNotNone(saved)
            self.assertEqual(len(frontier), 1)
            self.assertEqual(len(policy.saved_paths), 1)

    def test_pareto_archive_requires_measured_finite_data_accounting(self) -> None:
        policy = _RecordingPolicy()
        missing = _eval_point(0.05, 0.5e9)
        missing.pop("eval_max_data_accounting_violation_bits")
        nonfinite = _eval_point(0.05, 0.5e9)
        nonfinite["eval_max_data_accounting_violation_bits"] = float("nan")
        with tempfile.TemporaryDirectory() as model_dir:
            for update, point in enumerate(
                (missing, nonfinite),
                start=1,
            ):
                frontier, saved = update_pareto_eval_checkpoints(
                    policy, model_dir, [], update, point
                )
                self.assertIsNone(saved)
                self.assertEqual(frontier, [])
        self.assertEqual(policy.saved_paths, [])

    def test_scene_suite_retains_per_scene_pareto_metrics(self) -> None:
        scene_results = []
        for scene_index, nmse, link_bits in ((1, 0.10, 1.0e9), (2, 0.20, 2.0e9)):
            point = _eval_point(nmse, link_bits)
            point.update(
                {
                    "eval_scene_sample_index": scene_index,
                    "eval_scene_sample_tag": "",
                    "eval_num_episodes": 1,
                    "eval_reset_seed_base": 100 + scene_index,
                    "eval_best_nmse": nmse,
                    "eval_worst_nmse": nmse,
                    "eval_std_nmse": 0.0,
                    "eval_mean_return": 10.0 * scene_index,
                    "eval_std_return": 0.0,
                    "eval_mean_data_delivered_bits": link_bits * 0.8,
                    "eval_data_delivered_bits_at_best_nmse": link_bits * 0.8,
                    "eval_data_delivered_bits_at_worst_nmse": link_bits * 0.8,
                    "eval_mean_data_produced_bits": link_bits * 1.1,
                    "eval_mean_completed_packet_bits": link_bits * 0.9,
                    "eval_min_link_transmitted_bits": link_bits,
                    "eval_max_link_transmitted_bits": link_bits,
                    "eval_link_transmitted_bits_at_best_nmse": link_bits,
                    "eval_link_transmitted_bits_at_worst_nmse": link_bits,
                    "eval_nmse_at_min_link_transmitted_bits": nmse,
                    "eval_mean_outage_ratio": 0.1 * scene_index,
                    "eval_outage_ratio_at_best_nmse": 0.1 * scene_index,
                    "eval_outage_ratio_at_worst_nmse": 0.1 * scene_index,
                    "eval_service_completion_ratio_at_best_nmse": 1.0
                    - 0.1 * scene_index,
                    "eval_service_completion_ratio_at_worst_nmse": 1.0
                    - 0.1 * scene_index,
                    "eval_mean_uav_energy_remaining": 1000.0,
                    "eval_mean_ugv_comm_target_ratio": 1.0,
                    "eval_max_data_accounting_violation_bits": 0.0,
                    "eval_visualized_episode_index": 0,
                    "eval_visualized_reset_seed": 1000 + scene_index,
                }
            )
            scene_results.append(point)

        aggregate = aggregate_eval_results(scene_results, seed_base=100)
        self.assertEqual(aggregate["eval_per_scene_mean_link_transmitted_bits"], [1.0e9, 2.0e9])
        self.assertEqual(aggregate["eval_per_scene_mean_outage_ratio"], [0.1, 0.2])
        self.assertEqual(
            aggregate["eval_per_scene_mean_service_completion_ratio"],
            [1.0, 1.0],
        )
        self.assertEqual(aggregate["eval_min_scene_service_completion_ratio"], 1.0)
        self.assertAlmostEqual(aggregate["eval_mean_nmse"], 0.15)
        self.assertAlmostEqual(aggregate["eval_std_nmse"], 0.05)
        self.assertEqual(aggregate["eval_best_nmse"], 0.10)
        self.assertEqual(aggregate["eval_worst_nmse"], 0.20)
        self.assertEqual(aggregate["eval_num_episodes"], 2)
        self.assertEqual(aggregate["eval_num_episodes_per_scene"], [1, 1])
        self.assertEqual(aggregate["eval_visualized_reset_seed"], 1002)
        self.assertEqual(
            aggregate["eval_link_transmitted_bits_at_best_nmse"],
            1.0e9,
        )
        self.assertAlmostEqual(aggregate["eval_outage_ratio_at_best_nmse"], 0.1)
        self.assertAlmostEqual(aggregate["eval_outage_ratio_at_worst_nmse"], 0.2)
        self.assertAlmostEqual(
            aggregate["eval_service_completion_ratio_at_best_nmse"],
            0.9,
        )
        self.assertAlmostEqual(
            aggregate["eval_service_completion_ratio_at_worst_nmse"],
            0.8,
        )
        self.assertEqual(aggregate["eval_nmse_at_min_link_transmitted_bits"], 0.10)
        self.assertEqual(
            aggregate["eval_per_scene_metrics"][0]["eval_mean_prefill_observed_band_units"],
            80.0,
        )

    def test_guidance_greedy_override_moves_toward_target(self) -> None:
        policy = GuidanceGreedyUGVOverridePolicy(_StayPolicy(), _GuidanceEnv())
        result = policy.get_single_action(ugv_action_mask=np.ones(5, dtype=bool))
        self.assertEqual(result["ugv_action"], 1)

    @staticmethod
    def _comm_target_stub(
        environment_type,
        config,
        *,
        nx=3,
        ny=3,
        ugv_cell=(0, 0),
        scene=None,
    ):
        env = environment_type.__new__(environment_type)
        env.config = config
        env.Nx = int(nx)
        env.Ny = int(ny)
        env.ugv_pos = np.asarray(ugv_cell, dtype=float)
        env.ugv_step_count = env._grid_step_count(config.ugv.step_size)
        env.ugv_action_size = int(config.ugv.num_directions)
        env.scene = _OpenScene() if scene is None else scene
        env.walkable_mask = np.ones((env.Nx, env.Ny), dtype=bool)
        env._ugv_component_labels = np.zeros((env.Nx, env.Ny), dtype=np.int32)
        env._ugv_component_cells = {
            0: np.asarray(
                [(x, y) for x in range(env.Nx) for y in range(env.Ny)],
                dtype=np.int32,
            )
        }
        env.current_comm_units = (
            int(config.uav.total_bw_num)
            - int(config.uav.sensing_units_for_ratio(config.uav.default_bw_ratio))
        )
        env._ugv_comm_target_cache = {}
        env._ugv_service_link_cache = {}
        env._ugv_reference_road_goal_cache = {}
        env._ugv_local_distance_field_cache = {}
        env._ugv_action_mask_cache = {}
        env._mask_cache_max_entries = 1024
        return env

    def test_comm_target_keeps_nearest_service_feasible_ugv_cell(self) -> None:
        results = []
        for environment_type, config in (
            (NoQuantEnvironment, NoQuantConfig()),
            (QuantEnvironment, QuantConfig()),
        ):
            env = self._comm_target_stub(environment_type, config)
            result = env._select_ugv_path_corridor_target(
                np.array([2.0, 2.0])
            )
            results.append(result)
            self.assertEqual(result[0], (0, 0))
            self.assertTrue(result[1])
            self.assertTrue(result[2])

        self.assertEqual(results[0], results[1])

    def test_path_corridor_stops_at_expanded_horizon_when_service_is_impossible(self) -> None:
        for environment_type, config in (
            (NoQuantEnvironment, NoQuantConfig()),
            (QuantEnvironment, QuantConfig()),
        ):
            config.comm.snr_outage_threshold_db = 1_000.0
            env = self._comm_target_stub(environment_type, config, nx=20, ny=1)
            result = env._select_ugv_path_corridor_target(
                np.array([19.0, 0.0])
            )
            self.assertEqual(result[0], (19, 0))
            self.assertFalse(result[2])
            self.assertLessEqual(len(env._ugv_service_link_cache), 50)

    def test_path_corridor_finds_service_only_in_expanded_segment(self) -> None:
        class SelectiveLoSScene(_OpenScene):
            @staticmethod
            def has_line_of_sight(uav_position, ugv_position) -> bool:
                return int(round(float(np.asarray(ugv_position)[0]))) >= 17

        for environment_type, config in (
            (NoQuantEnvironment, NoQuantConfig()),
            (QuantEnvironment, QuantConfig()),
        ):
            config.comm.nlos_excess_db = 200.0
            env = self._comm_target_stub(
                environment_type,
                config,
                nx=20,
                ny=1,
                scene=SelectiveLoSScene(),
            )
            result = env._select_ugv_path_corridor_target(
                np.array([19.0, 0.0])
            )
            self.assertEqual(result[0], (17, 0))
            self.assertTrue(result[1])
            self.assertTrue(result[2])

    def test_bounded_path_and_distance_avoid_building(self) -> None:
        for environment_type, config in (
            (NoQuantEnvironment, NoQuantConfig()),
            (QuantEnvironment, QuantConfig()),
        ):
            env = self._comm_target_stub(environment_type, config, nx=4, ny=2)
            env.walkable_mask[1, 0] = False
            env._ugv_component_labels[1, 0] = -1
            env._ugv_component_cells[0] = np.argwhere(env.walkable_mask).astype(np.int32)
            path = env._bounded_astar_ugv_path((0, 0), (3, 0), 15)
            self.assertEqual(path[0], (0, 0))
            self.assertEqual(path[-1], (3, 0))
            self.assertNotIn((1, 0), path)
            self.assertEqual(
                env._occupancy_shortest_path_distance(
                    np.array([0.0, 0.0]),
                    np.array([3.0, 0.0]),
                ),
                5.0,
            )

    def test_path_corridor_bounded_route_can_follow_local_wall_edge(self) -> None:
        for environment_type, config in (
            (NoQuantEnvironment, NoQuantConfig()),
            (QuantEnvironment, QuantConfig()),
        ):
            config.comm.snr_outage_threshold_db = 1_000.0
            env = self._comm_target_stub(
                environment_type,
                config,
                nx=30,
                ny=21,
                ugv_cell=(5, 10),
            )
            env.walkable_mask[10, :20] = False
            (
                env._ugv_component_labels,
                env._ugv_component_cells,
            ) = env._build_ugv_walkable_components()

            target, _, service_feasible = env._select_ugv_path_corridor_target(
                np.array([25.0, 10.0])
            )

            self.assertEqual(target, (11, 16))
            self.assertFalse(service_feasible)

    def test_service_action_mask_uses_best_macro_action_intermediate_cell(self) -> None:
        for environment_type, config in (
            (NoQuantEnvironment, NoQuantConfig()),
            (QuantEnvironment, QuantConfig()),
        ):
            config.planner.ugv_service_action_mask = True
            env = self._comm_target_stub(
                environment_type,
                config,
                nx=15,
                ny=15,
                ugv_cell=(5, 5),
            )
            target = (3, 10)
            env._get_ugv_guidance_target_grid = lambda: (
                target,
                "communication_outage",
                True,
                True,
            )

            mask = env._build_ugv_action_mask()
            np.testing.assert_array_equal(
                mask,
                np.array([False, False, True, True, False]),
            )
            moved_steps = env._move_ugv(3, target_grid=target)
            self.assertEqual(moved_steps, 2)
            np.testing.assert_array_equal(env.ugv_pos, np.array([3.0, 5.0]))

            env.ugv_pos = np.asarray(target, dtype=float)
            np.testing.assert_array_equal(
                env._build_ugv_action_mask(),
                np.array([True, False, False, False, False]),
            )

            config.planner.ugv_service_action_mask = False
            env.ugv_pos = np.array([5.0, 5.0])
            np.testing.assert_array_equal(
                env._build_ugv_action_mask(),
                np.ones(5, dtype=bool),
            )
            self.assertEqual(env._move_ugv(3, target_grid=target), 5)
            np.testing.assert_array_equal(env.ugv_pos, np.array([0.0, 5.0]))

    def test_legacy_observation_flags_preserve_clean_dimensions(self) -> None:
        for environment_type, config in (
            (NoQuantEnvironment, NoQuantConfig()),
            (QuantEnvironment, QuantConfig()),
        ):
            env = environment_type.__new__(environment_type)
            env.config = config
            quant_flags = (False, True) if hasattr(config.obs, "include_quant_context") else (False,)
            for remaining_time in (False, True):
                for quant_context in quant_flags:
                    with self.subTest(
                        variant=environment_type.__module__,
                        remaining_time=remaining_time,
                        quant_context=quant_context,
                    ):
                        config.obs.include_remaining_time = remaining_time
                        if hasattr(config.obs, "include_quant_context"):
                            config.obs.include_quant_context = quant_context
                        env._setup_observation_spaces()
                        self.assertEqual(env.get_obs_dims(), {
                            "uav_obs": 14,
                            "ugv_obs": 14,
                            "critic_state": 19,
                        })

    @staticmethod
    def _auxiliary_obs_stub(environment_type, config, *, quantized: bool):
        env = environment_type.__new__(environment_type)
        env.config = config
        env.current_step = 50
        env.queue_capacity_bits = 160.0
        env.snr_norm_den = 30.0
        env.uav_data_queue = [
            SimpleNamespace(size_bits=100.0, transmitted_bits=20.0)
        ]
        env.ugv_channel_info = SimpleNamespace(
            large_scale_snr_db=-10.0,
            outage=True,
        )
        if quantized:
            env.current_quant_bits = 4
            env.source_measurement_bits = config.comm.source_measurement_bits
            env.sensing_band_num = 2
            env.source_max_packet_bits = 8.0 * config.comm.data_per_sample
        return env

    def test_auxiliary_observations_exclude_clock_and_quant_context(self) -> None:
        for environment_type, config, quantized in (
            (NoQuantEnvironment, NoQuantConfig(), False),
            (QuantEnvironment, QuantConfig(), True),
        ):
            config.obs.include_remaining_time = True
            if quantized:
                config.obs.include_quant_context = True
            env = self._auxiliary_obs_stub(
                environment_type, config, quantized=quantized,
            )
            bit_choices = config.uav.quant_bits if quantized else (None,)
            for step in (0, 50, config.training.episode_max_steps):
                for bits in bit_choices:
                    with self.subTest(
                        variant=environment_type.__module__, step=step, bits=bits,
                    ):
                        env.current_step = step
                        if quantized:
                            env.current_quant_bits = bits
                        self.assertEqual(env._extract_auxiliary_obs_features().shape, (0,))

    def test_queue_observation_and_penalty_share_remaining_bit_ratio(self) -> None:
        for environment_type, config in (
            (NoQuantEnvironment, NoQuantConfig()),
            (QuantEnvironment, QuantConfig()),
        ):
            env = environment_type.__new__(environment_type)
            env.config = config
            env.queue_capacity_bits = 160.0
            env.uav_data_queue = [
                SimpleNamespace(size_bits=100.0, transmitted_bits=20.0)
            ]

            self.assertAlmostEqual(env._queue_backlog_norm(), 0.5)
            backlog_norm, dropped_norm, reward = env._queue_penalty_terms(
                queue_bits_after_tx=80.0,
                dropped_bits=40.0,
            )
            self.assertAlmostEqual(backlog_norm, 0.5)
            self.assertAlmostEqual(dropped_norm, 0.25)
            self.assertAlmostEqual(reward, -config.reward.gamma_queue * 0.75)

    def test_queue_overflow_reports_remaining_dropped_bits(self) -> None:
        for environment_type in (NoQuantEnvironment, QuantEnvironment):
            env = environment_type.__new__(environment_type)
            env.queue_capacity_bits = 100.0
            env.uav_data_queue = [
                SimpleNamespace(size_bits=100.0, transmitted_bits=60.0, created_step=0),
                SimpleNamespace(size_bits=80.0, transmitted_bits=0.0, created_step=1),
            ]

            dropped_packets, dropped_bits = env._enforce_queue_capacity()
            self.assertEqual(dropped_packets, 1)
            self.assertAlmostEqual(dropped_bits, 40.0)
            self.assertEqual(len(env.uav_data_queue), 1)

    def test_data_accounting_checks_full_conservation_not_only_ordering(self) -> None:
        self.assertEqual(
            data_accounting_violation_bits(
                produced_bits=100.0,
                link_transmitted_bits=60.0,
                completed_packet_bits=50.0,
                delivered_bits=40.0,
                dropped_bits=10.0,
                final_queue_bits=30.0,
            ),
            0.0,
        )
        self.assertEqual(
            data_accounting_violation_bits(
                produced_bits=100.0,
                link_transmitted_bits=60.0,
                completed_packet_bits=50.0,
                delivered_bits=40.0,
                dropped_bits=0.0,
                final_queue_bits=30.0,
            ),
            10.0,
        )

    def test_data_accounting_ignores_only_gbit_scale_machine_roundoff(self) -> None:
        self.assertEqual(
            data_accounting_violation_bits(
                produced_bits=5.0e9,
                link_transmitted_bits=3.0e9 + 2.0e-6,
                completed_packet_bits=2.9e9,
                delivered_bits=2.8e9,
                dropped_bits=1.0e9,
                final_queue_bits=1.0e9,
            ),
            0.0,
        )
        self.assertTrue(
            np.isinf(
                data_accounting_violation_bits(
                    produced_bits=float("nan"),
                    link_transmitted_bits=0.0,
                    completed_packet_bits=0.0,
                    delivered_bits=0.0,
                    dropped_bits=0.0,
                    final_queue_bits=0.0,
                )
            )
        )

    def test_shared_strict_pareto_gate_and_dominance(self) -> None:
        self.assertTrue(
            is_strict_pareto_feasible(
                nmse=0.1,
                link_bits=1.0e9,
                energy_failure_rate=0.0,
                full_horizon_rate=1.0,
                max_data_accounting_violation_bits=1.0e-6,
            )
        )
        self.assertFalse(
            is_strict_pareto_feasible(
                nmse=0.1,
                link_bits=1.0e9,
                energy_failure_rate=0.0,
                full_horizon_rate=0.99,
                max_data_accounting_violation_bits=0.0,
            )
        )
        self.assertTrue(dominates_nmse_link(0.09, 1.0e9, 0.10, 1.0e9))
        self.assertFalse(dominates_nmse_link(0.09, 1.1e9, 0.10, 1.0e9))

    def test_comm_recovery_uses_visible_bit_backlog_not_hidden_packet_count(self) -> None:
        for environment_type, config in (
            (NoQuantEnvironment, NoQuantConfig()),
            (QuantEnvironment, QuantConfig()),
        ):
            env = environment_type.__new__(environment_type)
            env.config = config
            env.queue_capacity_bits = 100.0
            env.uav_pos = np.array([1.0, 1.0])
            env.ugv_channel_info = SimpleNamespace(outage=False)
            env._get_motion_target_grid = lambda: (9, 9)
            references = []

            def select_target(reference):
                references.append(tuple(np.asarray(reference, dtype=float)))
                return (0, 0), True, True

            env._select_ugv_path_corridor_target = select_target

            env.uav_data_queue = [
                SimpleNamespace(size_bits=100.0, transmitted_bits=40.0)
            ]
            _, source, _, _ = env._get_ugv_guidance_target_grid()
            self.assertEqual(source, "communication_queue")
            self.assertEqual(references[-1], (1.0, 1.0))

            env.uav_data_queue = [
                SimpleNamespace(size_bits=100.0, transmitted_bits=60.0)
            ]
            _, source, _, _ = env._get_ugv_guidance_target_grid()
            self.assertEqual(source, "communication_predictive")
            self.assertEqual(references[-1], (9.0, 9.0))

    def test_shared_payload_queue_and_progress_defaults(self) -> None:
        for config in (NoQuantConfig(), QuantConfig()):
            self.assertEqual(config.comm.source_measurement_bits, 32)
            self.assertEqual(config.comm.data_per_sample, 8e6)
            self.assertEqual(config.uav.queue_capacity_bits, 512e6)
            self.assertEqual(config.reward.lambda_uav_progress, 2.0)
            self.assertEqual(config.reward.lambda_uav_backtrack, 2.0)
            self.assertEqual(config.reward.lambda_ugv_progress, 2.0)
            self.assertEqual(config.reward.lambda_ugv_backtrack, 2.0)

    def test_quant_payload_scales_from_thirty_two_bit_source(self) -> None:
        config = QuantConfig()
        env = QuantEnvironment.__new__(QuantEnvironment)
        env.config = config
        env.source_measurement_bits = config.comm.source_measurement_bits
        env.sensing_band_num = 1

        expected_mbit = {10: 2.5, 8: 2.0, 6: 1.5}
        for bits, expected in expected_mbit.items():
            env.current_quant_bits = bits
            self.assertAlmostEqual(
                env._current_sample_packet_bits() / 1e6,
                expected,
            )

    def test_local_target_radius_matches_four_uav_actions(self) -> None:
        for config in (NoQuantConfig(), QuantConfig()):
            self.assertEqual(config.uav.step_size, 4.0)
            self.assertEqual(config.ugv.step_size, 5.0)
            self.assertEqual(config.planner.local_planner_radius, 15)
            self.assertEqual(
                int(np.ceil(config.planner.local_planner_radius / config.uav.step_size)),
                4,
            )

    def test_episode_horizon_is_the_only_rollout_length_setting(self) -> None:
        for config in (NoQuantConfig(), QuantConfig()):
            self.assertEqual(config.training.episode_max_steps, 200)
            self.assertFalse(hasattr(config.training, "rollout_length"))

    def test_physical_link_bits_are_not_a_direct_reward_term(self) -> None:
        for environment_type, config in (
            (NoQuantEnvironment, NoQuantConfig()),
            (QuantEnvironment, QuantConfig()),
        ):
            self.assertFalse(hasattr(config.reward, "beta_tx"))
            self.assertFalse(hasattr(environment_type, "_physical_link_cost_reward"))

    def test_ippo_uses_local_critics_with_shared_environment_interface(self) -> None:
        config = QuantConfig()
        config.training.device = "cpu"
        config.training.actor_hidden_dims = [8]
        config.training.critic_hidden_dims = [8]
        policy = QuantIPPOPolicy(
            {"uav_obs": 4, "ugv_obs": 3, "critic_state": 9},
            {"uav_action": 5, "ugv_action": 2},
            config.training,
        )
        action = policy.get_single_action(
            uav_obs=np.zeros(4, dtype=np.float32),
            ugv_obs=np.zeros(3, dtype=np.float32),
            critic_state=np.zeros(9, dtype=np.float32),
            uav_action_mask=np.array([1, 0, 0, 0, 0], dtype=bool),
            ugv_action_mask=np.array([0, 1], dtype=bool),
            deterministic=True,
        )
        self.assertEqual(action["uav_action"], 0)
        self.assertEqual(action["ugv_action"], 1)

    def test_random_baseline_samples_only_shared_environment_masks(self) -> None:
        env = SimpleNamespace(uav_action_size=4, ugv_action_size=3)
        policy = RandomValidPolicy(seed=7)
        policy.on_episode_reset(reset_seed=123, env=env)
        first = policy.get_single_action(
            uav_action_mask=np.array([0, 1, 0, 1], dtype=bool),
            ugv_action_mask=np.array([1, 0, 0], dtype=bool),
        )
        policy.on_episode_reset(reset_seed=123, env=env)
        repeated = policy.get_single_action(
            uav_action_mask=np.array([0, 1, 0, 1], dtype=bool),
            ugv_action_mask=np.array([1, 0, 0], dtype=bool),
        )
        self.assertIn(first["uav_action"], (1, 3))
        self.assertEqual(first["ugv_action"], 0)
        self.assertEqual(first, repeated)

    @staticmethod
    def _energy_stub(environment_type, config, *, quantized: bool):
        env = environment_type.__new__(environment_type)
        env.config = config
        env.uav_pos = np.array([0.0, 0.0])
        env.uav_energy = 1000.0
        env.uav_step_count = 1
        env.uav_direction_choices = 1
        env.uav_direction_ids = np.array([0], dtype=int)
        env.num_bw_choices = len(config.uav.bandwidth_ratios)
        env.bandwidth_ratios = np.asarray(config.uav.bandwidth_ratios, dtype=float)
        config.comm.tx_power_choices_dbm = [30.0]
        config.comm.tx_power_dbm = 30.0
        env._init_transmit_power()
        env.uav_action_size = env.num_bw_choices
        if quantized:
            env.num_quant_choices = len(config.uav.quant_bits)
            env.uav_action_size *= env.num_quant_choices
        env.current_step = 0
        env.scene = _OpenScene()
        env._uav_action_mask_cache = {}
        env._mask_cache_max_entries = 16
        env._uav_action_mask_cache_key = lambda: (0, 0, -1, -1)
        env._can_follow_direction = lambda **kwargs: True
        env._rollout_direction = lambda **kwargs: (
            np.asarray(kwargs["position"], dtype=float).copy(),
            0,
        )
        return env

    def test_quant_and_noquant_use_identical_band_scaled_sensing_energy(self) -> None:
        noquant_config = NoQuantConfig()
        quant_config = QuantConfig()
        ratio = noquant_config.uav.bandwidth_ratios[0]
        units = noquant_config.uav.sensing_units_for_ratio(ratio)
        expected_power = noquant_config.uav.sensing_power_for_units(units)

        noquant_env = self._energy_stub(
            NoQuantEnvironment, noquant_config, quantized=False
        )
        quant_env = self._energy_stub(QuantEnvironment, quant_config, quantized=True)
        noquant_env.sensing_band_num = units
        quant_env.sensing_band_num = units

        noquant_energy, noquant_moved_steps = noquant_env._move_uav(direction_idx=0)
        quant_energy, moved_steps = quant_env._move_uav(direction_idx=0)

        self.assertEqual(noquant_moved_steps, 0)
        self.assertEqual(moved_steps, 0)
        self.assertAlmostEqual(noquant_env.last_uav_sensing_power, expected_power)
        self.assertAlmostEqual(quant_env.last_uav_sensing_power, expected_power)
        self.assertAlmostEqual(quant_energy, noquant_energy)

    def test_quant_energy_mask_repeats_noquant_band_mask_over_quant_bits(self) -> None:
        noquant_config = NoQuantConfig()
        quant_config = QuantConfig()
        noquant_env = self._energy_stub(
            NoQuantEnvironment, noquant_config, quantized=False,
        )
        quant_env = self._energy_stub(QuantEnvironment, quant_config, quantized=True)

        step_costs = [
            (
                noquant_config.uav.hover_power + 1.0
                + noquant_config.uav.sensing_power_for_units(
                    noquant_config.uav.sensing_units_for_ratio(ratio)
                )
            ) * noquant_config.uav.step_duration
            for ratio in noquant_config.uav.bandwidth_ratios
        ]
        min_cost, max_cost = min(step_costs), max(step_costs)
        # Exercise both a partially affordable mask and an all-affordable mask.
        # Reusing the environments also checks that cached geometry does not
        # freeze the energy-dependent part of the mask.
        for energy, all_affordable in (
            ((min_cost + max_cost) / 2.0, False),
            (max_cost, True),
        ):
            with self.subTest(energy=energy):
                noquant_env.uav_energy = energy
                quant_env.uav_energy = energy
                noquant_mask = noquant_env._build_uav_action_mask()
                quant_mask = quant_env._build_uav_action_mask().reshape(
                    quant_env.num_bw_choices, quant_env.num_quant_choices,
                )
                np.testing.assert_array_equal(
                    quant_mask,
                    np.repeat(noquant_mask[:, None], quant_env.num_quant_choices, axis=1),
                )
                self.assertTrue(bool(noquant_mask[int(np.argmin(step_costs))]))
                self.assertEqual(
                    bool(noquant_mask[int(np.argmax(step_costs))]), all_affordable,
                )
                if all_affordable:
                    self.assertTrue(bool(np.all(noquant_mask)))

    def test_energy_mask_checks_current_cost_without_future_reserve(self) -> None:
        for environment_type, config, quantized in (
            (NoQuantEnvironment, NoQuantConfig(), False),
            (QuantEnvironment, QuantConfig(), True),
        ):
            env = self._energy_stub(environment_type, config, quantized=quantized)
            step_costs = [
                (
                    config.uav.hover_power + 1.0
                    + config.uav.sensing_power_for_units(
                        config.uav.sensing_units_for_ratio(ratio)
                    )
                ) * config.uav.step_duration
                for ratio in config.uav.bandwidth_ratios
            ]
            selected_cost = step_costs[1]
            self.assertGreater(selected_cost, min(step_costs))
            for horizon in (1, 4, 200):
                config.training.episode_max_steps = horizon
                for step in sorted({0, horizon - 1}):
                    env.current_step = step
                    for energy, feasible in (
                        (selected_cost + 1e-6, True),
                        (selected_cost, True),
                        (selected_cost - 1e-6, False),
                    ):
                        with self.subTest(
                            variant=environment_type.__module__,
                            horizon=horizon, step=step, energy=energy,
                        ):
                            env.uav_energy = energy
                            mask = env._build_uav_action_mask()
                            if quantized:
                                mask = mask.reshape(env.num_bw_choices, env.num_quant_choices)
                                np.testing.assert_array_equal(
                                    mask[1], np.full(env.num_quant_choices, feasible),
                                )
                            else:
                                self.assertEqual(bool(mask[1]), feasible)

            # Positive energy below every action cost must still be rejected.
            env.uav_energy = min(step_costs) - 1e-6
            with self.assertRaisesRegex(RuntimeError, "no feasible action"):
                env._build_uav_action_mask()

    def test_run_analysis_labels_smoke_and_keeps_low_completion_diagnostic(self) -> None:
        metrics = {
            "training": {
                "eval_update": [1],
                "eval_mean_nmse": [0.1],
                "eval_mean_link_transmitted_bits": [0.5e9],
                "eval_mean_data_produced_bits": [1.0e9],
                "eval_mean_completed_packet_bits": [0.5e9],
                "eval_energy_failure_rate": [0.0],
                "eval_full_horizon_rate": [1.0],
                "eval_max_data_accounting_violation_bits": [0.0],
                "eval_mean_outage_ratio": [0.5],
            },
            "config": {"mappo": {"total_timesteps": 20}},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            summary = summarize_run(metrics_path)

        self.assertEqual(summary["evidence_level"], "engineering_smoke_only")
        self.assertTrue(summary["points"][0]["feasible"])
        self.assertEqual(len(summary["frontier"]), 1)

    def test_run_analysis_uses_completed_not_planned_transitions(self) -> None:
        metrics = {
            "training": {
                "eval_update": [5],
                "global_step": [8_000],
                "eval_mean_nmse": [0.1],
                "eval_mean_link_transmitted_bits": [1.0e9],
                "eval_mean_data_produced_bits": [1.0e9],
                "eval_mean_completed_packet_bits": [1.0e9],
                "eval_mean_service_completion_ratio": [1.0],
                "eval_energy_failure_rate": [0.0],
                "eval_full_horizon_rate": [1.0],
                "eval_max_data_accounting_violation_bits": [0.0],
                "eval_mean_outage_ratio": [0.0],
            },
            "config": {"mappo": {"total_timesteps": 176_000}},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            summary = summarize_run(metrics_path)

        self.assertEqual(summary["planned_timesteps"], 176_000)
        self.assertEqual(summary["completed_timesteps"], 8_000)
        self.assertEqual(summary["evidence_level"], "engineering_smoke_only")

    def test_run_analysis_prefers_logged_mean_service_completion(self) -> None:
        metrics = {
            "training": {
                "eval_update": [1],
                "eval_mean_nmse": [0.1],
                "eval_mean_link_transmitted_bits": [1.0e9],
                "eval_mean_data_produced_bits": [1.0e9],
                "eval_mean_completed_packet_bits": [1.0e9],
                "eval_mean_service_completion_ratio": [0.5],
                "eval_energy_failure_rate": [0.0],
                "eval_full_horizon_rate": [1.0],
                "eval_max_data_accounting_violation_bits": [0.0],
                "eval_mean_outage_ratio": [0.5],
            },
            "config": {"mappo": {"total_timesteps": 32_000}},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_path = Path(temp_dir) / "metrics.json"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            summary = summarize_run(metrics_path)

        point = summary["points"][0]
        self.assertEqual(point["service_completion_ratio"], 0.5)
        self.assertEqual(
            point["service_completion_source"],
            "logged_mean_episode_ratio",
        )
        self.assertTrue(point["feasible"])

    def test_long_replay_summary_enforces_frozen_suite_and_prefill(self) -> None:
        aggregate = {
            "eval_mean_nmse": 0.1,
            "eval_mean_link_transmitted_bits": 1.0e9,
            "eval_mean_service_completion_ratio": 0.96,
            "eval_min_scene_service_completion_ratio": 0.96,
            "eval_mean_outage_ratio": 0.1,
            "eval_energy_failure_rate": 0.0,
            "eval_full_horizon_rate": 1.0,
            "eval_max_data_accounting_violation_bits": 0.0,
            "eval_mean_prefill_observed_band_units": 80.0,
            "eval_scene_count": 8,
            "eval_num_total_episodes": 24,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "o1"
            run_dir.mkdir()
            replay_path = run_dir / "replay_final_seed200042_n3.json"
            replay_path.write_text(
                json.dumps(
                    {
                        "evaluation_config": {
                            "checkpoint": "final_model.pt",
                            "device": "cuda:0",
                            "iibtd_device": "cuda:0",
                            "num_episodes_per_scene": 3,
                            "max_steps": 200,
                            "uav_max_energy": 9000.0,
                            "snr_outage_threshold_db": -5.0,
                            "prefill_percent": 5.0,
                            "prefill_budget_basis": 200,
                            "ugv_comm_target_mode": "path_corridor",
                            "ugv_comm_backlog_threshold": 0.5,
                            "ugv_comm_local_path_horizon": 10,
                            "ugv_comm_expanded_path_horizon": 20,
                            "ugv_comm_corridor_width": 1,
                            "ugv_service_action_mask": True,
                        },
                        "aggregate": aggregate,
                    }
                ),
                encoding="utf-8",
            )
            summary = summarize_replays(
                [("o1", run_dir)],
                expected_prefill_band_units=80.0,
                expected_scene_count=8,
                expected_episodes_per_scene=3,
            )
            self.assertTrue(summary["points"][0]["feasible"])

            aggregate["eval_min_scene_service_completion_ratio"] = 0.90
            replay_path.write_text(
                json.dumps(
                    {
                        "evaluation_config": {
                            "checkpoint": "final_model.pt",
                            "device": "cuda:0",
                            "iibtd_device": "cuda:0",
                            "num_episodes_per_scene": 3,
                            "max_steps": 200,
                            "uav_max_energy": 9000.0,
                            "snr_outage_threshold_db": -5.0,
                            "prefill_percent": 5.0,
                            "prefill_budget_basis": 200,
                            "ugv_comm_target_mode": "path_corridor",
                            "ugv_comm_backlog_threshold": 0.5,
                            "ugv_comm_local_path_horizon": 10,
                            "ugv_comm_expanded_path_horizon": 20,
                            "ugv_comm_corridor_width": 1,
                            "ugv_service_action_mask": True,
                        },
                        "aggregate": aggregate,
                    }
                ),
                encoding="utf-8",
            )
            summary = summarize_replays(
                [("o1", run_dir)],
                expected_prefill_band_units=80.0,
                expected_scene_count=8,
                expected_episodes_per_scene=3,
            )
            self.assertTrue(summary["points"][0]["feasible"])

            aggregate["eval_min_scene_service_completion_ratio"] = 0.96
            aggregate["eval_mean_prefill_observed_band_units"] = 79.0
            replay_path.write_text(
                json.dumps(
                    {
                        "evaluation_config": {
                            "checkpoint": "final_model.pt",
                            "device": "cuda:0",
                            "iibtd_device": "cuda:0",
                            "num_episodes_per_scene": 3,
                            "max_steps": 200,
                            "uav_max_energy": 9000.0,
                            "snr_outage_threshold_db": -5.0,
                            "prefill_percent": 5.0,
                            "prefill_budget_basis": 200,
                            "ugv_comm_target_mode": "path_corridor",
                            "ugv_comm_backlog_threshold": 0.5,
                            "ugv_comm_local_path_horizon": 10,
                            "ugv_comm_expanded_path_horizon": 20,
                            "ugv_comm_corridor_width": 1,
                            "ugv_service_action_mask": True,
                        },
                        "aggregate": aggregate,
                    }
                ),
                encoding="utf-8",
            )
            summary = summarize_replays(
                [("o1", run_dir)],
                expected_prefill_band_units=80.0,
                expected_scene_count=8,
                expected_episodes_per_scene=3,
            )
            self.assertFalse(summary["points"][0]["feasible"])


if __name__ == "__main__":
    unittest.main()
