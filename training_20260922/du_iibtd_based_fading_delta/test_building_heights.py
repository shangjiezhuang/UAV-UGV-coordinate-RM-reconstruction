"""Building geometry is fixed per scene and shared by all experimental methods."""

from dataclasses import asdict
from importlib import import_module
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.ndimage import label

from du_iibtd_based_fading_delta.building_heights import build_building_height_map, validate_building_height_config, partition_building_footprints
from du_iibtd_based_fading_delta.evaluation_common import apply_saved_config


class BuildingHeightTests(unittest.TestCase):
    def config(self, variant='quant'):
        return import_module(f'du_iibtd_based_fading_delta.shared.{variant}.config').Config()

    def test_connected_footprint_has_one_height_and_background_is_zero(self):
        config = self.config()
        mask = np.zeros((12, 12), dtype=bool)
        mask[1:3, 1:4] = True
        mask[4:7, 4:6] = True
        mask[9, 9] = mask[10, 10] = True
        heights = build_building_height_map(mask, config.scene)
        self.assertTrue(np.all(heights[~mask] == 0))
        self.assertTrue(np.all((heights[mask] >= 47) & (heights[mask] <= 53)))
        self.assertTrue(set(np.unique(heights[mask])).issubset(set(range(47, 54))))
        self.assertEqual(len(np.unique(heights[1:3, 1:4])), 1)
        self.assertEqual(len(np.unique(heights[4:7, 4:6])), 1)

    def test_reproducible_across_variants_and_channel_rng_is_untouched(self):
        mask = np.zeros((7, 7), dtype=bool)
        mask[::2, ::2] = True
        outputs = []
        for variant in ('quant', 'noquant'):
            module = import_module(f'du_iibtd_based_fading_delta.shared.{variant}.sim_models')
            sim = module.SimDataGen.__new__(module.SimDataGen)
            sim.config = self.config(variant)
            sim.config.training.seed = 901 if variant == 'quant' else 27
            sim.rng = np.random.RandomState(7)
            sim.config.scene.grid_size = mask.shape
            initial_state = sim.rng.get_state()
            height = sim._build_building_height_map(mask)
            np.testing.assert_array_equal(initial_state[1], sim.rng.get_state()[1])
            self.assertEqual(initial_state[2:], sim.rng.get_state()[2:])
            sim.reset_rng(123456)
            np.testing.assert_array_equal(height, sim._build_building_height_map(mask))
            scene = module.GridScene(sim.config, occupancy_grid=mask)
            np.testing.assert_array_equal(scene.building_heights, height)
            outputs.append(height)
        np.testing.assert_array_equal(outputs[0], outputs[1])

    def test_different_geometry_seed_changes_heights(self):
        config = self.config()
        mask = np.eye(8, dtype=bool)
        first = build_building_height_map(mask, config.scene)
        config.scene.building_height_seed += 1
        second = build_building_height_map(mask, config.scene)
        self.assertFalse(np.array_equal(first, second))

    def test_fixed_height_mode_and_empty_map(self):
        mask = np.array([[True, False], [False, True]])
        fixed = SimpleNamespace(building_height_m=25.)
        np.testing.assert_array_equal(build_building_height_map(mask, fixed), mask * 25.)
        config = self.config()
        np.testing.assert_array_equal(build_building_height_map(np.zeros((3, 4), bool), config.scene), np.zeros((3, 4)))
        config.scene.building_height_min_m = config.scene.building_height_max_m = 27.
        np.testing.assert_array_equal(build_building_height_map(mask, config.scene), mask * 27.)

    def test_height_change_affects_actual_los_geometry(self):
        for variant in ('quant', 'noquant'):
            config = self.config(variant)
            config.scene.grid_size = (16, 3)
            mask = np.zeros(config.scene.grid_size, dtype=bool)
            mask[1, 1] = True
            scene_type = import_module(f'du_iibtd_based_fading_delta.shared.{variant}.sim_models').GridScene
            outcomes = []
            for height in (45., 48.):
                config.scene.building_height_min_m = config.scene.building_height_max_m = height
                scene = scene_type(config, occupancy_grid=mask)
                outcomes.append(scene.has_line_of_sight(np.array([0., 1.]), np.array([15., 1.])))
            self.assertEqual(outcomes, [True, False])

    def test_saved_legacy_config_keeps_fixed_height(self):
        config = self.config()
        raw = asdict(config)
        raw['scene']['building_height_m'] = 25.
        raw['scene']['uav_height'] = 30.
        for field in ('building_height_mode', 'building_height_min_m', 'building_height_max_m', 'building_height_seed'):
            del raw['scene'][field]
        with self.assertWarnsRegex(UserWarning, 'Legacy building-height'):
            apply_saved_config(config, raw)
        self.assertEqual(config.scene.building_height_mode, 'fixed')
        np.testing.assert_array_equal(build_building_height_map(np.ones((3, 3)), config.scene), np.full((3, 3), 25.))
        apply_saved_config(config, asdict(self.config()))
        self.assertEqual(config.scene.building_height_mode, 'uniform_integer_per_building')

    def test_integer_levels_include_both_endpoints_and_diagonal_buildings_are_independent(self):
        config = self.config()
        mask = np.eye(80, dtype=bool)
        heights = build_building_height_map(mask, config.scene)
        self.assertEqual(set(heights[mask]), set(range(47, 54)))
        self.assertTrue(np.all(heights[mask] == np.rint(heights[mask])))

    def test_legacy_continuous_mode_is_preserved(self):
        config = self.config()
        config.scene.building_height_mode = 'uniform_per_building'
        config.scene.building_height_min_m = 25.
        config.scene.building_height_max_m = 29.
        mask = np.eye(8, dtype=bool)
        heights = build_building_height_map(mask, config.scene)
        self.assertTrue(np.any(heights[mask] != np.rint(heights[mask])))
        self.assertTrue(np.all((heights[mask] >= 25) & (heights[mask] <= 29)))

    def test_invalid_configuration_rejected(self):
        for field, value in [('building_height_mode', 'pixel_noise'), ('building_height_min_m', -1),
                             ('building_height_min_m', 54), ('building_height_min_m', 45.5), ('building_height_max_m', float('nan')),
                             ('building_height_seed', 1.5), ('building_height_seed', -1), ('building_height_seed', True),
                             ('building_split_min_area_m2', -1), ('building_split_min_area_m2', float('nan'))]:
            config = self.config()
            setattr(config.scene, field, value)
            with self.assertRaises(ValueError):
                validate_building_height_config(config.scene)

    def test_large_block_split_once_at_physical_area_threshold(self):
        config = self.config()
        config.scene.grid_spacing = 2  # Explicit scale for these hand-calculated areas.
        mask = np.zeros((70, 50), dtype=bool)
        mask[1:21, 1:21] = True  # 1600 m2: split into equal connected halves.
        mask[24:34, 1:21] = True  # 800 m2: unchanged.
        mask[40:70, 1:41] = True  # 4800 m2: also split only once.
        ids, count = partition_building_footprints(mask, config.scene)
        self.assertEqual(count, 5)
        np.testing.assert_array_equal(ids > 0, mask)
        self.assertEqual(len(np.unique(ids[1:21, 1:21])), 2)
        self.assertEqual(len(np.unique(ids[24:34, 1:21])), 1)
        self.assertEqual(len(np.unique(ids[40:70, 1:41])), 2)
        self.assertEqual(sorted(np.unique(ids[1:21, 1:21], return_counts=True)[1]), [200, 200])
        for block_id in range(1, count + 1):
            self.assertEqual(label(ids == block_id)[1], 1)
        config.scene.grid_spacing = 3
        self.assertEqual(partition_building_footprints(mask, config.scene)[1], 6)
        config.scene.grid_spacing = 1
        self.assertEqual(partition_building_footprints(mask, config.scene)[1], 4)

    def test_irregular_connected_outline_remains_connected_after_split(self):
        config = self.config()
        mask = np.zeros((45, 55), dtype=bool)
        mask[1:31, 1:11] = True
        mask[20:31, 10:41] = True
        mask[2:21, 30:41] = True
        ids, count = partition_building_footprints(mask, config.scene)
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(ids > 0, mask)
        for block_id in (1, 2):
            self.assertEqual(label(ids == block_id)[1], 1)
            self.assertGreater(np.count_nonzero(ids == block_id), mask.sum() // 4)

    def test_partition_and_heights_shared_across_variants(self):
        mask = np.zeros((50, 55), dtype=bool)
        mask[1:31, 1:31] = True
        mask[40:45, 40:45] = True
        outputs = []
        for variant in ('quant', 'noquant'):
            config = self.config(variant)
            ids, count = partition_building_footprints(mask, config.scene)
            heights = build_building_height_map(mask, config.scene)
            self.assertEqual(count, 3)
            for block_id in range(1, count + 1):
                self.assertEqual(len(np.unique(heights[ids == block_id])), 1)
            outputs.append((ids, heights))
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        np.testing.assert_array_equal(outputs[0][1], outputs[1][1])

    def test_old_saved_config_disables_new_partition_rule(self):
        config = self.config()
        raw = asdict(config)
        del raw['scene']['building_split_min_area_m2']
        apply_saved_config(config, raw)
        self.assertEqual(config.scene.building_split_min_area_m2, 0)
        self.assertEqual(partition_building_footprints(np.ones((50, 50)), config.scene)[1], 1)
        apply_saved_config(config, asdict(self.config()))
        self.assertEqual(config.scene.building_split_min_area_m2, 1200)


if __name__ == '__main__':
    unittest.main()
