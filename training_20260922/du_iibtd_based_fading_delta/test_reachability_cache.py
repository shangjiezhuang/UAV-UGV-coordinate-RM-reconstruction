"""Compare the optimized geometry to the prior movement-based implementation."""
import importlib
import unittest
from collections import deque
from types import SimpleNamespace
import numpy as np
from du_iibtd_based_fading_delta.uav_navigation import HeightAwareUAVNavigation

class Scene(HeightAwareUAVNavigation):
    def __init__(self, obstacles):
        self.Nx,self.Ny=obstacles.shape;self.occupancy=obstacles.copy()
        self.building_heights=np.where(obstacles,52.,45.);self.uav_height=50
        self._init_uav_navigation()
    def _grid_index(self,p):return tuple(np.rint(p).astype(int))
    def _is_within_bounds(self,p):return 0<=p[0]<self.Nx and 0<=p[1]<self.Ny

def reference_mask(env,horizon):
    start=np.asarray(env.uav_pos,dtype=float);start_cell=env._grid_cell(start)
    reached=np.zeros((env.Nx,env.Ny),bool);reached[start_cell]=True
    queue=deque([(start,0)]);seen={start_cell:0}
    while queue:
        pos,depth=queue.popleft()
        if depth>=horizon:continue
        for direction in env.uav_direction_ids:
            nxt,moved=env._rollout_direction(pos,direction,env.uav_step_count,env.scene.is_uav_position_valid,False)
            cell=env._grid_cell(nxt)
            if moved<=0 or seen.get(cell,float('inf'))<=depth+1:continue
            seen[cell]=depth+1;reached[cell]=True;queue.append((nxt,depth+1))
    sampling=reached & env.sampling_valid_mask
    if env.target_arrival_radius<=0:return sampling
    out=np.zeros_like(sampling)
    for x,y in np.argwhere(sampling):
        out |= (env.grid_x_coords-x)**2+(env.grid_y_coords-y)**2<=env.target_arrival_radius**2+1e-9
    return out

class ReachabilityCacheTests(unittest.TestCase):
    def test_exact_masks_across_walls_macro_steps_radii_and_modes(self):
        rng=np.random.default_rng(317)
        cases=0
        for variant in ['quant','noquant']:
            em=importlib.import_module(f'du_iibtd_based_fading_delta.shared.{variant}.environment')
            for density in [0.,.15,.5]:
                obstacles=rng.random((11,13))<density;obstacles[0,0]=False
                scene=Scene(obstacles)
                env=em.UAVUGVEnvironment.__new__(em.UAVUGVEnvironment)
                env.Nx,env.Ny=obstacles.shape;env.scene=scene
                env.uav_direction_ids=[0,1,2,3,4]
                env.grid_x_coords,env.grid_y_coords=np.indices(obstacles.shape)
                env.sampling_valid_mask=(rng.random(obstacles.shape)>.15)&~obstacles
                for start in [(0,0),tuple(np.argwhere(~obstacles)[-1])]:
                    env.uav_pos=np.array(start,dtype=float)
                    for step in [1,3,5]:
                        env.uav_step_count=step
                        for horizon in [1,4,20]:
                            env._planner_reachability_horizon=lambda _:horizon
                            for radius in [0.,1.,2.5]:
                                env.target_arrival_radius=radius
                                expected=reference_mask(env,horizon)
                                actual=env._build_reachable_target_mask('local')
                                np.testing.assert_array_equal(actual,expected)
                                # Cached call must also agree.
                                np.testing.assert_array_equal(env._build_reachable_target_mask('global'),expected)
                                cases+=1
                scene.occupancy[:]=True;scene._init_uav_navigation()
                self.assertFalse(scene._uav_macro_reachable_cache)
                self.assertFalse(scene._uav_macro_transition_cache)
        self.assertEqual(cases,324)

if __name__=='__main__':unittest.main()
