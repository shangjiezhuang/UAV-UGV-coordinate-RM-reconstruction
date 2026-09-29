"""Fixed-altitude UAV collision geometry and obstacle-aware route forecasts."""

from collections import deque
import heapq
import itertools

import numpy as np
from scipy.ndimage import label


class HeightAwareUAVNavigation:
    """Mixin for GridScene; ground-vehicle occupancy remains independent."""

    def _init_uav_navigation(self):
        # Equality is a collision with the roof, not a flyable clearance.
        self.uav_obstacles = self.occupancy & (
            self.building_heights >= self.uav_height - 1e-6
        )
        self.has_uav_obstacles = bool(np.any(self.uav_obstacles))
        self._uav_component_labels, _ = label(
            ~self.uav_obstacles,
            structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
        )
        self._uav_distance_cache = {}
        self._uav_route_cache = {}
        self._uav_macro_transition_cache = {}
        self._uav_macro_reachable_cache = {}

    def _uav_macro_transitions(self, macro_step, offsets):
        """Executable action endpoints on this fixed-height, fixed-building scene.

        Every crossed cell is checked; a blocked action stays at its last valid
        cell. This is the same movement primitive as stop_at_target=False.
        """
        key = (int(macro_step), tuple(offsets))
        cached = self._uav_macro_transition_cache.get(key)
        if cached is not None:
            return cached
        x, y = np.indices((self.Nx, self.Ny))
        table = np.empty((self.Nx*self.Ny, len(offsets)), dtype=np.int32)
        for col, (dx, dy) in enumerate(offsets):
            ex, ey = x.copy(), y.copy()
            active = np.ones_like(self.uav_obstacles, dtype=bool)
            for _ in range(max(0, int(macro_step))):
                px, py = ex+dx, ey+dy
                valid = (px >= 0) & (px < self.Nx) & (py >= 0) & (py < self.Ny)
                valid &= ~self.uav_obstacles[np.clip(px, 0, self.Nx-1), np.clip(py, 0, self.Ny-1)]
                active &= valid
                ex[active], ey[active] = px[active], py[active]
            table[:, col] = (ex*self.Ny+ey).ravel()
        table.setflags(write=False)
        # Runtime uses one step size; bound storage for alternate configurations.
        if len(self._uav_macro_transition_cache) >= 8:
            self._uav_macro_transition_cache.clear()
        self._uav_macro_transition_cache[key] = table
        return table

    def uav_macro_reachable_mask(self, start, *, macro_step, action_horizon, offsets):
        """Reachable action endpoints; the caller applies its sampling mask/radius."""
        start = self._grid_index(start)
        offsets = tuple((int(dx), int(dy)) for dx, dy in offsets)
        key = (start, int(macro_step), int(action_horizon), offsets)
        cached = self._uav_macro_reachable_cache.get(key)
        if cached is not None:
            return cached
        transitions = self._uav_macro_transitions(macro_step, offsets)
        first = start[0]*self.Ny+start[1]
        reached = np.zeros(self.Nx*self.Ny, dtype=bool)
        reached[first] = True
        frontier = deque([(first, 0)])
        while frontier:
            cell, depth = frontier.popleft()
            if depth >= action_horizon:
                continue
            for endpoint in transitions[cell]:
                endpoint = int(endpoint)
                if not reached[endpoint]:
                    reached[endpoint] = True
                    frontier.append((endpoint, depth+1))
        result = reached.reshape(self.Nx, self.Ny)
        result.setflags(write=False)
        if len(self._uav_macro_reachable_cache) >= 128:
            self._uav_macro_reachable_cache.clear()
        self._uav_macro_reachable_cache[key] = result
        return result

    def is_uav_position_valid(self, grid_position):
        if not self._is_within_bounds(grid_position):
            return False
        return not bool(self.uav_obstacles[self._grid_index(grid_position)])

    def _uav_distance_field(self, target):
        goal = self._grid_index(target)
        cached = self._uav_distance_cache.get(goal)
        if cached is not None:
            return cached
        distances = np.full((self.Nx, self.Ny), np.inf)
        if self.is_uav_position_valid(goal):
            distances[goal] = 0.0
            frontier = deque([goal])
            while frontier:
                x, y = frontier.popleft()
                for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                    neighbor = (x + dx, y + dy)
                    nx, ny = neighbor
                    if not (0 <= nx < self.Nx and 0 <= ny < self.Ny):
                        continue
                    if self.uav_obstacles[neighbor] or np.isfinite(distances[neighbor]):
                        continue
                    distances[neighbor] = distances[x, y] + 1.0
                    frontier.append(neighbor)
        if len(self._uav_distance_cache) >= 64:
            self._uav_distance_cache.clear()
        self._uav_distance_cache[goal] = distances
        return distances

    def uav_shortest_path_distance(self, start, goal):
        if not self.is_uav_position_valid(start) or not self.is_uav_position_valid(goal):
            return float("inf")
        if not self.has_uav_obstacles:
            return float(np.abs(np.asarray(start) - np.asarray(goal)).sum())
        return float(self._uav_distance_field(goal)[self._grid_index(start)])

    def uav_reachable_mask(self, start):
        if not self.is_uav_position_valid(start):
            return np.zeros((self.Nx, self.Ny), dtype=bool)
        if not self.has_uav_obstacles:
            return np.ones((self.Nx, self.Ny), dtype=bool)
        component = self._uav_component_labels[self._grid_index(start)]
        return self._uav_component_labels == component

    def uav_path_prefix(self, start, goal, *, macro_step, horizon, first_axis):
        """A* over executable cardinal macro actions, checking every crossed cell.

        Each edge advances up to macro_step cells, stopping at a wall, boundary,
        or the goal, exactly as the shared environment's movement primitive.
        Axis priority breaks ties between equally short action sequences.
        """
        start, goal = self._grid_index(start), self._grid_index(goal)
        macro_step, horizon = max(1, int(macro_step)), max(1, int(horizon))
        if first_axis not in (0, 1):
            raise ValueError("first_axis must be 0 or 1")
        if not self.is_uav_position_valid(start) or not self.is_uav_position_valid(goal):
            return ()
        if start == goal:
            return (goal,)
        if not np.isfinite(self.uav_shortest_path_distance(start, goal)):
            return ()
        key = (start, goal, macro_step, first_axis)
        cached = self._uav_route_cache.get(key)
        if cached is not None:
            return cached[:horizon]

        def heuristic(cell):
            return (abs(cell[0] - goal[0]) + abs(cell[1] - goal[1]) + macro_step - 1) // macro_step

        serial = itertools.count()
        frontier = [(heuristic(start), heuristic(start), next(serial), 0, start)]
        cost, parent = {start: 0}, {}
        route = ()
        while frontier:
            _, _, _, depth, cell = heapq.heappop(frontier)
            if cost.get(cell) != depth:
                continue
            if cell == goal:
                reverse = []
                while cell != start:
                    reverse.append(cell)
                    cell = parent[cell]
                route = tuple(reversed(reverse))
                break
            directions = []
            for axis in (first_axis, 1 - first_axis):
                sign = 1 if goal[axis] >= cell[axis] else -1
                offset = (sign, 0) if axis == 0 else (0, sign)
                directions.append(offset)
            directions += [(-dx, -dy) for dx, dy in directions]
            for dx, dy in directions:
                endpoint = cell
                for _ in range(macro_step):
                    proposed = (endpoint[0] + dx, endpoint[1] + dy)
                    if not self.is_uav_position_valid(proposed):
                        break
                    endpoint = proposed
                    if endpoint == goal:
                        break
                next_depth = depth + 1
                if endpoint == cell or next_depth >= cost.get(endpoint, float("inf")):
                    continue
                cost[endpoint], parent[endpoint] = next_depth, cell
                h = heuristic(endpoint)
                heapq.heappush(frontier, (next_depth + h, h, next(serial), next_depth, endpoint))
        if len(self._uav_route_cache) >= 128:
            self._uav_route_cache.clear()
        self._uav_route_cache[key] = route
        return route[:horizon]
