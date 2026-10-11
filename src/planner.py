"""
src/planner.py
==============
A* Path Planning implementation for Autonomous Vehicle Simulation (Phase 3).

Key Features:
- Standard A* search using f(n) = g(n) + h(n).
- Admissible Manhattan distance heuristic for 4-directional movements (and Octile/Euclidean for diagonal).
- Diagonal movement support with strict corner-cutting prevention:
    A diagonal step from (c, r) to (c+dc, r+dr) is only legal if neither
    (c+dc, r) nor (c, r+dr) is occupied.
- Traversability restricted to valid road cells.
- Endpoints validation (start and goal must be in-bounds, on the road, and not occupied).
- Clear PathResult data structure recording:
    - success (bool)
    - path (List[Tuple[int, int]])
    - path_cost (float)
    - cell_count (int)
    - nodes_expanded (int)
    - execution_time_ms (float)
    - message (str)
- Strictly decoupled from UI and rendering.
"""

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import config

Path = List[Tuple[int, int]]  # sequence of (col, row) grid coordinates


@dataclass
class PathResult:
    """Detailed outcome of a path planning attempt (A*, Dijkstra, or GBFS)."""
    success: bool
    path: Path = field(default_factory=list)
    path_cost: float = 0.0
    cell_count: int = 0
    nodes_expanded: int = 0
    execution_time_ms: float = 0.0
    message: str = ""
    algorithm: str = "astar"

    @property
    def status_label(self) -> str:
        if self.success:
            return "SUCCESS"
        return "FAILED"


def manhattan_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Manhattan distance heuristic h(n) = |x1 - x2| + |y1 - y2|."""
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def octile_distance(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Octile distance heuristic for 8-directional movement."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    # cost of straight = 1.0, cost of diagonal = sqrt(2) ~= 1.41421356
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


class Planner:
    """Path Planner supporting A*, Dijkstra, and Greedy Best-First Search (GBFS)."""

    def __init__(self, allow_diagonal: bool = True, algorithm: str = "astar") -> None:
        self.allow_diagonal = allow_diagonal
        self.algorithm = algorithm.lower()
        self._last_result: Optional[PathResult] = None

    @property
    def last_result(self) -> Optional[PathResult]:
        return self._last_result

    @property
    def path(self) -> Path:
        if self._last_result and self._last_result.success:
            return self._last_result.path
        return []

    def clear(self) -> None:
        """Clear planning state and cached path."""
        self._last_result = None

    def plan(
        self,
        grid,  # OccupancyGrid
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
        algorithm: Optional[str] = None,
    ) -> PathResult:
        """
        Compute a collision-free path from start to goal under selected algorithm.

        Parameters:
            grid: OccupancyGrid instance providing bounds and cell status.
            start: (col, row) tuple. If None, defaults to grid.start_pos or config default.
            goal: (col, row) tuple. If None, defaults to grid.dest_pos or config default.
            algorithm: "astar" (default), "dijkstra", or "gbfs". Overrides self.algorithm if provided.

        Returns:
            PathResult with details of the computed path or failure reason.
        """
        start_time = time.perf_counter()
        algo = (algorithm or self.algorithm).lower()
        if algo not in ("astar", "dijkstra", "gbfs"):
            algo = "astar"

        if start is None:
            start = getattr(grid, "start_pos", (config.START_GRID_COL, config.START_GRID_ROW))
        if goal is None:
            goal = getattr(grid, "dest_pos", (config.DEST_GRID_COL, config.DEST_GRID_ROW))

        # 1. Validate endpoints
        endpoint_err = self._validate_endpoint(grid, start, "Start")
        if endpoint_err:
            res = PathResult(success=False, message=endpoint_err, algorithm=algo)
            res.execution_time_ms = (time.perf_counter() - start_time) * 1000.0
            self._last_result = res
            return res

        endpoint_err = self._validate_endpoint(grid, goal, "Goal")
        if endpoint_err:
            res = PathResult(success=False, message=endpoint_err, algorithm=algo)
            res.execution_time_ms = (time.perf_counter() - start_time) * 1000.0
            self._last_result = res
            return res

        # Trivial path if start == goal
        if start == goal:
            res = PathResult(
                success=True,
                path=[start],
                path_cost=0.0,
                cell_count=1,
                nodes_expanded=0,
                message="Start is destination",
                algorithm=algo,
            )
            res.execution_time_ms = (time.perf_counter() - start_time) * 1000.0
            self._last_result = res
            return res

        # 2. Heuristic function (Octile for 8-dir, Manhattan for 4-dir)
        raw_heuristic = octile_distance if self.allow_diagonal else manhattan_distance

        def h(node: Tuple[int, int]) -> float:
            if algo == "dijkstra":
                return 0.0
            return raw_heuristic(node, goal)

        # 3. Priority queue (min-heap) entries: (priority_f, g_score, counter, node)
        counter = 0
        open_set = []

        if algo == "gbfs":
            f_start = h(start)
        elif algo == "dijkstra":
            f_start = 0.0
        else:  # astar
            f_start = h(start)

        heapq.heappush(open_set, (f_start, 0.0, counter, start))

        # Maps node -> cheapest known cost from start to node
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}

        # Maps node -> parent node that gave the lowest g_score
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # Closed set: set of nodes whose shortest path is already finalized
        closed_set = set()

        nodes_expanded = 0

        while open_set:
            current_f, current_g, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            closed_set.add(current)
            nodes_expanded += 1

            # Goal reached!
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                exec_time = (time.perf_counter() - start_time) * 1000.0
                res = PathResult(
                    success=True,
                    path=path,
                    path_cost=round(current_g, 3),
                    cell_count=len(path),
                    nodes_expanded=nodes_expanded,
                    execution_time_ms=round(exec_time, 3),
                    message="Path found successfully",
                    algorithm=algo,
                )
                self._last_result = res
                return res

            # Explore neighbors
            for neighbor, step_cost in self._get_valid_neighbors(grid, current):
                if neighbor in closed_set:
                    continue

                tentative_g = current_g + step_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    if algo == "gbfs":
                        f_score = h(neighbor)
                    elif algo == "dijkstra":
                        f_score = tentative_g
                    else:  # astar
                        f_score = tentative_g + h(neighbor)

                    counter += 1
                    heapq.heappush(open_set, (f_score, tentative_g, counter, neighbor))

        # Open set exhausted without reaching goal -> Unreachable
        exec_time = (time.perf_counter() - start_time) * 1000.0
        res = PathResult(
            success=False,
            path=[],
            path_cost=0.0,
            cell_count=0,
            nodes_expanded=nodes_expanded,
            execution_time_ms=round(exec_time, 3),
            message="No path found (goal unreachable)",
            algorithm=algo,
        )
        self._last_result = res
        return res

    def compute_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid=None,
    ) -> Path:
        """Compatibility wrapper returning List[Tuple[int, int]]."""
        if grid is None:
            return []
        res = self.plan(grid, start, goal)
        return res.path

    # ------------------------------------------------------------------ #
    # Helper validation and neighbor generation
    # ------------------------------------------------------------------ #
    def _validate_endpoint(self, grid, pos: Tuple[int, int], name: str) -> Optional[str]:
        col, row = pos
        if not grid.in_bounds(col, row):
            return f"{name} position {pos} is out of bounds."
        if not grid.is_road(col, row):
            return f"{name} position {pos} is not on the road."
        if grid.is_occupied(col, row):
            return f"{name} position {pos} is blocked by an obstacle."
        return None

    def _get_valid_neighbors(
        self, grid, current: Tuple[int, int]
    ) -> List[Tuple[Tuple[int, int], float]]:
        """
        Generate legal neighboring cells and their step costs.
        Diagonal movement prevents corner cutting:
        Moving (c, r) -> (c+dc, r+dr) requires both (c+dc, r) and (c, r+dr) to be free and on road.
        """
        c, r = current
        neighbors: List[Tuple[Tuple[int, int], float]] = []

        # 4 cardinal directions (dx, dy, cost)
        cardinals = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
        ]

        # Check cardinal neighbors
        cardinal_valid: Dict[Tuple[int, int], bool] = {}
        for dc, dr, cost in cardinals:
            nc, nr = c + dc, r + dr
            is_valid = (
                grid.in_bounds(nc, nr)
                and grid.is_road(nc, nr)
                and not grid.is_occupied(nc, nr)
            )
            cardinal_valid[(dc, dr)] = is_valid
            if is_valid:
                neighbors.append(((nc, nr), cost))

        # 4 diagonal directions
        if self.allow_diagonal:
            diagonals = [
                (1, 1, math.sqrt(2)),
                (1, -1, math.sqrt(2)),
                (-1, 1, math.sqrt(2)),
                (-1, -1, math.sqrt(2)),
            ]
            for dc, dr, cost in diagonals:
                nc, nr = c + dc, r + dr
                # Cell itself must be in bounds, on road, and free
                if not (grid.in_bounds(nc, nr) and grid.is_road(nc, nr) and not grid.is_occupied(nc, nr)):
                    continue

                # Corner cutting prevention:
                # Both adjacent cardinal cells (c + dc, r) and (c, r + dr) must be traversable
                if cardinal_valid.get((dc, 0), False) and cardinal_valid.get((0, dr), False):
                    neighbors.append(((nc, nr), cost))

        return neighbors

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]
    ) -> Path:
        """Backtrack from goal to start to produce ordered forward path."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
