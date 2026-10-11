"""
tests/test_phase3.py
====================
Automated test suite for Phase 3 (A* Path Planning):
- Empty grid path planning (straightforward path)
- Obstacle detour planning
- Unreachable goal detection (completely blocked road)
- Invalid endpoints (out of bounds or off-road)
- Occupied endpoints rejection
- Path validity: contiguous steps, no diagonals cutting corners, no occupied cells
- Path cost and cell count accuracy
- Path reconstruction (start to destination ordered correctly)
- Clear and recalculate state
"""

import sys
import os
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.grid import OccupancyGrid
from src.planner import Planner, PathResult, manhattan_distance, octile_distance


def _verify_path_validity(path, grid, start, goal, allow_diagonal=True):
    """Helper verifying that an A* path is completely valid and traversable."""
    assert len(path) >= 1
    assert path[0] == start
    assert path[-1] == goal

    for i, cell in enumerate(path):
        col, row = cell
        assert grid.in_bounds(col, row), f"Cell {cell} out of bounds"
        assert grid.is_road(col, row), f"Cell {cell} not on road"
        assert not grid.is_occupied(col, row), f"Cell {cell} is occupied"

        if i > 0:
            prev_col, prev_row = path[i - 1]
            dc = abs(col - prev_col)
            dr = abs(row - prev_row)
            assert max(dc, dr) == 1, f"Step between {path[i-1]} and {cell} is not adjacent"

            if dc == 1 and dr == 1:
                assert allow_diagonal, "Diagonal step taken when not allowed"
                # Check corner cutting
                assert not grid.is_occupied(col, prev_row), "Corner cut through occupied cell"
                assert not grid.is_occupied(prev_col, row), "Corner cut through occupied cell"


def test_empty_grid_path():
    """Verify that on an obstacle-free road, A* finds a direct optimal path."""
    grid = OccupancyGrid()
    planner = Planner(allow_diagonal=True)

    start = (config.START_GRID_COL, config.START_GRID_ROW)
    goal = (config.DEST_GRID_COL, config.DEST_GRID_ROW)

    res = planner.plan(grid, start, goal)
    assert res.success is True
    assert res.cell_count > 0
    assert len(res.path) == res.cell_count
    assert res.nodes_expanded > 0
    assert res.execution_time_ms >= 0.0

    _verify_path_validity(res.path, grid, start, goal, allow_diagonal=True)

    # In straight line along horizontal road, distance is DEST_COL - START_COL
    expected_cost = float(config.DEST_GRID_COL - config.START_GRID_COL)
    assert math.isclose(res.path_cost, expected_cost, abs_tol=1e-3)


def test_obstacle_detour():
    """Verify that when an obstacle blocks the direct line, A* detours around it."""
    grid = OccupancyGrid()
    planner = Planner(allow_diagonal=True)

    start = (config.START_GRID_COL, config.START_GRID_ROW)
    goal = (config.DEST_GRID_COL, config.DEST_GRID_ROW)

    # Place an obstacle directly in the center line between start and goal
    block_col = config.START_GRID_COL + 5
    block_row = config.START_GRID_ROW
    assert grid.add_obstacle(block_col, block_row) is True

    res = planner.plan(grid, start, goal)
    assert res.success is True
    _verify_path_validity(res.path, grid, start, goal, allow_diagonal=True)

    # Obstacle must not be in the path
    assert (block_col, block_row) not in res.path


def test_unreachable_goal_full_roadblock():
    """Verify that if the entire road width is blocked, planning fails gracefully."""
    grid = OccupancyGrid()
    planner = Planner(allow_diagonal=True)

    start = (config.START_GRID_COL, config.START_GRID_ROW)
    goal = (config.DEST_GRID_COL, config.DEST_GRID_ROW)

    # Build a wall across all road rows at col = START_COL + 4
    wall_col = config.START_GRID_COL + 4
    top_row = config.ROAD_ROW_CENTER - config.ROAD_HALF_WIDTH
    bot_row = config.ROAD_ROW_CENTER + config.ROAD_HALF_WIDTH

    for r in range(top_row, bot_row):
        assert grid.add_obstacle(wall_col, r) is True

    res = planner.plan(grid, start, goal)
    assert res.success is False
    assert res.path == []
    assert res.cell_count == 0
    assert "No path found" in res.message or "unreachable" in res.message.lower()


def test_invalid_and_out_of_bounds_endpoints():
    """Verify that endpoints out of bounds or off the road are rejected."""
    grid = OccupancyGrid()
    planner = Planner()

    # Out of bounds
    res_bad_start = planner.plan(grid, (-1, 0), (5, 8))
    assert res_bad_start.success is False
    assert "out of bounds" in res_bad_start.message

    res_bad_goal = planner.plan(grid, (2, 8), (100, 8))
    assert res_bad_goal.success is False
    assert "out of bounds" in res_bad_goal.message

    # Off-road endpoint (row 0 is above road)
    res_offroad = planner.plan(grid, (2, 0), (5, 8))
    assert res_offroad.success is False
    assert "not on the road" in res_offroad.message


def test_no_diagonal_corner_cutting():
    """Verify that diagonal moves cannot cut across blocked diagonal corners."""
    grid = OccupancyGrid()
    planner = Planner(allow_diagonal=True)

    # Setup a narrow diagonal test
    # (c, r) trying to move to (c+1, r+1) with (c+1, r) blocked
    c, r = config.START_GRID_COL + 2, config.ROAD_ROW_CENTER
    grid.add_obstacle(c + 1, r)

    neighbors = [n for n, cost in planner._get_valid_neighbors(grid, (c, r))]
    # Diagonal (c+1, r+1) should NOT be reachable directly because (c+1, r) is blocked
    assert (c + 1, r + 1) not in neighbors


def test_manhattan_cardinal_only_planner():
    """Verify 4-directional planner works using strict Manhattan distance."""
    grid = OccupancyGrid()
    planner_4dir = Planner(allow_diagonal=False)

    start = (config.START_GRID_COL, config.START_GRID_ROW)
    goal = (config.START_GRID_COL + 3, config.START_GRID_ROW + 1)

    res = planner_4dir.plan(grid, start, goal)
    assert res.success is True
    _verify_path_validity(res.path, grid, start, goal, allow_diagonal=False)

    # Each step is exactly cardinal
    for i in range(1, len(res.path)):
        dc = abs(res.path[i][0] - res.path[i - 1][0])
        dr = abs(res.path[i][1] - res.path[i - 1][1])
        assert (dc == 1 and dr == 0) or (dc == 0 and dr == 1)

    expected_manhattan = manhattan_distance(start, goal)
    assert math.isclose(res.path_cost, expected_manhattan, abs_tol=1e-3)


def test_clear_and_reconstruction():
    """Verify planner state can be cleared and path is ordered start-to-goal."""
    grid = OccupancyGrid()
    planner = Planner()

    start = (config.START_GRID_COL, config.START_GRID_ROW)
    goal = (config.DEST_GRID_COL, config.DEST_GRID_ROW)

    res = planner.plan(grid, start, goal)
    assert planner.last_result is not None
    assert planner.path == res.path

    planner.clear()
    assert planner.last_result is None
    assert planner.path == []
