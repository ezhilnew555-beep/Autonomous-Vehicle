"""
tests/test_phase2.py
====================
Automated test suite for Phase 2:
- Occupancy-grid initialization
- Adding an obstacle
- Removing an obstacle
- Rejecting out-of-bounds coordinates
- Preventing obstacles from overlapping start and destination
- Clearing and resetting the environment
- Road mask constraints
"""

import sys
import os
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.grid import OccupancyGrid, CELL_FREE, CELL_OCCUPIED


def test_occupancy_grid_initialization():
    """Verify occupancy grid initializes with correct shape and all cells are 0 (free)."""
    grid = OccupancyGrid()
    assert grid.obstacle_count == 0
    assert grid.grid_array.shape == (config.GRID_ROWS, config.GRID_COLS)
    assert np.all(grid.grid_array == CELL_FREE)


def test_add_valid_obstacle():
    """Verify adding an obstacle on a valid road cell succeeds and updates count."""
    grid = OccupancyGrid()
    # Find a valid road cell that is neither start nor destination
    col = config.START_GRID_COL + 2
    row = config.ROAD_ROW_CENTER

    assert grid.is_road(col, row)
    assert not grid.is_protected(col, row)

    # Add obstacle
    success = grid.add_obstacle(col, row)
    assert success is True
    assert grid.is_occupied(col, row)
    assert not grid.is_free(col, row)
    assert grid.obstacle_count == 1
    assert grid.grid_array[row, col] == CELL_OCCUPIED

    # Adding again to the same spot should fail
    assert grid.add_obstacle(col, row) is False
    assert grid.obstacle_count == 1


def test_remove_obstacle():
    """Verify removing an obstacle clears the cell and updates count."""
    grid = OccupancyGrid()
    col = config.START_GRID_COL + 2
    row = config.ROAD_ROW_CENTER

    grid.add_obstacle(col, row)
    assert grid.obstacle_count == 1

    # Remove existing
    removed = grid.remove_obstacle(col, row)
    assert removed is True
    assert grid.obstacle_count == 0
    assert grid.is_free(col, row)
    assert grid.grid_array[row, col] == CELL_FREE

    # Removing from empty cell should return False
    assert grid.remove_obstacle(col, row) is False


def test_reject_out_of_bounds():
    """Verify out-of-bounds coordinates are rejected for adding, removing, and querying."""
    grid = OccupancyGrid()

    bad_coords = [
        (-1, 0),
        (0, -1),
        (config.GRID_COLS, 0),
        (0, config.GRID_ROWS),
        (100, 100),
    ]

    for col, row in bad_coords:
        assert grid.in_bounds(col, row) is False
        assert grid.is_road(col, row) is False
        assert grid.add_obstacle(col, row) is False
        assert grid.remove_obstacle(col, row) is False
        assert grid.is_free(col, row) is False
        assert grid.is_occupied(col, row) is False


def test_prevent_overlap_start_and_destination():
    """Verify obstacles cannot be placed on start or destination markers."""
    grid = OccupancyGrid()

    # Start marker
    assert grid.is_protected(config.START_GRID_COL, config.START_GRID_ROW)
    assert grid.add_obstacle(config.START_GRID_COL, config.START_GRID_ROW) is False
    assert grid.is_occupied(config.START_GRID_COL, config.START_GRID_ROW) is False

    # Destination marker
    assert grid.is_protected(config.DEST_GRID_COL, config.DEST_GRID_ROW)
    assert grid.add_obstacle(config.DEST_GRID_COL, config.DEST_GRID_ROW) is False
    assert grid.is_occupied(config.DEST_GRID_COL, config.DEST_GRID_ROW) is False

    assert grid.obstacle_count == 0


def test_reject_non_road_cells():
    """Verify obstacles cannot be placed outside the valid road surface."""
    grid = OccupancyGrid()

    # A cell clearly above the road
    off_road_row = 0
    col = config.START_GRID_COL + 2
    assert grid.is_road(col, off_road_row) is False
    assert grid.add_obstacle(col, off_road_row) is False
    assert grid.obstacle_count == 0


def test_clear_obstacles():
    """Verify clear_obstacles removes all placed obstacles."""
    grid = OccupancyGrid()

    # Add several obstacles
    c1, r1 = config.START_GRID_COL + 1, config.ROAD_ROW_CENTER
    c2, r2 = config.START_GRID_COL + 2, config.ROAD_ROW_CENTER
    c3, r3 = config.START_GRID_COL + 3, config.ROAD_ROW_CENTER

    grid.add_obstacle(c1, r1)
    grid.add_obstacle(c2, r2)
    grid.add_obstacle(c3, r3)
    assert grid.obstacle_count == 3

    grid.clear_obstacles()
    assert grid.obstacle_count == 0
    assert np.all(grid.grid_array == CELL_FREE)


def test_reset_environment():
    """Verify reset restores the environment to clean state."""
    grid = OccupancyGrid()

    c1, r1 = config.START_GRID_COL + 1, config.ROAD_ROW_CENTER
    grid.add_obstacle(c1, r1)
    assert grid.obstacle_count == 1

    grid.reset()
    assert grid.obstacle_count == 0
    assert grid.is_free(c1, r1)
