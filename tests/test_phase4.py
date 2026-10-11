"""
tests/test_phase4.py
====================
Automated test suite for Phase 4: Autonomous Vehicle Movement & Path Following
- Vehicle initialization
- Movement toward next waypoint
- Waypoint progression
- Speed limits enforcement
- Correct arrival at destination
- Pause and resume behaviour
- Reset behaviour
- Collision prevention and BLOCKED state
- Blocking movement when no valid path exists
"""

import sys
import os
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.grid import OccupancyGrid
from src.planner import Planner
from src.vehicle import Vehicle, VehicleState, grid_to_world


def test_vehicle_initialization():
    """Verify vehicle initializes at start position with 0 speed and IDLE state."""
    veh = Vehicle()
    start_x, start_y = grid_to_world(config.START_GRID_COL, config.START_GRID_ROW)
    assert math.isclose(veh.x, start_x, abs_tol=1e-3)
    assert math.isclose(veh.y, start_y, abs_tol=1e-3)
    assert veh.speed == 0.0
    assert veh.state == VehicleState.IDLE
    assert veh.target_waypoint_index == 0
    assert veh.elapsed_time == 0.0
    assert veh.total_waypoints == 0


def test_no_path_movement_blocked():
    """Verify vehicle does not move if no valid path exists."""
    grid = OccupancyGrid()
    veh = Vehicle()

    # Attempt to start without path
    assert veh.start() is False
    assert veh.state == VehicleState.IDLE

    # Update should not move position
    initial_x = veh.x
    veh.update(0.1, grid)
    assert veh.x == initial_x
    assert veh.speed == 0.0


def test_waypoint_progression_and_movement():
    """Verify vehicle starts moving and progresses through waypoints."""
    grid = OccupancyGrid()
    planner = Planner()
    path_res = planner.plan(grid)
    assert path_res.success is True

    veh = Vehicle()
    assert veh.set_path(path_res.path) is True
    assert veh.total_waypoints == len(path_res.path)
    assert veh.start() is True
    assert veh.state == VehicleState.MOVING

    # Run small physics steps
    initial_x = veh.x
    for _ in range(10):
        veh.update(0.05, grid)

    # Position must have moved forward along +X axis towards destination
    assert veh.x > initial_x
    assert veh.speed > 0.0
    assert veh.speed <= veh.max_speed


def test_speed_limit_enforcement():
    """Verify vehicle accelerates but never exceeds max_speed."""
    grid = OccupancyGrid()
    planner = Planner()
    path_res = planner.plan(grid)

    veh = Vehicle(max_speed=80.0, acceleration=500.0)
    veh.set_path(path_res.path)
    veh.start()

    # Step for several seconds to allow maximum acceleration
    for _ in range(50):
        veh.update(0.05, grid)

    assert veh.speed <= 80.0
    assert math.isclose(veh.speed, 80.0, abs_tol=1e-2)


def test_arrival_at_destination():
    """Verify vehicle smoothly travels the full path and stops at ARRIVED."""
    grid = OccupancyGrid()
    planner = Planner()
    path_res = planner.plan(grid)

    # Fast vehicle for quick testing
    veh = Vehicle(max_speed=300.0, acceleration=600.0, deceleration=600.0)
    veh.set_path(path_res.path)
    veh.start()

    # Simulate up to 5 seconds of motion
    max_steps = 200
    for _ in range(max_steps):
        veh.update(0.05, grid)
        if veh.state == VehicleState.ARRIVED:
            break

    assert veh.state == VehicleState.ARRIVED
    assert veh.speed == 0.0
    # Destination coordinates
    dest_x, dest_y = grid_to_world(config.DEST_GRID_COL, config.DEST_GRID_ROW)
    assert math.isclose(veh.x, dest_x, abs_tol=1e-1)
    assert math.isclose(veh.y, dest_y, abs_tol=1e-1)


def test_pause_and_resume():
    """Verify pause freezes movement and resume continues seamlessly."""
    grid = OccupancyGrid()
    planner = Planner()
    path_res = planner.plan(grid)

    veh = Vehicle()
    veh.set_path(path_res.path)
    veh.start()

    # Move a few steps
    veh.update(0.1, grid)
    assert veh.state == VehicleState.MOVING
    pos_x = veh.x

    # Pause
    veh.pause()
    assert veh.state == VehicleState.PAUSED
    veh.update(0.1, grid)
    assert veh.x == pos_x  # No motion while paused

    # Resume
    veh.resume()
    assert veh.state == VehicleState.MOVING
    veh.update(0.1, grid)
    assert veh.x > pos_x  # Motion resumes


def test_reset_behaviour():
    """Verify reset restores initial position, 0 speed, and IDLE state."""
    grid = OccupancyGrid()
    planner = Planner()
    path_res = planner.plan(grid)

    veh = Vehicle()
    veh.set_path(path_res.path)
    veh.start()

    # Move for 0.5s
    for _ in range(10):
        veh.update(0.05, grid)
    assert veh.x > grid_to_world(config.START_GRID_COL, config.START_GRID_ROW)[0]

    # Reset
    veh.reset()
    start_x, start_y = grid_to_world(config.START_GRID_COL, config.START_GRID_ROW)
    assert math.isclose(veh.x, start_x, abs_tol=1e-3)
    assert math.isclose(veh.y, start_y, abs_tol=1e-3)
    assert veh.speed == 0.0
    assert veh.state == VehicleState.IDLE


def test_collision_prevention_stops_vehicle():
    """Verify vehicle halts and enters BLOCKED state before entering an obstacle."""
    grid = OccupancyGrid()
    planner = Planner()
    # Plan empty path first
    path_res = planner.plan(grid)

    veh = Vehicle()
    veh.set_path(path_res.path)
    veh.start()

    # Now drop an obstacle directly in the vehicle's path ahead
    obstacle_col = config.START_GRID_COL + 2
    obstacle_row = config.START_GRID_ROW
    assert grid.add_obstacle(obstacle_col, obstacle_row) is True

    # Advance simulation until vehicle detects blocked route or encounters obstacle
    for _ in range(20):
        veh.update(0.05, grid)
        if veh.state == VehicleState.BLOCKED:
            break

    assert veh.state == VehicleState.BLOCKED
    assert veh.speed == 0.0
    # Ensure vehicle never enters the occupied cell
    assert not (veh.grid_col == obstacle_col and veh.grid_row == obstacle_row)
