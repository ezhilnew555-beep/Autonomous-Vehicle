"""
tests/test_phase7.py
====================
Automated test suite for Phase 7: Collision Avoidance, Safety Supervisor,
and Dynamic Replanning.

Tests:
1. Static collision detection along planned route
2. Dynamic & predicted collision detection with temporal overlap
3. Vehicle footprint and oriented bounding box clearance
4. Conservative handling of uncertain / unreliable predictions
5. Speed reduction under CAUTION state
6. Emergency stopping under imminent collision risk
7. Automatic A* replanning and safe path acceptance
8. No-safe-path handling when all detours are blocked
9. Replanning cooldown enforcement
10. Recovery to SAFE state when obstacle clears
11. Verification of the 5 demonstration scenarios (Safe, Blocked, Crossing, No-Path, Recovery)
"""

import sys
import os
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.grid import OccupancyGrid
from src.obstacles import ObstacleType, DynamicObstacle
from src.perception import PerceivedObject, Observation
from src.prediction import Predictor, PredictedTrajectory
from src.planner import Planner
from src.vehicle import Vehicle, VehicleState, grid_to_world, world_to_grid
from src.collision import (
    SafetySupervisor,
    SafetyState,
    CollisionConflict,
    check_collision,
    check_footprint_collision,
    evaluate_path_conflicts,
)
from src.scenarios import (
    load_safe_route_scenario,
    load_blocked_route_scenario,
    load_crossing_obstacle_scenario,
    load_no_safe_route_scenario,
    load_recovery_scenario,
)


def _create_perceived_track(
    track_id: int = 1,
    category: ObstacleType = ObstacleType.CAR,
    x: float = 200.0,
    y: float = config.ROAD_CENTER_Y,
    vx: float = 0.0,
    vy: float = 0.0,
    history_count: int = 3,
) -> PerceivedObject:
    """Helper creating a PerceivedObject with specified kinematics and history."""
    obj = PerceivedObject(
        track_id=track_id,
        category=category,
        estimated_x=x,
        estimated_y=y,
        estimated_heading=0.0,
        estimated_vx=vx,
        estimated_vy=vy,
        estimated_speed=math.hypot(vx, vy),
    )
    for i in range(history_count):
        obj._history.append(
            Observation(
                timestamp=i * 0.1,
                x=x,
                y=y,
                heading=0.0,
                category=category,
                length=24.0,
                width=14.0,
            )
        )
    return obj


def test_backward_compatible_check_collision():
    """Verify Phase 1 stub function behaves as expected."""
    assert check_collision((2, 3), [(1, 1), (2, 3)]) is True
    assert check_collision((2, 3), [(1, 1), (4, 5)]) is False


def test_footprint_collision_checks():
    """Verify check_footprint_collision tests vehicle boundaries and obstacles."""
    grid = OccupancyGrid()

    # Center of road is clear
    cx, cy = grid_to_world(5, config.ROAD_ROW_CENTER)
    assert check_footprint_collision(cx, cy, heading=0.0, grid=grid) is False

    # Off-road coordinate must trigger footprint collision
    off_road_y = config.ROAD_TOP_Y - 50.0
    assert check_footprint_collision(cx, off_road_y, heading=0.0, grid=grid) is True

    # Cell with static obstacle must trigger footprint collision
    grid.add_obstacle(5, config.ROAD_ROW_CENTER)
    assert check_footprint_collision(cx, cy, heading=0.0, grid=grid) is True


def test_static_collision_detection_along_route():
    """Verify evaluate_path_conflicts detects blocked cells on planned path."""
    grid = OccupancyGrid()
    path = [(2, config.ROAD_ROW_CENTER), (3, config.ROAD_ROW_CENTER), (4, config.ROAD_ROW_CENTER)]
    waypoints = [grid_to_world(c, r) for c, r in path]

    # Clear grid -> No conflicts
    conflicts, unsafe_idx = evaluate_path_conflicts(
        waypoints=waypoints,
        vehicle_x=waypoints[0][0],
        vehicle_y=waypoints[0][1],
        vehicle_speed=50.0,
        vehicle_heading=0.0,
        grid=grid,
        perceived_objects=[],
        predicted_trajectories=[],
    )
    assert len(conflicts) == 0
    assert unsafe_idx is None

    # Block waypoint (3, ROAD_ROW_CENTER)
    grid.add_obstacle(3, config.ROAD_ROW_CENTER)
    conflicts, unsafe_idx = evaluate_path_conflicts(
        waypoints=waypoints,
        vehicle_x=waypoints[0][0],
        vehicle_y=waypoints[0][1],
        vehicle_speed=50.0,
        vehicle_heading=0.0,
        grid=grid,
        perceived_objects=[],
        predicted_trajectories=[],
    )
    assert len(conflicts) > 0
    assert any(c.conflict_type == "STATIC" for c in conflicts)
    assert unsafe_idx == 1


def test_dynamic_collision_detection_temporal_overlap():
    """Verify dynamic obstacle along route is flagged as conflict."""
    grid = OccupancyGrid()
    path = [(2, config.ROAD_ROW_CENTER), (5, config.ROAD_ROW_CENTER), (8, config.ROAD_ROW_CENTER)]
    waypoints = [grid_to_world(c, r) for c, r in path]

    # Moving pedestrian positioned near waypoint 5
    ped_x, ped_y = grid_to_world(5, config.ROAD_ROW_CENTER)
    ped = _create_perceived_track(track_id=10, category=ObstacleType.PEDESTRIAN, x=ped_x, y=ped_y, history_count=3)

    predictor = Predictor(horizon=2.0)
    trajs = predictor.predict_all([ped])

    conflicts, unsafe_idx = evaluate_path_conflicts(
        waypoints=waypoints,
        vehicle_x=waypoints[0][0],
        vehicle_y=waypoints[0][1],
        vehicle_speed=50.0,
        vehicle_heading=0.0,
        grid=grid,
        perceived_objects=[ped],
        predicted_trajectories=trajs,
    )
    assert len(conflicts) > 0
    assert any(c.obstacle_id == 10 for c in conflicts)


def test_conservative_margin_for_uncertain_predictions():
    """Verify unreliable or low-confidence tracks receive wider safety margins."""
    grid = OccupancyGrid()
    path = [(2, config.ROAD_ROW_CENTER), (5, config.ROAD_ROW_CENTER)]
    waypoints = [grid_to_world(c, r) for c, r in path]

    # Create target slightly offset to the side
    ped_x, ped_y = grid_to_world(3, config.ROAD_ROW_CENTER)
    ped_y += 32.0  # lateral offset

    # Case A: Well-tracked object (history = 4, high confidence)
    obj_certain = _create_perceived_track(x=ped_x, y=ped_y, history_count=4)
    predictor = Predictor(horizon=1.0)
    trajs_certain = predictor.predict_all([obj_certain])

    conflicts_certain, _ = evaluate_path_conflicts(
        waypoints=waypoints,
        vehicle_x=waypoints[0][0],
        vehicle_y=waypoints[0][1],
        vehicle_speed=50.0,
        vehicle_heading=0.0,
        grid=grid,
        perceived_objects=[obj_certain],
        predicted_trajectories=trajs_certain,
        base_margin=8.0,
        uncertain_margin=25.0,
    )

    # Case B: Uncertain object (history = 1, low confidence)
    obj_uncertain = _create_perceived_track(x=ped_x, y=ped_y, history_count=1)
    trajs_uncertain = predictor.predict_all([obj_uncertain])

    conflicts_uncertain, _ = evaluate_path_conflicts(
        waypoints=waypoints,
        vehicle_x=waypoints[0][0],
        vehicle_y=waypoints[0][1],
        vehicle_speed=50.0,
        vehicle_heading=0.0,
        grid=grid,
        perceived_objects=[obj_uncertain],
        predicted_trajectories=trajs_uncertain,
        base_margin=8.0,
        uncertain_margin=25.0,
    )

    # The uncertain track's larger buffer should produce equal or more conflicts
    assert len(conflicts_uncertain) >= len(conflicts_certain)


def test_speed_reduction_under_caution_and_cooldown():
    """Verify future warning conflict reduces speed multiplier to 50% and enters CAUTION."""
    supervisor = SafetySupervisor()
    grid = OccupancyGrid()
    planner = Planner()
    vehicle = Vehicle()

    # Route with waypoint 300px ahead
    path = [(config.START_GRID_COL, config.ROAD_ROW_CENTER), (10, config.ROAD_ROW_CENTER)]
    vehicle.set_path(path)
    vehicle.start()
    vehicle.speed = 80.0

    # Put a warning obstacle far enough ahead to not be critical (< 38px), but in lookahead path
    tgt_x, tgt_y = grid_to_world(7, config.ROAD_ROW_CENTER)
    obj = _create_perceived_track(x=tgt_x, y=tgt_y, history_count=3)
    predictor = Predictor(horizon=2.0)
    trajs = predictor.predict_all([obj])

    # Put cooldown active so it doesn't immediately replan
    supervisor.replan_cooldown = 1.0

    state = supervisor.update(
        sim_time=0.5,
        dt=0.05,
        vehicle=vehicle,
        grid=grid,
        planner=planner,
        perceived_objects=[obj],
        predicted_trajectories=trajs,
    )

    assert state == SafetyState.CAUTION
    assert supervisor.speed_multiplier == config.CAUTION_SPEED_FACTOR
    assert vehicle.speed > 0.0  # Still moving, but regulated


def test_emergency_stop_under_imminent_risk():
    """Verify close-range obstacle triggers EMERGENCY_STOP with zero speed."""
    supervisor = SafetySupervisor()
    grid = OccupancyGrid()
    planner = Planner()
    vehicle = Vehicle()

    path = [(config.START_GRID_COL, config.ROAD_ROW_CENTER), (5, config.ROAD_ROW_CENTER)]
    vehicle.set_path(path)
    vehicle.start()
    vehicle.speed = 60.0

    # Place obstacle directly next to vehicle position (< 30px)
    obj = _create_perceived_track(x=vehicle.x + 20.0, y=vehicle.y, history_count=3)
    predictor = Predictor(horizon=1.0)
    trajs = predictor.predict_all([obj])

    state = supervisor.update(
        sim_time=0.5,
        dt=0.05,
        vehicle=vehicle,
        grid=grid,
        planner=planner,
        perceived_objects=[obj],
        predicted_trajectories=trajs,
    )

    assert state == SafetyState.EMERGENCY_STOP
    assert supervisor.speed_multiplier == 0.0
    assert vehicle.speed == 0.0


def test_automatic_replanning_and_safe_path_acceptance():
    """Verify supervisor successfully replans a detour around an obstruction."""
    supervisor = SafetySupervisor()
    grid = OccupancyGrid()
    planner = Planner()
    vehicle = Vehicle()

    # Initial straight line route
    p_res = planner.plan(grid)
    assert p_res.success
    vehicle.set_path(p_res.path)
    vehicle.start()
    vehicle.speed = 50.0

    # Place obstacle midway on centerline
    mid_col = config.GRID_COLS // 2
    mid_row = config.ROAD_ROW_CENTER
    grid.add_obstacle(mid_col, mid_row)

    # Zero cooldown so replan triggers immediately
    supervisor.replan_cooldown = 0.0

    state = supervisor.update(
        sim_time=1.0,
        dt=0.05,
        vehicle=vehicle,
        grid=grid,
        planner=planner,
        perceived_objects=[],
        predicted_trajectories=[],
    )

    # Replan should succeed and transition to CAUTION with valid detour
    assert state == SafetyState.CAUTION
    assert supervisor.total_replans_succeeded >= 1
    # Check that new vehicle path detours around (mid_col, mid_row)
    assert (mid_col, mid_row) not in vehicle.path


def test_no_safe_path_handling():
    """Verify supervisor transitions to NO_SAFE_PATH and halts when all routes are blocked."""
    supervisor = SafetySupervisor()
    grid = OccupancyGrid()
    planner = Planner()
    vehicle = Vehicle()

    vehicle.set_path([(config.START_GRID_COL, config.ROAD_ROW_CENTER), (config.DEST_GRID_COL, config.ROAD_ROW_CENTER)])
    vehicle.start()
    vehicle.speed = 50.0

    # Build impassable wall across all road lanes
    mid_col = config.GRID_COLS // 2
    top_row = config.ROAD_ROW_CENTER - config.ROAD_HALF_WIDTH
    bot_row = config.ROAD_ROW_CENTER + config.ROAD_HALF_WIDTH
    for r in range(top_row, bot_row):
        grid.add_obstacle(mid_col, r)

    supervisor.replan_cooldown = 0.0

    state = supervisor.update(
        sim_time=1.0,
        dt=0.05,
        vehicle=vehicle,
        grid=grid,
        planner=planner,
        perceived_objects=[],
        predicted_trajectories=[],
    )

    assert state == SafetyState.NO_SAFE_PATH
    assert supervisor.speed_multiplier == 0.0
    assert vehicle.speed == 0.0


def test_recovery_when_obstacle_clears():
    """Verify supervisor recovers from CAUTION / STOP to SAFE when road clears."""
    supervisor = SafetySupervisor()
    grid = OccupancyGrid()
    planner = Planner()
    vehicle = Vehicle()

    vehicle.set_path([(config.START_GRID_COL, config.ROAD_ROW_CENTER), (config.DEST_GRID_COL, config.ROAD_ROW_CENTER)])
    vehicle.start()

    # Place temporary obstacle causing CAUTION
    obs_x, obs_y = grid_to_world(5, config.ROAD_ROW_CENTER)
    obj = _create_perceived_track(x=obs_x, y=obs_y)
    trajs = Predictor().predict_all([obj])
    supervisor.replan_cooldown = 2.0

    # First update: CAUTION
    supervisor.update(1.0, 0.05, vehicle, grid, planner, [obj], trajs)
    assert supervisor.state == SafetyState.CAUTION

    # Obstacle moves away: empty perceived objects and predictions
    state_cleared = supervisor.update(1.5, 0.05, vehicle, grid, planner, [], [])
    assert state_cleared == SafetyState.SAFE
    assert supervisor.speed_multiplier == 1.0


def test_demonstration_scenarios_loading():
    """Verify all 5 demonstration scenarios load proper obstacle and state configurations."""
    # 1. Safe route
    s1 = load_safe_route_scenario()
    assert s1.name == "Safe Route"
    assert len(s1.obstacle_positions) == 0

    # 2. Blocked route
    s2 = load_blocked_route_scenario()
    assert s2.name == "Blocked Route"
    assert len(s2.obstacle_positions) > 0

    # 3. Crossing obstacle
    s3 = load_crossing_obstacle_scenario()
    assert s3.name == "Crossing Obstacle"
    assert len(s3.dynamic_obstacles) > 0
    assert s3.dynamic_obstacles[0].category == ObstacleType.PEDESTRIAN

    # 4. No safe route
    s4 = load_no_safe_route_scenario()
    assert s4.name == "No Safe Route"
    assert len(s4.obstacle_positions) >= 4  # full wall across lanes

    # 5. Recovery
    s5 = load_recovery_scenario()
    assert s5.name == "Recovery"
    assert len(s5.dynamic_obstacles) > 0
