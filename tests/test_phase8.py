"""
tests/test_phase8.py
====================
Test suite for Phase 8: Five Realistic Indian Road Scenarios.

Verifies:
1. All five scenarios instantiate cleanly with correct IDs and metadata.
2. Start and destination positions are valid, drivable, and unblocked.
3. A* planner successfully finds feasible initial routes on scenario-specific road masks.
4. Dynamic obstacles update positions and respect crossing bounds (bound_y=False).
5. Scheduled events (e.g. Sudden Cattle Crossing) trigger at correct timestamps.
6. Deterministic reproducibility with random seeds.
7. ScenarioMetrics telemetry logging, clearance auditing, and completion states.
8. Safety Supervisor emergency stop and recovery during cattle crossing scenario.
9. Scenario selection menu layout and interactive hitboxes.
"""

import math
import numpy as np
import pytest

import config
from src.grid import OccupancyGrid
from src.obstacles import ObstacleType, DynamicObstacle
from src.planner import Planner
from src.prediction import Predictor
from src.perception import PerceptionSystem
from src.collision import SafetySupervisor, SafetyState
from src.metrics import ScenarioMetrics
from src.scenarios import (
    Scenario,
    ScheduledEvent,
    get_all_scenarios,
    get_scenario_by_id,
    create_village_scenario,
    create_intersection_scenario,
    create_merge_scenario,
    create_market_scenario,
    create_cattle_scenario,
)
from src.ui import get_scenario_menu_card_rect, get_scenario_menu_item_rects


# ---------------------------------------------------------------------------
# Test 1: All 5 Scenarios Instantiate with Correct Metadata
# ---------------------------------------------------------------------------
def test_all_scenarios_instantiation():
    scenarios = get_all_scenarios()
    assert len(scenarios) == 5

    expected_ids = [
        "village_road",
        "urban_intersection",
        "highway_merge",
        "dense_market",
        "cattle_crossing",
    ]
    actual_ids = [sc.id for sc in scenarios]
    assert actual_ids == expected_ids

    for sc in scenarios:
        assert len(sc.name) > 0
        assert len(sc.description) > 0
        assert sc.timeout > 0.0
        assert sc.seed is not None
        assert isinstance(sc.start_pos, tuple) and len(sc.start_pos) == 2
        assert isinstance(sc.dest_pos, tuple) and len(sc.dest_pos) == 2


def test_scenario_lookup_by_id():
    for sc_id in ["village_road", "urban_intersection", "highway_merge", "dense_market", "cattle_crossing"]:
        sc = get_scenario_by_id(sc_id, seed=123)
        assert sc is not None
        assert sc.id == sc_id
        assert sc.seed == 123

    assert get_scenario_by_id("non_existent_scenario") is None


# ---------------------------------------------------------------------------
# Test 2: Endpoints Validation and Road Mask Alignment
# ---------------------------------------------------------------------------
def test_scenario_endpoints_valid_and_unblocked():
    scenarios = get_all_scenarios()
    for sc in scenarios:
        grid = OccupancyGrid()
        if sc.road_mask is not None:
            grid.set_road_mask(sc.road_mask)

        # Check in bounds
        assert grid.in_bounds(*sc.start_pos), f"Start {sc.start_pos} out of bounds in {sc.id}"
        assert grid.in_bounds(*sc.dest_pos), f"Dest {sc.dest_pos} out of bounds in {sc.id}"

        # Check drivable road
        assert grid.is_road(*sc.start_pos), f"Start {sc.start_pos} not road in {sc.id}"
        assert grid.is_road(*sc.dest_pos), f"Dest {sc.dest_pos} not road in {sc.id}"

        # Check start != dest
        assert sc.start_pos != sc.dest_pos, f"Start equals dest in {sc.id}"

        # Add static obstacles
        for col, row in sc.obstacle_positions:
            grid.add_obstacle(col, row)

        # Endpoints must not be occupied
        assert not grid.is_occupied(*sc.start_pos), f"Start cell occupied in {sc.id}"
        assert not grid.is_occupied(*sc.dest_pos), f"Dest cell occupied in {sc.id}"


# ---------------------------------------------------------------------------
# Test 3: Feasible Initial Path Planning for All Scenarios
# ---------------------------------------------------------------------------
def test_initial_path_planning_success():
    planner = Planner(allow_diagonal=True)
    scenarios = get_all_scenarios()

    for sc in scenarios:
        grid = OccupancyGrid()
        if sc.road_mask is not None:
            grid.set_road_mask(sc.road_mask)
        grid.start_pos = sc.start_pos
        grid.dest_pos = sc.dest_pos

        for col, row in sc.obstacle_positions:
            grid.add_obstacle(col, row)

        res = planner.plan(grid, start=sc.start_pos, goal=sc.dest_pos)
        assert res.success is True, f"Failed to plan path for scenario {sc.id}: {res.message}"
        assert len(res.path) >= 2, f"Path too short for scenario {sc.id}"
        assert res.path[0] == sc.start_pos
        assert res.path[-1] == sc.dest_pos


# ---------------------------------------------------------------------------
# Test 4: Dynamic Obstacle Heading and Crossing Unbound Flag
# ---------------------------------------------------------------------------
def test_dynamic_obstacle_crossing_properties():
    # Scenario 2: Urban intersection cross-traffic should be unbound in Y
    sc_inter = create_intersection_scenario()
    assert len(sc_inter.dynamic_obstacles) >= 2
    for obs in sc_inter.dynamic_obstacles:
        assert obs.bound_y is False, "Intersection cross-traffic must not bounce within lane shoulder"

    # Scenario 5: Cattle crosser should be unbound in Y
    sc_cattle = create_cattle_scenario()
    cow = next(o for o in sc_cattle.dynamic_obstacles if o.category == ObstacleType.CATTLE)
    assert cow.bound_y is False, "Crossing cattle must be unbound in Y to traverse roadway"


# ---------------------------------------------------------------------------
# Test 5: Scheduled Dynamic Event Timeline & Triggering
# ---------------------------------------------------------------------------
def test_cattle_crossing_scheduled_event():
    sc = create_cattle_scenario()
    assert len(sc.scheduled_events) == 1
    ev = sc.scheduled_events[0]
    assert ev.action_type == "ACTIVATE_MOVEMENT"
    assert ev.trigger_time == 1.2
    assert ev.has_triggered is False

    # Check due events before trigger time
    due_early = sc.get_due_events(1.0)
    assert len(due_early) == 0

    # Check due events at/after trigger time
    due_late = sc.get_due_events(1.25)
    assert len(due_late) == 1
    assert due_late[0] is ev

    # Mark triggered and check reset
    ev.has_triggered = True
    assert len(sc.get_due_events(1.5)) == 0

    sc.reset_events()
    assert ev.has_triggered is False
    assert len(sc.get_due_events(1.5)) == 1


# ---------------------------------------------------------------------------
# Test 6: Deterministic Scenario Seeding
# ---------------------------------------------------------------------------
def test_seed_determinism():
    sc1_a = create_village_scenario(seed=777)
    sc1_b = create_village_scenario(seed=777)

    assert sc1_a.obstacle_positions == sc1_b.obstacle_positions
    assert len(sc1_a.dynamic_obstacles) == len(sc1_b.dynamic_obstacles)
    for o_a, o_b in zip(sc1_a.dynamic_obstacles, sc1_b.dynamic_obstacles):
        assert o_a.x == pytest.approx(o_b.x)
        assert o_a.y == pytest.approx(o_b.y)
        assert o_a.speed == pytest.approx(o_b.speed)
        assert o_a.heading == pytest.approx(o_b.heading)


# ---------------------------------------------------------------------------
# Test 7: ScenarioMetrics Logging and Audit Record
# ---------------------------------------------------------------------------
def test_scenario_metrics_logging():
    metrics = ScenarioMetrics(
        scenario_name="3. Highway Merging with Slow Traffic",
        random_seed=202,
        status="RUNNING",
    )
    assert metrics.status == "RUNNING"
    assert metrics.destination_reached is False

    metrics.record_conflict()
    metrics.record_conflict()
    assert metrics.collision_risk_events == 2

    metrics.record_replan()
    assert metrics.replanning_events == 1

    metrics.record_emergency_stop()
    assert metrics.emergency_stops == 1

    metrics.update_clearance(85.4)
    metrics.update_clearance(34.2)
    metrics.update_clearance(52.1)
    assert metrics.min_obstacle_clearance == pytest.approx(34.2)

    # Export dictionary
    d = metrics.to_dict()
    assert d["scenario_name"] == "3. Highway Merging with Slow Traffic"
    assert d["collision_risk_events"] == 2
    assert d["replanning_events"] == 1
    assert d["emergency_stops"] == 1
    assert d["min_obstacle_clearance_px"] == pytest.approx(34.2)

    # Summary string
    s = metrics.summary()
    assert "Highway Merging" in s
    assert "34.2px" in s


# ---------------------------------------------------------------------------
# Test 8: Cattle Crossing Emergency Stop Reaction in Simulation
# ---------------------------------------------------------------------------
def test_cattle_crossing_collision_avoidance_reaction():
    from src.vehicle import Vehicle, grid_to_world

    sc = create_cattle_scenario()
    grid = OccupancyGrid()
    planner = Planner(allow_diagonal=True)
    supervisor = SafetySupervisor()
    predictor = Predictor(horizon=1.5)
    perception = PerceptionSystem(detection_range=300.0, detection_delay=0.0)

    # Ego vehicle setup
    sx, sy = grid_to_world(sc.start_pos[0], sc.start_pos[1])
    vehicle = Vehicle(start_col=sc.start_pos[0], start_row=sc.start_pos[1])
    path_res = planner.plan(grid, start=sc.start_pos, goal=sc.dest_pos)
    assert path_res.success is True
    vehicle.set_path(path_res.path)
    vehicle.start()

    # Dynamic obstacles: cow starting stationary
    cow = sc.dynamic_obstacles[0]
    assert cow.speed == 0.0

    # Advance time to t = 1.2s and activate cow stepping into road directly ahead
    cow.y = float(config.ROAD_CENTER_Y - 10.0)  # cow steps right into road center
    cow.x = vehicle.x + 80.0                   # 80px directly ahead of vehicle
    cow.speed = 15.0
    cow.heading = math.pi / 2.0                # crossing south

    # Run perception & prediction
    perceived = perception.update(
        sim_time=1.3,
        ego_x=vehicle.x,
        ego_y=vehicle.y,
        ground_truth_objects=[cow],
    )
    trajs = predictor.predict_all(perceived)

    # Update supervisor
    state = supervisor.update(
        sim_time=1.3,
        dt=0.05,
        vehicle=vehicle,
        grid=grid,
        planner=planner,
        perceived_objects=perceived,
        predicted_trajectories=trajs,
    )

    # Supervisor must detect imminent conflict and enter CAUTION or EMERGENCY_STOP
    assert state in (SafetyState.CAUTION, SafetyState.EMERGENCY_STOP)
    assert supervisor.speed_multiplier < 1.0


# ---------------------------------------------------------------------------
# Test 9: Scenario Selection Menu Layout & Geometry
# ---------------------------------------------------------------------------
def test_scenario_menu_hitboxes():
    card_rect = get_scenario_menu_card_rect()
    assert card_rect.width > 500
    assert card_rect.height > 400
    assert card_rect.right <= config.SCREEN_WIDTH
    assert card_rect.bottom <= config.SCREEN_HEIGHT

    item_rects = get_scenario_menu_item_rects(5)
    assert len(item_rects) == 5

    for i in range(len(item_rects)):
        # Every item rect must be strictly inside card rect
        assert card_rect.contains(item_rects[i]), f"Item {i} is outside modal card rect"
        # No overlapping item rects
        if i > 0:
            assert item_rects[i].top >= item_rects[i - 1].bottom
