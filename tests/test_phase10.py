"""
tests/test_phase10.py
=====================
Automated test suite for Phase 10: Final Integration, Documentation,
and Deterministic Project Demonstration.

Tests:
1. Phase 10 configuration metadata & window title.
2. DemoController lifecycle: start, advance, step back, stop.
3. Deterministic 7-stage demonstration workflow metadata.
4. Execution of demonstration stages on headless simulation.
5. Interactive multi-algorithm cycling (A*, Dijkstra, GBFS).
6. End-to-end integration: perception -> prediction -> supervisor -> replanner.
7. UI geometry verification (status bar, demo banner, side panel).
8. Environment reset behavior and state isolation.
"""

import math
import os
import sys
import pytest
import pygame

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.grid import OccupancyGrid
from src.obstacles import ObstacleType, DynamicObstacle
from src.planner import Planner
from src.vehicle import Vehicle, VehicleState
from src.perception import PerceptionSystem
from src.prediction import Predictor
from src.collision import SafetySupervisor, SafetyState
from src.scenarios import get_all_scenarios, get_scenario_by_id
from src.demo import DemoController, DemoStage, STAGE_METADATA
from src.simulation import Simulation


# ---------------------------------------------------------------------------
# Test 1: Phase 10 Metadata & System Configuration
# ---------------------------------------------------------------------------
def test_phase10_metadata_and_configuration():
    """Verify Phase 10 version identifiers, window title, and planner lists."""
    assert config.CURRENT_PHASE == 10
    assert "Phase 10" in config.PHASE_LABEL
    assert "Phase 10" in config.WINDOW_TITLE
    assert "astar" in config.PLANNER_ALGORITHMS
    assert "dijkstra" in config.PLANNER_ALGORITHMS
    assert "gbfs" in config.PLANNER_ALGORITHMS
    assert config.DEMO_STAGE_DURATION > 0.0


# ---------------------------------------------------------------------------
# Test 2: Demo Controller Lifecycle
# ---------------------------------------------------------------------------
def test_demo_controller_lifecycle():
    """Verify DemoController initializes, steps forward/back, and toggles."""
    ctrl = DemoController()
    assert ctrl.is_active is False
    assert ctrl.current_stage_idx == 0

    class DummySim:
        def __init__(self):
            self.grid = OccupancyGrid()
            self.planner = Planner()
            self.vehicle = Vehicle()
            self.path_result = None
            self.supervisor = SafetySupervisor()
            self.predictor = Predictor()
            self.ground_truth_objects = []
            self.obstacle_registry = {}
            self.in_benchmark_screen = False
            self.active_scenario = None

        def load_scenario(self, sc):
            self.active_scenario = sc

        def plan_path(self):
            self.path_result = self.planner.plan(self.grid)

        def execute_benchmark_suite(self):
            pass

        def export_benchmark_reports(self):
            pass

    sim = DummySim()

    # Start demo
    ctrl.start_demo(sim)
    assert ctrl.is_active is True
    assert ctrl.current_stage_idx == 0

    # Advance stages
    ctrl.next_stage(sim)
    assert ctrl.current_stage_idx == 1
    ctrl.next_stage(sim)
    assert ctrl.current_stage_idx == 2

    # Step backward
    ctrl.prev_stage(sim)
    assert ctrl.current_stage_idx == 1

    # Stop demo
    ctrl.stop_demo(sim)
    assert ctrl.is_active is False

    # Toggle
    ctrl.toggle(sim)
    assert ctrl.is_active is True
    ctrl.toggle(sim)
    assert ctrl.is_active is False


# ---------------------------------------------------------------------------
# Test 3: Demonstration Workflow Stage Structure
# ---------------------------------------------------------------------------
def test_demonstration_workflow_stages_metadata():
    """Verify that all 7 required demonstration stages are defined in order."""
    assert len(STAGE_METADATA) == 7

    stages = [m["stage"] for m in STAGE_METADATA]
    expected_sequence = [
        DemoStage.SCENARIO_AND_PLAN,
        DemoStage.VEHICLE_NAVIGATION,
        DemoStage.STATIC_OBSTACLE_REPLAN,
        DemoStage.DYNAMIC_TRACKING_PREDICTION,
        DemoStage.COLLISION_AVOIDANCE,
        DemoStage.CATTLE_CROSSING,
        DemoStage.BENCHMARKS_EXPORT,
    ]
    assert stages == expected_sequence

    for m in STAGE_METADATA:
        assert len(m["title"]) > 10
        assert len(m["description"]) > 20
        assert m["duration"] >= 4.0


# ---------------------------------------------------------------------------
# Test 4: Execution of Demo Stages on Live Simulation
# ---------------------------------------------------------------------------
def test_demo_stages_execution_on_simulation():
    """Verify demo controller correctly executes setup actions for stages."""
    pygame.init()
    sim = Simulation()

    ctrl = DemoController()

    # Stage 1: Load scenario and plan A* route
    ctrl._enter_stage(0, sim)
    assert sim.active_scenario is not None
    assert sim.active_scenario.id == "village_road"
    assert sim.path_result is not None
    assert sim.path_result.success is True

    # Stage 2: Vehicle navigation begins
    ctrl._enter_stage(1, sim)
    assert sim.vehicle.state in (VehicleState.MOVING, VehicleState.IDLE)
    assert len(sim.vehicle.waypoints) > 0

    # Stage 3: Static obstacle placed on route
    ctrl._enter_stage(2, sim)
    mid_c, mid_r = 10, config.ROAD_ROW_CENTER
    assert sim.grid.is_occupied(mid_c, mid_r) is True

    # Stage 6: Sudden cattle crossing scenario loaded
    ctrl._enter_stage(5, sim)
    assert sim.active_scenario.id == "cattle_crossing"
    assert any(o.category == ObstacleType.CATTLE for o in sim.ground_truth_objects)


# ---------------------------------------------------------------------------
# Test 5: Interactive Multi-Algorithm Planner Cycling
# ---------------------------------------------------------------------------
def test_planner_algorithm_cycling():
    """Verify cycling through A*, Dijkstra, and GBFS planners at runtime."""
    pygame.init()
    sim = Simulation()

    initial_algo = sim.planner.algorithm
    assert initial_algo in config.PLANNER_ALGORITHMS

    # Cycle 1
    sim.cycle_planner_algorithm()
    algo_1 = sim.planner.algorithm
    assert algo_1 != initial_algo

    # Cycle 2
    sim.cycle_planner_algorithm()
    algo_2 = sim.planner.algorithm
    assert algo_2 not in (initial_algo, algo_1)

    # Cycle 3: returns to initial
    sim.cycle_planner_algorithm()
    assert sim.planner.algorithm == initial_algo


# ---------------------------------------------------------------------------
# Test 6: End-to-End Pipeline Integration
# ---------------------------------------------------------------------------
def test_end_to_end_pipeline_integration():
    """
    Verify complete integration:
    Grid -> Planner -> Vehicle -> Perception -> Prediction -> Safety Supervisor -> Replanner.
    """
    grid = OccupancyGrid()
    planner = Planner(algorithm="astar")
    path_res = planner.plan(grid)
    assert path_res.success is True

    veh = Vehicle()
    veh.set_path(path_res.path)
    veh.start()
    assert veh.state == VehicleState.MOVING

    # Ground truth road users
    gt_obs = [
        DynamicObstacle(
            category=ObstacleType.MOTORCYCLE,
            x=veh.x + 200.0,
            y=config.ROAD_CENTER_Y,
            heading=math.pi,
            speed=25.0,
        )
    ]

    perception = PerceptionSystem()
    predictor = Predictor(horizon=1.5)
    supervisor = SafetySupervisor()

    # Step simulation
    for step in range(20):
        t = step * 0.05
        for o in gt_obs:
            o.update(0.05)
        perceived = perception.update(t, veh.x, veh.y, gt_obs)
        trajectories = predictor.predict_all(perceived)
        state = supervisor.update(t, 0.05, veh, grid, planner, perceived, trajectories)
        veh.speed_multiplier = supervisor.speed_multiplier
        veh.update(0.05, grid)

    assert len(supervisor.event_log) > 0
    assert veh.x > 80.0  # Vehicle moved forward successfully


# ---------------------------------------------------------------------------
# Test 7: UI Elements and Demonstration Banner Geometry
# ---------------------------------------------------------------------------
def test_ui_geometry_and_demo_banner():
    """Verify simulation viewport, status bar, and demo banner dimensions."""
    assert config.SCREEN_WIDTH == 1280
    assert config.SCREEN_HEIGHT == 720
    assert config.SIM_AREA_WIDTH == 960
    assert config.PANEL_WIDTH == 320

    ctrl = DemoController()
    ctrl.is_active = True

    surface = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
    font_title = pygame.font.SysFont("segoeui", 20)
    font_body = pygame.font.SysFont("segoeui", 15)
    font_small = pygame.font.SysFont("segoeui", 12)

    # Render banner - must not raise exception
    ctrl.draw_banner(surface, font_title, font_body, font_small)


# ---------------------------------------------------------------------------
# Test 8: State Isolation and Environment Reset
# ---------------------------------------------------------------------------
def test_state_isolation_and_environment_reset():
    """Verify reset_environment restores pristine state without memory leaks."""
    pygame.init()
    sim = Simulation()

    # Add obstacles
    sim.grid.add_obstacle(5, 5)
    sim.obstacle_registry[(5, 5)] = ObstacleType.CAR
    sim.vehicle.speed = 40.0

    # Reset
    sim.reset_environment()

    assert sim.grid.is_occupied(5, 5) is False
    assert len(sim.obstacle_registry) == 0
    assert sim.vehicle.speed == 0.0
    assert sim.supervisor.state == SafetyState.SAFE
