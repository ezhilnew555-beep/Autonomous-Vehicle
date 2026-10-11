"""
src/simulation.py
=================
Top-level Simulation class coordinating:
- Pygame main loop with fixed or delta-time physics updates
- Autonomous vehicle dynamics & path following
- Simulated Road Users / DynamicObstacles ground truth list
- Simulated Perception System (detection range & tracking)
- Motion Prediction System (Constant-Velocity Model & Expanding Uncertainty)
- OccupancyGrid state & obstacle interactions
- A* Path Planning (Plan, Recalculate, Clear)
- Comprehensive UI controls and Prediction dashboard
"""

import math
import sys
from typing import Dict, Tuple, Optional, List
import pygame

import config
from src.grid import OccupancyGrid
from src.obstacles import (
    ObstacleType,
    DynamicObstacle,
    OBJECT_DIMENSIONS,
    draw_all_obstacles,
    draw_ground_truth_object,
)
from src.perception import (
    PerceptionSystem,
    draw_perception_frustum,
    draw_perceived_tracks,
)
from src.prediction import (
    Predictor,
    PredictedTrajectory,
    draw_predicted_trajectories,
)
from src.planner import Planner, PathResult
from src.road import draw_road
from src.vehicle import Vehicle, VehicleState, draw_vehicle
from src.collision import (
    SafetySupervisor,
    SafetyState,
    draw_collision_zones,
    draw_unsafe_path_highlight,
)
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
    load_default_scenario,
    load_safe_route_scenario,
    load_blocked_route_scenario,
    load_crossing_obstacle_scenario,
    load_no_safe_route_scenario,
    load_recovery_scenario,
)
from src.metrics import ScenarioMetrics
from src.benchmarks import BenchmarkSuite
from src.demo import DemoController
from src.ui import (
    Button,
    draw_status_bar,
    draw_grid,
    draw_markers,
    draw_cell_highlight,
    draw_planned_path,
    draw_side_panel,
    draw_scenario_menu,
    get_scenario_menu_item_rects,
    get_scenario_menu_card_rect,
    draw_benchmark_screen,
    get_benchmark_screen_card_rect,
)


SPAWN_CATEGORIES = [
    ObstacleType.CAR,
    ObstacleType.MOTORCYCLE,
    ObstacleType.AUTORICKSHAW,
    ObstacleType.PEDESTRIAN,
    ObstacleType.CATTLE,
    ObstacleType.BARRIER,
]


class Simulation:
    """Phase 8 interactive simulation with Indian road scenarios, motion prediction, and safety supervisor."""

    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((config.SCREEN_WIDTH, config.SCREEN_HEIGHT))
        pygame.display.set_caption(config.WINDOW_TITLE)
        self.clock = pygame.time.Clock()
        self.running = True

        # Environment & Planning State
        self.grid = OccupancyGrid()
        self.planner = Planner(allow_diagonal=True)
        self.path_result: Optional[PathResult] = None
        self.is_path_outdated: bool = False

        self.obstacle_registry: Dict[Tuple[int, int], ObstacleType] = {}
        self.current_mode = config.MODE_ADD_OBSTACLE
        self.selected_type_idx = 0
        self.selected_obs_type = SPAWN_CATEGORIES[self.selected_type_idx]
        self.show_grid = True

        # Autonomous Vehicle State (Phase 4)
        self.vehicle = Vehicle()
        self.sim_speed_idx = 0
        self.sim_speed = config.SIM_SPEED_OPTIONS[self.sim_speed_idx]

        # Simulated Perception System (Phase 5)
        self.range_idx = 1
        self.delay_idx = 1
        self.perception = PerceptionSystem(
            detection_range=config.DETECTION_RANGE_OPTIONS[self.range_idx],
            detection_delay=config.DETECTION_DELAY_OPTIONS[self.delay_idx],
        )

        # Motion Prediction System (Phase 6)
        self.horizon_idx = config.DEFAULT_HORIZON_INDEX
        self.predictor = Predictor(horizon=config.PREDICTION_HORIZONS[self.horizon_idx])
        self.predicted_trajectories: List[PredictedTrajectory] = []

        # Collision Avoidance & Safety Supervisor (Phase 7)
        self.supervisor = SafetySupervisor()

        # Phase 8: Scenarios & Menu State
        self.all_scenarios = get_all_scenarios()
        self.active_scenario: Optional[Scenario] = None
        self.scenario_state: str = "IDLE"
        self.in_scenario_menu: bool = False
        self.selected_menu_idx: int = 0
        self.metrics = ScenarioMetrics()

        # Phase 9: Benchmark Suite & Screen State
        self.benchmark_suite = BenchmarkSuite()
        self.in_benchmark_screen: bool = False
        self.benchmark_status_msg: str = ""

        # Phase 10: Multi-Planner Selection & Demonstration Engine
        self.algo_idx = 0
        self.planner.algorithm = config.PLANNER_ALGORITHMS[self.algo_idx]
        self.demo_controller = DemoController()

        # Ground Truth Dynamic Road Users
        self.ground_truth_objects: List[DynamicObstacle] = []
        self._init_default_road_users()

        # Fonts
        self.font_title = pygame.font.SysFont("segoeui", config.FONT_SIZE_TITLE)
        self.font_body  = pygame.font.SysFont("segoeui", config.FONT_SIZE_BODY)
        self.font_small = pygame.font.SysFont("segoeui", config.FONT_SIZE_SMALL)

        # UI Controls / Buttons
        self.buttons = self._init_buttons()

    def _init_default_road_users(self) -> None:
        """Seed initial road users."""
        self.ground_truth_objects.append(
            DynamicObstacle(
                category=ObstacleType.PEDESTRIAN,
                x=350.0,
                y=config.ROAD_CENTER_Y - 30.0,
                heading=math.pi / 2.0,
                speed=12.0,
            )
        )
        self.ground_truth_objects.append(
            DynamicObstacle(
                category=ObstacleType.AUTORICKSHAW,
                x=550.0,
                y=config.ROAD_CENTER_Y + 20.0,
                heading=0.0,
                speed=25.0,
                min_x=300.0,
                max_x=750.0,
            )
        )
        self.ground_truth_objects.append(
            DynamicObstacle(
                category=ObstacleType.CATTLE,
                x=700.0,
                y=config.ROAD_BOTTOM_Y - 25.0,
                heading=-0.2,
                speed=0.0,
            )
        )

    # ------------------------------------------------------------------ #
    # Button Setup
    # ------------------------------------------------------------------ #
    def _init_buttons(self) -> list:
        buttons = []
        panel_x = config.SIM_AREA_WIDTH + 14
        start_y = config.SIM_AREA_TOP + 24
        w = config.PANEL_WIDTH - 28
        h = 20
        gap = 3

        # Row 0: Start / Pause Vehicle
        btn_start_pause = Button(pygame.Rect(panel_x, start_y, w, h), "Start Simulation", self.toggle_start_pause)
        buttons.append(btn_start_pause)

        # Row 1: Reset Vehicle
        btn_reset_veh = Button(pygame.Rect(panel_x, start_y + (h + gap), w, h), "Reset Vehicle", self.reset_vehicle)
        buttons.append(btn_reset_veh)

        # Row 2: Sim Speed
        btn_speed = Button(pygame.Rect(panel_x, start_y + (h + gap) * 2, w, h), f"Sim Speed: {self.sim_speed:.1f}x", self.cycle_sim_speed)
        buttons.append(btn_speed)

        # Row 3: Plan Path
        btn_plan = Button(pygame.Rect(panel_x, start_y + (h + gap) * 3, w, h), "Plan Path (A*)", self.plan_path)
        buttons.append(btn_plan)

        # Row 4: Prediction Horizon (Phase 6)
        btn_horizon = Button(pygame.Rect(panel_x, start_y + (h + gap) * 4, w, h), f"Horizon: {self.predictor.horizon:.1f}s", self.cycle_horizon)
        buttons.append(btn_horizon)

        # Row 5: Toggle Uncertainty (Phase 6)
        btn_unc = Button(pygame.Rect(panel_x, start_y + (h + gap) * 5, w, h), "Toggle Uncertainty", self.toggle_uncertainty_view)
        buttons.append(btn_unc)

        # Row 6: Spawn Obstacle
        btn_spawn = Button(pygame.Rect(panel_x, start_y + (h + gap) * 6, w, h), f"Spawn {self.selected_obs_type.value}", self.spawn_selected_object)
        buttons.append(btn_spawn)

        # Row 7: Cycle Type
        btn_type = Button(pygame.Rect(panel_x, start_y + (h + gap) * 7, w, h), f"Type: {self.selected_obs_type.value}", self.cycle_obstacle_type)
        buttons.append(btn_type)

        # Row 8: Cycle Range
        btn_range = Button(pygame.Rect(panel_x, start_y + (h + gap) * 8, w, h), f"Range: {int(self.perception.detection_range)}px", self.cycle_detection_range)
        buttons.append(btn_range)

        half_w = (w - 4) // 2
        # Row 9: Reset Environment & Restart Scenario (Phase 8)
        btn_reset_env = Button(pygame.Rect(panel_x, start_y + (h + gap) * 9, half_w, h), "Reset Env", self.reset_environment)
        btn_restart_scen = Button(pygame.Rect(panel_x + half_w + 4, start_y + (h + gap) * 9, half_w, h), "Restart (F6)", self.restart_current_scenario)
        buttons.append(btn_reset_env)
        buttons.append(btn_restart_scen)

        # Row 10: Toggle Grid & Scenarios Menu (Phase 8)
        btn_grid = Button(pygame.Rect(panel_x, start_y + (h + gap) * 10, half_w, h), "Grid", self.toggle_grid)
        btn_menu = Button(pygame.Rect(panel_x + half_w + 4, start_y + (h + gap) * 10, half_w, h), "Scenarios (M)", self.toggle_scenario_menu)
        buttons.append(btn_grid)
        buttons.append(btn_menu)

        # Row 11: Benchmark & Demo Mode (Phase 10)
        btn_bench = Button(pygame.Rect(panel_x, start_y + (h + gap) * 11, half_w, h), "Benchmarks (B)", self.toggle_benchmark_screen)
        btn_demo = Button(pygame.Rect(panel_x + half_w + 4, start_y + (h + gap) * 11, half_w, h), "Demo (D)", self.toggle_demo_mode)
        buttons.append(btn_bench)
        buttons.append(btn_demo)

        # Row 12: Planner Algorithm Selector (Phase 10)
        btn_algo = Button(pygame.Rect(panel_x, start_y + (h + gap) * 12, w, h), f"Planner: {self.planner.algorithm.upper()} (A)", self.cycle_planner_algorithm)
        buttons.append(btn_algo)

        return buttons

    # ------------------------------------------------------------------ #
    # Control Actions
    # ------------------------------------------------------------------ #
    def toggle_start_pause(self) -> None:
        if self.vehicle.state == VehicleState.MOVING:
            self.vehicle.pause()
            self.buttons[0].label = "Resume"
            self.buttons[0].is_active = False
        elif self.vehicle.state == VehicleState.PAUSED:
            self.vehicle.resume()
            self.buttons[0].label = "Pause"
            self.buttons[0].is_active = True
        else:
            if not self.path_result or not self.path_result.success:
                self.plan_path()
            if self.path_result and self.path_result.success:
                self.vehicle.set_path(self.path_result.path)
                if self.vehicle.start():
                    self.buttons[0].label = "Pause"
                    self.buttons[0].is_active = True

    def reset_vehicle(self) -> None:
        self.vehicle.reset()
        if self.path_result and self.path_result.success:
            self.vehicle.set_path(self.path_result.path)
        self.buttons[0].label = "Start Simulation"
        self.buttons[0].is_active = False

    def cycle_sim_speed(self) -> None:
        self.sim_speed_idx = (self.sim_speed_idx + 1) % len(config.SIM_SPEED_OPTIONS)
        self.sim_speed = config.SIM_SPEED_OPTIONS[self.sim_speed_idx]
        self.buttons[2].label = f"Sim Speed: {self.sim_speed:.1f}x"

    def cycle_horizon(self) -> None:
        """Cycle prediction horizons: 0.5s, 1.0s, 1.5s, 2.0s."""
        self.horizon_idx = (self.horizon_idx + 1) % len(config.PREDICTION_HORIZONS)
        new_h = config.PREDICTION_HORIZONS[self.horizon_idx]
        self.predictor.set_horizon(new_h)
        self.buttons[4].label = f"Horizon: {new_h:.1f}s"

    def toggle_uncertainty_view(self) -> None:
        self.predictor.show_uncertainty = not self.predictor.show_uncertainty

    def cycle_obstacle_type(self) -> None:
        self.selected_type_idx = (self.selected_type_idx + 1) % len(SPAWN_CATEGORIES)
        self.selected_obs_type = SPAWN_CATEGORIES[self.selected_type_idx]
        self.buttons[6].label = f"Spawn {self.selected_obs_type.value}"
        self.buttons[7].label = f"Type: {self.selected_obs_type.value}"

    def spawn_selected_object(self, world_x: Optional[float] = None, world_y: Optional[float] = None) -> None:
        if world_x is None or world_y is None:
            world_x = 450.0 + (len(self.ground_truth_objects) * 40.0) % 300.0
            world_y = float(config.ROAD_CENTER_Y)

        _, _, def_speed = OBJECT_DIMENSIONS.get(self.selected_obs_type, (24.0, 14.0, 0.0))
        speed = def_speed if self.selected_obs_type != ObstacleType.BARRIER else 0.0

        new_obj = DynamicObstacle(
            category=self.selected_obs_type,
            x=world_x,
            y=world_y,
            heading=0.0,
            speed=speed,
            min_x=120.0,
            max_x=850.0,
        )
        self.ground_truth_objects.append(new_obj)

    def cycle_detection_range(self) -> None:
        self.range_idx = (self.range_idx + 1) % len(config.DETECTION_RANGE_OPTIONS)
        new_range = config.DETECTION_RANGE_OPTIONS[self.range_idx]
        self.perception.detection_range = new_range
        self.buttons[8].label = f"Range: {int(new_range)}px"

    def plan_path(self) -> None:
        self.path_result = self.planner.plan(self.grid)
        self.is_path_outdated = False
        if self.path_result.success:
            if self.vehicle.state in (VehicleState.IDLE, VehicleState.BLOCKED, VehicleState.ARRIVED):
                self.vehicle.set_path(self.path_result.path)

    def clear_path(self) -> None:
        self.planner.clear()
        self.path_result = None
        self.is_path_outdated = False
        self.reset_vehicle()

    def reset_environment(self) -> None:
        self.grid.reset()
        self.grid.reset_road_mask()
        self.obstacle_registry.clear()
        self.planner.clear()
        self.path_result = None
        self.is_path_outdated = False
        self.vehicle.start_col = config.START_GRID_COL
        self.vehicle.start_row = config.START_GRID_ROW
        self.vehicle.reset()
        self.vehicle.set_path([])
        self.perception.clear()
        self.predictor.clear()
        self.supervisor.reset()
        self.ground_truth_objects.clear()
        self._init_default_road_users()
        self.active_scenario = None
        self.scenario_state = "IDLE"
        self.metrics = ScenarioMetrics()
        self.buttons[0].label = "Start Simulation"
        self.buttons[0].is_active = False

    def toggle_scenario_menu(self) -> None:
        """Toggle scenario selection modal dialog."""
        self.in_scenario_menu = not self.in_scenario_menu

    def restart_current_scenario(self) -> None:
        """Restart the active scenario or reset environment if none."""
        if self.active_scenario is not None:
            fresh = get_scenario_by_id(self.active_scenario.id, seed=self.active_scenario.seed)
            if fresh is not None:
                self.load_scenario(fresh)
                return
            else:
                self.load_scenario(self.active_scenario)
                return
        self.reset_environment()

    def toggle_benchmark_screen(self) -> None:
        """Toggle Phase 9 benchmark evaluation modal screen."""
        self.in_benchmark_screen = not self.in_benchmark_screen
        if self.in_benchmark_screen:
            self.in_scenario_menu = False

    def execute_benchmark_suite(self) -> None:
        """Execute the benchmark suite (static algorithms & dynamic simulations)."""
        self.benchmark_status_msg = "Running benchmark suite across all scenarios..."
        self.screen.fill(config.C_BACKGROUND)
        draw_benchmark_screen(
            self.screen,
            self.benchmark_suite,
            self.font_title,
            self.font_body,
            self.font_small,
            status_msg="Running benchmark suite across all scenarios...",
        )
        pygame.display.flip()

        runs = self.benchmark_suite.run_benchmarks(
            include_static=True,
            include_dynamic=True,
            repetitions=config.BENCHMARK_DEFAULT_REPS,
            seed_base=config.BENCHMARK_DEFAULT_SEED,
        )
        self.benchmark_status_msg = f"Completed {len(runs)} benchmark runs! Press [E] to export CSV & charts."

    def export_benchmark_reports(self) -> None:
        """Export CSV data, JSON metadata, and performance charts."""
        exported = self.benchmark_suite.export_all()
        chart_count = sum(1 for k in exported if k.startswith("chart_"))
        self.benchmark_status_msg = f"Exported {len(exported)} files ({chart_count} charts) to reports/"

    def cycle_planner_algorithm(self) -> None:
        """Cycle between A*, Dijkstra, and GBFS planners (Phase 10)."""
        self.algo_idx = (self.algo_idx + 1) % len(config.PLANNER_ALGORITHMS)
        new_algo = config.PLANNER_ALGORITHMS[self.algo_idx]
        self.planner.algorithm = new_algo
        if len(self.buttons) > 15:
            self.buttons[15].label = f"Planner: {new_algo.upper()} (A)"
        if self.path_result is not None:
            self.plan_path()

    def toggle_demo_mode(self) -> None:
        """Toggle Phase 10 demonstration walkthrough mode."""
        self.demo_controller.toggle(self)
        if len(self.buttons) > 14:
            self.buttons[14].is_active = self.demo_controller.is_active
            self.buttons[14].label = "Exit Demo (D)" if self.demo_controller.is_active else "Demo (D)"

    def _execute_scheduled_event(self, ev: ScheduledEvent) -> None:
        """Execute a scheduled dynamic event (e.g. cattle crossing or vehicle movement)."""
        if ev.action_type == "ACTIVATE_MOVEMENT":
            target_id = ev.payload.get("obstacle_id")
            for obj in self.ground_truth_objects:
                if target_id is None or obj.id == target_id or (target_id == 77 and obj.category == ObstacleType.CATTLE):
                    if "speed" in ev.payload:
                        obj.speed = ev.payload["speed"]
                    if "heading" in ev.payload:
                        obj.heading = ev.payload["heading"]
                    if "bound_y" in ev.payload:
                        obj.bound_y = ev.payload["bound_y"]
                    break
        elif ev.action_type == "SPAWN_OBSTACLE":
            if "obstacle" in ev.payload:
                self.ground_truth_objects.append(ev.payload["obstacle"])

        self.supervisor.log_event(
            self.vehicle.elapsed_time,
            self.supervisor.state,
            "SCHEDULED_EVENT",
            ev.description or f"Triggered {ev.action_type}",
        )

    def load_scenario(self, scenario_or_loader) -> None:
        """Load and activate a Phase 7 or Phase 8 scenario."""
        if callable(scenario_or_loader):
            scenario = scenario_or_loader()
        else:
            scenario = scenario_or_loader

        self.reset_environment()
        self.ground_truth_objects.clear()

        # Set road mask
        if scenario.road_mask is not None:
            self.grid.set_road_mask(scenario.road_mask)
        else:
            self.grid.reset_road_mask()

        # Set scenario start and destination
        self.grid.start_pos = scenario.start_pos
        self.grid.dest_pos = scenario.dest_pos

        self.vehicle.start_col = scenario.start_pos[0]
        self.vehicle.start_row = scenario.start_pos[1]
        self.vehicle.reset()

        # Add static obstacles
        for col, row in scenario.obstacle_positions:
            if self.grid.add_obstacle(col, row):
                self.obstacle_registry[(col, row)] = ObstacleType.BARRIER

        # Add dynamic obstacles (make copy so restart is fresh)
        import copy
        self.ground_truth_objects = [copy.deepcopy(obs) for obs in scenario.dynamic_obstacles]

        # Reset events
        scenario.reset_events()

        self.active_scenario = scenario
        self.scenario_state = "RUNNING"
        self.metrics = ScenarioMetrics(
            scenario_name=scenario.name,
            random_seed=scenario.seed,
            status="RUNNING",
        )
        self.in_scenario_menu = False

        # Plan initial route
        self.plan_path()
        if self.path_result and self.path_result.success:
            self.vehicle.set_path(self.path_result.path)
            self.vehicle.start()
            self.metrics.path_length = self.path_result.path_cost * config.GRID_CELL_SIZE
            self.metrics.planning_time_ms = self.path_result.execution_time_ms
            self.buttons[0].label = "Pause"
            self.buttons[0].is_active = True
        else:
            self.supervisor.transition_to(
                SafetyState.NO_SAFE_PATH,
                "No initial route for scenario",
                self.vehicle.elapsed_time,
            )
            self.scenario_state = "NO_SAFE_PATH"
            self.metrics.status = "NO_SAFE_PATH"
            self.metrics.termination_reason = "NO_INITIAL_PATH"

        self.supervisor.log_event(
            self.vehicle.elapsed_time,
            SafetyState.SAFE,
            "SCENARIO_LOADED",
            f"Scenario loaded: {scenario.name}",
        )

    def toggle_grid(self) -> None:
        self.show_grid = not self.show_grid

    # ------------------------------------------------------------------ #
    # Coordinates Conversion
    # ------------------------------------------------------------------ #
    def pixel_to_grid(self, px: int, py: int) -> Optional[Tuple[int, int]]:
        sim_x, sim_y, sim_w, sim_h = config.SIM_RECT
        if not (sim_x <= px < sim_x + sim_w and sim_y <= py < sim_y + sim_h):
            return None
        col = (px - sim_x) // config.GRID_CELL_SIZE
        row = (py - sim_y) // config.GRID_CELL_SIZE
        return int(col), int(row)

    # ------------------------------------------------------------------ #
    # Event Handling
    # ------------------------------------------------------------------ #
    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            # When Benchmark Screen overlay is active (Phase 9)
            if self.in_benchmark_screen:
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_b):
                        self.in_benchmark_screen = False
                        return
                    elif event.key in (pygame.K_RETURN, pygame.K_r):
                        self.execute_benchmark_suite()
                        return
                    elif event.key == pygame.K_e:
                        self.export_benchmark_reports()
                        return

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    card_rect = get_benchmark_screen_card_rect()
                    if not card_rect.collidepoint(event.pos):
                        self.in_benchmark_screen = False
                        return
                continue

            # When Scenario Menu overlay is active
            if self.in_scenario_menu:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.in_scenario_menu = False
                        return
                    elif event.key == pygame.K_UP:
                        self.selected_menu_idx = (self.selected_menu_idx - 1) % len(self.all_scenarios)
                        return
                    elif event.key == pygame.K_DOWN:
                        self.selected_menu_idx = (self.selected_menu_idx + 1) % len(self.all_scenarios)
                        return
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        self.load_scenario(self.all_scenarios[self.selected_menu_idx])
                        return
                    elif event.key in (pygame.K_F1, pygame.K_1):
                        self.selected_menu_idx = 0
                        self.load_scenario(self.all_scenarios[0])
                        return
                    elif event.key in (pygame.K_F2, pygame.K_2):
                        self.selected_menu_idx = 1
                        self.load_scenario(self.all_scenarios[1])
                        return
                    elif event.key in (pygame.K_F3, pygame.K_3):
                        self.selected_menu_idx = 2
                        self.load_scenario(self.all_scenarios[2])
                        return
                    elif event.key in (pygame.K_F4, pygame.K_4):
                        self.selected_menu_idx = 3
                        self.load_scenario(self.all_scenarios[3])
                        return
                    elif event.key in (pygame.K_F5, pygame.K_5):
                        self.selected_menu_idx = 4
                        self.load_scenario(self.all_scenarios[4])
                        return
                    elif event.key in (pygame.K_F6, pygame.K_r):
                        self.restart_current_scenario()
                        self.in_scenario_menu = False
                        return

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    card_rect = get_scenario_menu_card_rect()
                    if not card_rect.collidepoint(event.pos):
                        self.in_scenario_menu = False
                        return
                    item_rects = get_scenario_menu_item_rects(len(self.all_scenarios))
                    for idx, irect in enumerate(item_rects):
                        if irect.collidepoint(event.pos):
                            self.selected_menu_idx = idx
                            self.load_scenario(self.all_scenarios[idx])
                            return
                continue

            # When Demonstration Mode is active (Phase 10)
            if self.demo_controller.is_active:
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_d):
                        self.toggle_demo_mode()
                        return
                    elif event.key in (pygame.K_SPACE, pygame.K_RIGHT):
                        self.demo_controller.next_stage(self)
                        return
                    elif event.key in (pygame.K_BACKSPACE, pygame.K_LEFT):
                        self.demo_controller.prev_stage(self)
                        return
                    elif event.key == pygame.K_r:
                        self.demo_controller.start_demo(self)
                        return

            # Standard Simulation Mode Controls
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return
                elif event.key == pygame.K_d:
                    self.toggle_demo_mode()
                elif event.key == pygame.K_a:
                    self.cycle_planner_algorithm()
                elif event.key == pygame.K_b:
                    self.toggle_benchmark_screen()
                elif event.key == pygame.K_m:
                    self.toggle_scenario_menu()
                elif event.key == pygame.K_SPACE:
                    self.toggle_start_pause()
                elif event.key == pygame.K_x:
                    self.reset_vehicle()
                elif event.key == pygame.K_p:
                    self.plan_path()
                elif event.key == pygame.K_k:
                    self.clear_path()
                elif event.key == pygame.K_h:
                    self.cycle_horizon()
                elif event.key == pygame.K_u:
                    self.toggle_uncertainty_view()
                elif event.key == pygame.K_s:
                    mouse_pos = pygame.mouse.get_pos()
                    if mouse_pos[0] < config.SIM_AREA_WIDTH and mouse_pos[1] >= config.SIM_AREA_TOP:
                        self.spawn_selected_object(float(mouse_pos[0]), float(mouse_pos[1]))
                    else:
                        self.spawn_selected_object()
                elif event.key == pygame.K_g:
                    self.toggle_grid()
                elif event.key == pygame.K_r:
                    if self.active_scenario is not None:
                        self.restart_current_scenario()
                    else:
                        self.reset_environment()
                elif event.key == pygame.K_F1:
                    self.selected_menu_idx = 0
                    self.load_scenario(self.all_scenarios[0])
                elif event.key == pygame.K_F2:
                    self.selected_menu_idx = 1
                    self.load_scenario(self.all_scenarios[1])
                elif event.key == pygame.K_F3:
                    self.selected_menu_idx = 2
                    self.load_scenario(self.all_scenarios[2])
                elif event.key == pygame.K_F4:
                    self.selected_menu_idx = 3
                    self.load_scenario(self.all_scenarios[3])
                elif event.key == pygame.K_F5:
                    self.selected_menu_idx = 4
                    self.load_scenario(self.all_scenarios[4])
                elif event.key == pygame.K_F6:
                    self.restart_current_scenario()
                elif pygame.K_1 <= event.key <= pygame.K_6:
                    self.selected_type_idx = event.key - pygame.K_1
                    self.selected_obs_type = SPAWN_CATEGORIES[self.selected_type_idx]
                    self.buttons[6].label = f"Spawn {self.selected_obs_type.value}"
                    self.buttons[7].label = f"Type: {self.selected_obs_type.value}"

            # Check buttons
            button_handled = False
            for btn in self.buttons:
                if btn.handle_event(event):
                    button_handled = True
                    break

            if button_handled:
                continue

            # Mouse click in simulation canvas
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if event.pos[0] < config.SIM_AREA_WIDTH and config.ROAD_TOP_Y <= event.pos[1] <= config.ROAD_BOTTOM_Y:
                        self.spawn_selected_object(float(event.pos[0]), float(event.pos[1]))
                    else:
                        grid_pos = self.pixel_to_grid(*event.pos)
                        if grid_pos is not None:
                            col, row = grid_pos
                            if self.grid.add_obstacle(col, row):
                                self.obstacle_registry[(col, row)] = self.selected_obs_type
                                if self.path_result is not None:
                                    self.is_path_outdated = True

                elif event.button == 3:
                    mx, my = event.pos
                    removed = False
                    for idx, obj in enumerate(self.ground_truth_objects):
                        if math.hypot(obj.x - mx, obj.y - my) < 20.0:
                            self.ground_truth_objects.pop(idx)
                            removed = True
                            break
                    if not removed:
                        grid_pos = self.pixel_to_grid(*event.pos)
                        if grid_pos is not None:
                            col, row = grid_pos
                            if self.grid.remove_obstacle(col, row):
                                self.obstacle_registry.pop((col, row), None)
                                if self.path_result is not None:
                                    self.is_path_outdated = True

    # ------------------------------------------------------------------ #
    # Update Physics, Perception, Motion Prediction & Safety Supervisor
    # ------------------------------------------------------------------ #
    def _update(self, dt: float) -> None:
        scaled_dt = dt * self.sim_speed
        sim_time = self.vehicle.elapsed_time

        # Phase 10: Demonstration Walkthrough Controller Update
        if self.demo_controller.is_active:
            self.demo_controller.update(scaled_dt, self)

        # 0. Phase 8 Scheduled Events execution
        if self.active_scenario is not None and self.scenario_state == "RUNNING":
            due_events = self.active_scenario.get_due_events(sim_time)
            for ev in due_events:
                self._execute_scheduled_event(ev)
                ev.has_triggered = True

        # 1. Dynamic obstacles physics
        for obj in self.ground_truth_objects:
            obj.update(scaled_dt)

        # 2. Perception system (converts ground truth -> perceived tracks)
        perceived = self.perception.update(
            sim_time=sim_time,
            ego_x=self.vehicle.x,
            ego_y=self.vehicle.y,
            ground_truth_objects=self.ground_truth_objects,
        )

        # 3. Motion Prediction (Phase 6): consumes perceived objects ONLY
        self.predicted_trajectories = self.predictor.predict_all(perceived)

        # 4. Collision Avoidance & Safety Supervisor (Phase 7)
        self.supervisor.update(
            sim_time=sim_time,
            dt=scaled_dt,
            vehicle=self.vehicle,
            grid=self.grid,
            planner=self.planner,
            perceived_objects=self.perception.perceived_objects,
            predicted_trajectories=self.predicted_trajectories,
        )
        self.vehicle.speed_multiplier = self.supervisor.speed_multiplier

        # 5. Vehicle kinematics
        self.vehicle.update(scaled_dt, self.grid)

        # 6. Phase 8 Scenario Lifecycle & Metrics Tracking
        if self.active_scenario is not None:
            self.metrics.simulation_duration = self.vehicle.elapsed_time
            self.metrics.replanning_events = self.supervisor.replan_count
            self.metrics.emergency_stops = self.supervisor.emergency_stop_count
            self.metrics.collision_risk_events = len(self.supervisor.conflict_history)

            # Minimum clearance tracking
            for obj in self.ground_truth_objects:
                dist = math.hypot(obj.x - self.vehicle.x, obj.y - self.vehicle.y)
                self.metrics.update_clearance(dist)

            if self.scenario_state == "RUNNING":
                dest_c, dest_r = self.active_scenario.dest_pos
                veh_c, veh_r = self.vehicle.grid_col, self.vehicle.grid_row
                # Arrived at destination
                if self.vehicle.state == VehicleState.ARRIVED or (veh_c == dest_c and veh_r == dest_r):
                    self.scenario_state = "COMPLETED"
                    self.metrics.status = "COMPLETED"
                    self.metrics.destination_reached = True
                    self.metrics.termination_reason = "DESTINATION_REACHED"
                # Scenario timeout
                elif self.vehicle.elapsed_time >= self.active_scenario.timeout:
                    self.scenario_state = "TIMEOUT"
                    self.metrics.status = "TIMEOUT"
                    self.metrics.destination_reached = False
                    self.metrics.termination_reason = f"TIMEOUT ({self.active_scenario.timeout:.0f}s)"
                    self.vehicle.pause()
                # Supervisor blocked with no safe path
                elif self.supervisor.state == SafetyState.NO_SAFE_PATH:
                    self.scenario_state = "NO_SAFE_PATH"
                    self.metrics.status = "NO_SAFE_PATH"
                    self.metrics.destination_reached = False
                    self.metrics.termination_reason = "NO_SAFE_PATH"

        if self.vehicle.state in (VehicleState.ARRIVED, VehicleState.BLOCKED, VehicleState.IDLE):
            self.buttons[0].label = "Start Simulation"
            self.buttons[0].is_active = False

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #
    def _draw(self) -> None:
        # 1. Background
        self.screen.fill(config.C_BACKGROUND)

        # 2. Simulation area background
        sim_rect = pygame.Rect(*config.SIM_RECT)
        pygame.draw.rect(self.screen, config.C_SIM_BG, sim_rect)

        # 3. Road surface (supports distinct Phase 8 geometries)
        road_type = self.active_scenario.road_type if self.active_scenario else "straight"
        draw_road(self.screen, road_type=road_type)

        # 4. Grid overlay
        if self.show_grid:
            draw_grid(self.screen)

        # 5. Static Grid Obstacles
        draw_all_obstacles(self.screen, self.grid, self.obstacle_registry)

        # 6. Planned Path (Phase 3)
        if self.path_result and self.path_result.success:
            draw_planned_path(self.screen, self.path_result.path, self.is_path_outdated)
            # Phase 7: Highlight Unsafe Path Segments
            if self.supervisor.unsafe_waypoint_index is not None:
                draw_unsafe_path_highlight(
                    self.screen,
                    self.vehicle.waypoints,
                    self.supervisor.unsafe_waypoint_index,
                )

        # 7. Start & Destination markers
        draw_markers(self.screen, start_pos=self.grid.start_pos, dest_pos=self.grid.dest_pos)

        # 8. Simulated Perception Sensor Frustum (Phase 5)
        if self.perception.show_detection_range:
            draw_perception_frustum(
                self.screen,
                self.vehicle.x,
                self.vehicle.y,
                self.perception.detection_range,
            )

        # 9. Ground Truth Road Users
        for obj in self.ground_truth_objects:
            draw_ground_truth_object(
                self.screen,
                obj,
                show_label=self.perception.show_labels,
                font=self.font_small,
            )

        # 10. Perceived / Tracked Bounding Boxes & Tags
        draw_perceived_tracks(
            self.screen,
            self.perception.perceived_objects,
            show_labels=self.perception.show_labels,
            font=self.font_small,
        )

        # 11. Motion Prediction Trajectories & Uncertainty Regions (Phase 6)
        if self.predictor.show_trajectories:
            draw_predicted_trajectories(
                self.screen,
                self.predicted_trajectories,
                show_uncertainty=self.predictor.show_uncertainty,
                show_time_labels=self.predictor.show_time_labels,
                font=self.font_small,
            )

        # 11b. Phase 7: Conflict Warning Zones
        draw_collision_zones(self.screen, self.supervisor.active_conflicts, self.font_small)

        # 12. Autonomous Ego Vehicle (Phase 4)
        draw_vehicle(self.screen, self.vehicle)

        # 13. Hover cursor preview (suppressed in menu/benchmark overlay)
        if not self.in_scenario_menu and not self.in_benchmark_screen:
            mouse_pos = pygame.mouse.get_pos()
            grid_pos = self.pixel_to_grid(*mouse_pos)
            if grid_pos is not None:
                col, row = grid_pos
                is_valid_target = self.grid.is_road(col, row) and not self.grid.is_protected(col, row)
                draw_cell_highlight(self.screen, col, row, is_valid_target)

        # 14. Top Status Bar
        draw_status_bar(
            self.screen,
            self.font_title,
            self.current_mode,
            len(self.ground_truth_objects) + self.grid.obstacle_count,
            len(self.perception.perceived_objects),
            self.vehicle.state.value,
            self.sim_speed,
            self.predictor.horizon,
            safety_state=self.supervisor.state.value,
            scenario_name=self.active_scenario.name if self.active_scenario else None,
            scenario_status=self.scenario_state if self.active_scenario else None,
        )

        # 15. Side panel (Controls, Telemetry, Prediction List, Legend, Safety Supervisor)
        draw_side_panel(
            self.screen,
            self.buttons,
            self.selected_obs_type,
            self.vehicle,
            self.path_result,
            self.is_path_outdated,
            self.perception.perceived_objects,
            self.predicted_trajectories,
            self.predictor.horizon,
            self.font_body,
            self.font_small,
            supervisor=self.supervisor,
            scenario_name=self.active_scenario.name if self.active_scenario else None,
            scenario_status=self.scenario_state if self.active_scenario else None,
            scenario_time=self.vehicle.elapsed_time if self.active_scenario else None,
            scenario_timeout=self.active_scenario.timeout if self.active_scenario else None,
            planner_algorithm=self.planner.algorithm,
            demo_active=self.demo_controller.is_active,
        )

        # 16. Phase 10: Demonstration Walkthrough Banner Overlay
        if self.demo_controller.is_active and not self.in_scenario_menu and not self.in_benchmark_screen:
            self.demo_controller.draw_banner(
                self.screen,
                self.font_title,
                self.font_body,
                self.font_small,
            )

        # 17. Modal Scenario Selection Menu (Phase 8)
        if self.in_scenario_menu:
            draw_scenario_menu(
                self.screen,
                self.all_scenarios,
                self.selected_menu_idx,
                self.font_title,
                self.font_body,
                self.font_small,
            )

        # 17. Modal Performance Evaluation Screen (Phase 9)
        if self.in_benchmark_screen:
            draw_benchmark_screen(
                self.screen,
                self.benchmark_suite,
                self.font_title,
                self.font_body,
                self.font_small,
                status_msg=self.benchmark_status_msg,
            )

        pygame.display.flip()

    # ------------------------------------------------------------------ #
    # Main Loop
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        while self.running:
            raw_dt = self.clock.tick(config.FPS) / 1000.0
            dt = min(raw_dt, 0.05)
            self._handle_events()
            self._update(dt)
            self._draw()

        pygame.quit()
        sys.exit(0)
