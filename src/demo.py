"""
src/demo.py
===========
Deterministic Demonstration Workflow Engine for Phase 10.

Coordinates a repeatable, automated or step-by-step walkthrough demonstrating:
1. Scenario loading and baseline A* path planning.
2. Vehicle kinematic cruise and pure pursuit waypoint tracking.
3. Sudden static obstacle placement and automatic dynamic replanning.
4. Perceived object tracking and multi-horizon motion prediction with uncertainty.
5. Collision avoidance response (CAUTION speed reduction and EMERGENCY STOP).
6. Scenario 5 (Sudden Cattle Crossing) reactive avoidance and recovery.
7. Comparative benchmark suite execution and CSV/JSON/chart reports export.

Never bypasses safety checks or fabricates outcomes; drives the live simulation
using actual physics, perception, prediction, and safety evaluation.
"""

from enum import Enum
from typing import List, Optional, Tuple, Dict, Any
import math
import pygame

import config
from src.obstacles import ObstacleType, DynamicObstacle
from src.scenarios import get_scenario_by_id


class DemoStage(Enum):
    SCENARIO_AND_PLAN = 0
    VEHICLE_NAVIGATION = 1
    STATIC_OBSTACLE_REPLAN = 2
    DYNAMIC_TRACKING_PREDICTION = 3
    COLLISION_AVOIDANCE = 4
    CATTLE_CROSSING = 5
    BENCHMARKS_EXPORT = 6


STAGE_METADATA = [
    {
        "stage": DemoStage.SCENARIO_AND_PLAN,
        "title": "Stage 1/7: Scenario Initialization & Global Path Planning",
        "description": "Loads rural village scenario. Computes optimal 8-connected A* trajectory avoiding roadside gravel.",
        "duration": 5.0,
    },
    {
        "stage": DemoStage.VEHICLE_NAVIGATION,
        "title": "Stage 2/7: Autonomous Kinematic Navigation",
        "description": "Autonomous vehicle accelerates along planned path, respecting road curvature and velocity limits.",
        "duration": 6.0,
    },
    {
        "stage": DemoStage.STATIC_OBSTACLE_REPLAN,
        "title": "Stage 3/7: Road Obstruction & Dynamic Replanning",
        "description": "Unexpected roadside obstacle placed on route. Safety supervisor detects conflict and executes A* detour.",
        "duration": 6.5,
    },
    {
        "stage": DemoStage.DYNAMIC_TRACKING_PREDICTION,
        "title": "Stage 4/7: Perception Tracking & Multi-Horizon Motion Prediction",
        "description": "Sensors detect oncoming road users. System forecasts future positions with time-expanding uncertainty.",
        "duration": 6.5,
    },
    {
        "stage": DemoStage.COLLISION_AVOIDANCE,
        "title": "Stage 5/7: Predictive Collision Avoidance & Emergency Braking",
        "description": "Predictive conflict detected. Safety supervisor modulates speed to 50% and safely engages emergency brake.",
        "duration": 6.0,
    },
    {
        "stage": DemoStage.CATTLE_CROSSING,
        "title": "Stage 6/7: Scenario 5 — Sudden Cattle Crossing",
        "description": "Demonstrates classic Indian road scenario: wandering cattle crossing road. AV halts safely, then recovers.",
        "duration": 8.0,
    },
    {
        "stage": DemoStage.BENCHMARKS_EXPORT,
        "title": "Stage 7/7: Comparative Benchmarking & Reports Export",
        "description": "Executes repeatable benchmarks (A* vs Dijkstra vs GBFS). Exports authentic CSV runs, summary, and charts.",
        "duration": 7.0,
    },
]


class DemoController:
    """
    Manages the Phase 10 demonstration walkthrough.
    Ensures deterministic, repeatable transitions across all simulation subsystems.
    """

    def __init__(self) -> None:
        self.is_active: bool = False
        self.current_stage_idx: int = 0
        self.stage_elapsed_time: float = 0.0
        self.is_paused: bool = False
        self.stage_executed: bool = False
        self.status_message: str = ""

    @property
    def current_meta(self) -> Dict[str, Any]:
        return STAGE_METADATA[self.current_stage_idx]

    @property
    def total_stages(self) -> int:
        return len(STAGE_METADATA)

    def start_demo(self, sim) -> None:
        """Start or restart the demonstration from Stage 1."""
        self.is_active = True
        self.current_stage_idx = 0
        self.stage_elapsed_time = 0.0
        self.is_paused = False
        self.stage_executed = False
        self.status_message = "Starting Demonstration..."
        self._enter_stage(self.current_stage_idx, sim)

    def stop_demo(self, sim) -> None:
        """Exit demonstration mode."""
        self.is_active = False
        self.stage_elapsed_time = 0.0
        self.status_message = "Demonstration exited."

    def toggle(self, sim) -> None:
        """Toggle Demo Mode on or off."""
        if self.is_active:
            self.stop_demo(sim)
        else:
            self.start_demo(sim)

    def next_stage(self, sim) -> None:
        """Advance to next demonstration stage."""
        if self.current_stage_idx < self.total_stages - 1:
            self.current_stage_idx += 1
            self.stage_elapsed_time = 0.0
            self.stage_executed = False
            self._enter_stage(self.current_stage_idx, sim)
        else:
            self.stop_demo(sim)

    def prev_stage(self, sim) -> None:
        """Return to previous demonstration stage."""
        if self.current_stage_idx > 0:
            self.current_stage_idx -= 1
            self.stage_elapsed_time = 0.0
            self.stage_executed = False
            self._enter_stage(self.current_stage_idx, sim)

    def _enter_stage(self, stage_idx: int, sim) -> None:
        """Execute stage setup actions deterministically."""
        meta = STAGE_METADATA[stage_idx]
        stage = meta["stage"]

        if stage == DemoStage.SCENARIO_AND_PLAN:
            # 1. Load Scenario 1 and compute initial route
            sc1 = get_scenario_by_id("village_road")
            if sc1:
                sim.load_scenario(sc1)
            sim.planner.algorithm = "astar"
            sim.plan_path()
            self.status_message = "Loaded Scenario 1 (Village Road). A* route planned."

        elif stage == DemoStage.VEHICLE_NAVIGATION:
            # 2. Start vehicle cruise along planned path
            if not sim.path_result or not sim.path_result.success:
                sim.plan_path()
            if sim.path_result and sim.path_result.success:
                sim.vehicle.set_path(sim.path_result.path)
                sim.vehicle.start()
            self.status_message = "Vehicle navigating route using pure pursuit controller."

        elif stage == DemoStage.STATIC_OBSTACLE_REPLAN:
            # 3. Inject a static obstacle midway on path and observe replan
            mid_c = 10
            mid_r = config.ROAD_ROW_CENTER
            if not sim.grid.is_occupied(mid_c, mid_r):
                sim.grid.add_obstacle(mid_c, mid_r)
                sim.obstacle_registry[(mid_c, mid_r)] = ObstacleType.BARRIER
            # Ensure supervisor is ready to replan
            sim.supervisor.replan_cooldown = 0.0
            self.status_message = f"Placed static barrier at ({mid_c}, {mid_r}). Replanning triggered."

        elif stage == DemoStage.DYNAMIC_TRACKING_PREDICTION:
            # 4. Show dynamic obstacle tracking & predictions
            sim.predictor.horizon = 1.5
            # Ensure oncoming road users exist
            has_dyn = any(not o.is_stationary for o in sim.ground_truth_objects)
            if not has_dyn:
                sim.ground_truth_objects.append(
                    DynamicObstacle(
                        category=ObstacleType.MOTORCYCLE,
                        x=sim.vehicle.x + 220.0,
                        y=config.ROAD_CENTER_Y - 15.0,
                        heading=math.pi,
                        speed=26.0,
                    )
                )
            self.status_message = "Tracking dynamic road users with 1.5s motion predictions."

        elif stage == DemoStage.COLLISION_AVOIDANCE:
            # 5. Bring obstacle closer to demonstrate CAUTION / EMERGENCY STOP
            sim.predictor.horizon = 1.0
            self.status_message = "Evaluating collision zones. Safety supervisor moderating speed."

        elif stage == DemoStage.CATTLE_CROSSING:
            # 6. Load Scenario 5: Sudden Cattle Crossing
            sc5 = get_scenario_by_id("cattle_crossing")
            if sc5:
                sim.load_scenario(sc5)
            sim.planner.algorithm = "astar"
            sim.plan_path()
            if sim.path_result and sim.path_result.success:
                sim.vehicle.set_path(sim.path_result.path)
                sim.vehicle.start()
            self.status_message = "Loaded Scenario 5 (Cattle Crossing). Cattle crossing road ahead."

        elif stage == DemoStage.BENCHMARKS_EXPORT:
            # 7. Open Benchmark Screen, run benchmarks, export reports
            sim.in_benchmark_screen = True
            sim.execute_benchmark_suite()
            sim.export_benchmark_reports()
            self.status_message = "Benchmark suite executed! Exported reports to reports/."

        self.stage_executed = True

    def update(self, dt: float, sim) -> None:
        """Update demo timers and handle automated stage progression."""
        if not self.is_active or self.is_paused:
            return

        self.stage_elapsed_time += dt
        meta = self.current_meta
        duration = meta["duration"]

        # Stage-specific mid-point behaviors
        stage = meta["stage"]
        if stage == DemoStage.CATTLE_CROSSING and self.stage_elapsed_time >= 3.0:
            # Trigger cattle movement if not yet moving
            for obj in sim.ground_truth_objects:
                if obj.category == ObstacleType.CATTLE and obj.is_stationary:
                    obj.speed = 14.0
                    obj.heading = -math.pi / 2.0  # cross northward

        # Auto-advance when stage duration finishes
        if self.stage_elapsed_time >= duration:
            self.next_stage(sim)

    def draw_banner(
        self,
        surface: pygame.Surface,
        title_font: pygame.font.Font,
        body_font: pygame.font.Font,
        small_font: pygame.font.Font,
    ) -> None:
        """Render a polished overlay banner across the top of the simulation viewport."""
        if not self.is_active:
            return

        banner_x = 12
        banner_y = config.SIM_AREA_TOP + 10
        banner_w = config.SIM_AREA_WIDTH - 24
        banner_h = 76

        banner_rect = pygame.Rect(banner_x, banner_y, banner_w, banner_h)

        # Semi-transparent dark background
        bg_surf = pygame.Surface((banner_w, banner_h), pygame.SRCALPHA)
        bg_surf.fill(config.C_DEMO_BANNER_BG)
        surface.blit(bg_surf, (banner_x, banner_y))

        # Accent border
        pygame.draw.rect(surface, config.C_DEMO_ACCENT, banner_rect, width=2, border_radius=6)

        meta = self.current_meta
        stage_num = self.current_stage_idx + 1

        # Stage Number Badge
        badge_rect = pygame.Rect(banner_x + 10, banner_y + 10, 80, 22)
        pygame.draw.rect(surface, config.C_DEMO_BADGE, badge_rect, border_radius=4)
        b_txt = small_font.render(f"STEP {stage_num}/7", True, (255, 255, 255))
        surface.blit(b_txt, (badge_rect.centerx - b_txt.get_width() // 2, badge_rect.centery - b_txt.get_height() // 2))

        # Title
        t_surf = title_font.render(meta["title"], True, config.C_DEMO_ACCENT)
        surface.blit(t_surf, (banner_x + 100, banner_y + 8))

        # Description
        d_surf = small_font.render(meta["description"], True, config.C_TEXT_PRIMARY)
        surface.blit(d_surf, (banner_x + 14, banner_y + 36))

        # Status & Navigation Hint
        hint_text = "[SPACE] Next Step | [BACKSPACE] Prev | [D / ESC] Exit Demo Mode"
        hint_surf = small_font.render(hint_text, True, (150, 210, 255))
        surface.blit(hint_surf, (banner_right_edge := banner_x + banner_w - hint_surf.get_width() - 14, banner_y + 54))

        # Progress bar
        bar_x = banner_x + 14
        bar_y = banner_y + 58
        bar_w = banner_w - hint_surf.get_width() - 38
        bar_h = 8

        pygame.draw.rect(surface, (20, 32, 55), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        progress = min(1.0, max(0.0, self.stage_elapsed_time / meta["duration"]))
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            pygame.draw.rect(surface, config.C_DEMO_PROGRESS_BAR, (bar_x, bar_y, fill_w, bar_h), border_radius=3)
        pygame.draw.rect(surface, (70, 100, 150), (bar_x, bar_y, bar_w, bar_h), width=1, border_radius=3)
