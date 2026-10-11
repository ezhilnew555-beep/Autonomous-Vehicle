"""
src/collision.py
================
Phase 7: Collision Avoidance, Safety Supervisor, and Dynamic Replanning.

Features:
- Backward-compatible `check_collision` function.
- Multi-step temporal trajectory & footprint verification against:
  - Static occupancy grid (roadway boundaries & barriers)
  - Detected moving obstacles (current PerceivedObject positions)
  - Phase 6 Constant-Velocity predicted positions and expanding uncertainty bounds
- Conservative handling of missing or unreliable predictions (inflated uncertainty margin)
- Safety Supervisor FSM with states:
  SAFE, CAUTION, REPLANNING, EMERGENCY_STOP, NO_SAFE_PATH
- Adaptive speed reduction as risk increases, emergency stopping on imminent collision
- Automatic A* replanning with candidate validation before route resumption
- Configurable replanning triggers, cooldowns, and audit event logging
- Visualization helpers for unsafe route segments and active conflict zones
"""

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import List, Tuple, Optional, Dict
import pygame

import config
from src.grid import OccupancyGrid
from src.obstacles import ObstacleType
from src.perception import PerceivedObject
from src.prediction import PredictedTrajectory, PredictedState
from src.planner import Planner, PathResult
from src.vehicle import Vehicle, VehicleState, world_to_grid, grid_to_world


# ---------------------------------------------------------------------------
# Backward Compatibility (Phase 1 Stub)
# ---------------------------------------------------------------------------
def check_collision(
    vehicle_pos: Tuple[int, int],
    obstacle_positions: List[Tuple[int, int]],
) -> bool:
    """Return True if vehicle overlaps any obstacle (preserved for Phase 1 tests)."""
    return vehicle_pos in obstacle_positions


# ---------------------------------------------------------------------------
# Phase 7 Safety States & Conflict Data Structures
# ---------------------------------------------------------------------------
class SafetyState(Enum):
    SAFE = "SAFE"
    CAUTION = "CAUTION"
    REPLANNING = "REPLANNING"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    NO_SAFE_PATH = "NO_SAFE_PATH"


@dataclass
class CollisionConflict:
    """Detailed record of an identified spatial or temporal collision conflict."""
    conflict_type: str         # "STATIC", "DYNAMIC_DETECTED", "PREDICTED", "UNCERTAINTY"
    obstacle_id: Optional[int] # Track ID (or None for static barrier/roadblock)
    time_offset: float        # Forward lookahead time (seconds)
    conflict_point: Tuple[float, float]
    distance_along_path: float
    severity: str             # "CRITICAL" (imminent collision) or "WARNING" (future conflict)
    details: str = ""


@dataclass
class SafetyLogEvent:
    """Audit log entry capturing state transitions, conflict triggers, and replans."""
    timestamp: float
    state: SafetyState
    event_type: str          # "STATE_CHANGE", "CONFLICT", "BRAKE", "REPLAN_SUCCESS", "REPLAN_FAILED", "RECOVERY"
    message: str


# ---------------------------------------------------------------------------
# Footprint & Geometric Intersection Checks
# ---------------------------------------------------------------------------
def check_footprint_collision(
    x: float,
    y: float,
    heading: float,
    length: float = config.VEHICLE_LENGTH,
    width: float = config.VEHICLE_WIDTH,
    grid: Optional[OccupancyGrid] = None,
    safety_margin: float = config.SAFETY_MARGIN_BASE,
) -> bool:
    """
    Check if placing the vehicle footprint (with safety margin) at (x, y)
    intersects any occupied grid cell or leaves valid road boundaries.
    """
    if grid is None:
        return False

    half_l = (length + 2.0 * safety_margin) * 0.5
    half_w = (width + 2.0 * safety_margin) * 0.5

    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    # Test center, corners, and midpoint edges for complete footprint coverage
    sample_offsets = [
        (0.0, 0.0),
        (half_l, half_w),
        (half_l, -half_w),
        (-half_l, half_w),
        (-half_l, -half_w),
        (half_l, 0.0),
        (-half_l, 0.0),
        (0.0, half_w),
        (0.0, -half_w),
    ]

    for dx, dy in sample_offsets:
        sx = x + (dx * cos_h - dy * sin_h)
        sy = y + (dx * sin_h + dy * cos_h)
        col, row = world_to_grid(sx, sy)

        if not grid.in_bounds(col, row):
            return True
        if not grid.is_road(col, row):
            return True
        if grid.is_occupied(col, row):
            return True

    return False


# ---------------------------------------------------------------------------
# Temporal Route & Conflict Evaluation
# ---------------------------------------------------------------------------
def evaluate_path_conflicts(
    waypoints: List[Tuple[float, float]],
    vehicle_x: float,
    vehicle_y: float,
    vehicle_speed: float,
    vehicle_heading: float,
    grid: OccupancyGrid,
    perceived_objects: List[PerceivedObject],
    predicted_trajectories: List[PredictedTrajectory],
    horizon: float = config.COLLISION_HORIZON_DEFAULT,
    sub_steps: int = config.COLLISION_LOOKAHEAD_STEPS,
    base_margin: float = config.SAFETY_MARGIN_BASE,
    uncertain_margin: float = config.SAFETY_MARGIN_UNCERTAIN,
) -> Tuple[List[CollisionConflict], Optional[int]]:
    """
    Simulate future vehicle progression along waypoints across multiple time steps.
    Check for temporal overlaps against:
      1. Static occupancy grid
      2. Detected obstacles at t=0
      3. Predicted trajectories & expanding uncertainty disks for t > 0
    Returns:
      (conflicts, first_unsafe_waypoint_index)
    """
    if not waypoints:
        return [], None

    conflicts: List[CollisionConflict] = []
    first_unsafe_idx: Optional[int] = None

    # Calculate cumulative path distances
    cum_dists = [0.0]
    total_len = 0.0
    for i in range(len(waypoints) - 1):
        d = math.hypot(waypoints[i + 1][0] - waypoints[i][0], waypoints[i + 1][1] - waypoints[i][1])
        total_len += d
        cum_dists.append(total_len)

    def get_point_along_path(dist: float) -> Tuple[float, float, float]:
        """Return (x, y, heading) at given distance along path."""
        if dist <= 0.0 or len(waypoints) < 2:
            h = vehicle_heading if len(waypoints) < 2 else math.atan2(
                waypoints[1][1] - waypoints[0][1], waypoints[1][0] - waypoints[0][0]
            )
            return waypoints[0][0], waypoints[0][1], h

        if dist >= total_len:
            last = waypoints[-1]
            prev = waypoints[-2]
            return last[0], last[1], math.atan2(last[1] - prev[1], last[0] - prev[0])

        for i in range(len(cum_dists) - 1):
            if cum_dists[i] <= dist <= cum_dists[i + 1]:
                seg_len = cum_dists[i + 1] - cum_dists[i]
                if seg_len < 1e-4:
                    return waypoints[i][0], waypoints[i][1], vehicle_heading
                frac = (dist - cum_dists[i]) / seg_len
                p1 = waypoints[i]
                p2 = waypoints[i + 1]
                px = p1[0] + frac * (p2[0] - p1[0])
                py = p1[1] + frac * (p2[1] - p1[1])
                ph = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                return px, py, ph

        return waypoints[-1][0], waypoints[-1][1], vehicle_heading

    # 1. Dense sampling along the entire path to catch any static grid obstacles
    sample_step = 16.0
    num_samples = max(1, int(math.ceil(total_len / sample_step)))
    for s_i in range(num_samples + 1):
        s_dist = min(total_len, s_i * sample_step)
        px, py, ph = get_point_along_path(s_dist)
        pc, pr = world_to_grid(px, py)
        if not grid.in_bounds(pc, pr) or not grid.is_road(pc, pr) or grid.is_occupied(pc, pr):
            sev = "CRITICAL" if s_dist < config.EMERGENCY_STOP_THRESHOLD else "WARNING"
            conflicts.append(
                CollisionConflict(
                    conflict_type="STATIC",
                    obstacle_id=None,
                    time_offset=s_dist / max(vehicle_speed, 10.0),
                    conflict_point=(px, py),
                    distance_along_path=s_dist,
                    severity=sev,
                    details=f"Static cell ({pc},{pr}) blocked on route at {s_dist:.1f}px",
                )
            )
            if first_unsafe_idx is None:
                for w_idx in range(len(waypoints)):
                    if cum_dists[w_idx] >= s_dist:
                        first_unsafe_idx = max(0, w_idx)
                        break
                if first_unsafe_idx is None:
                    first_unsafe_idx = len(waypoints) - 1
            break

    # Map trajectory predictions by track_id
    pred_map = {traj.track_id: traj for traj in predicted_trajectories}
    veh_radius = min(config.VEHICLE_LENGTH, config.VEHICLE_WIDTH) * 0.5
    effective_speed = max(vehicle_speed, 25.0)

    # 2. Path Proximity / Obstruction Check for Perceived Objects sitting on the route
    for obj in perceived_objects:
        traj = pred_map.get(obj.track_id)
        is_unreliable = (traj is None or traj.is_uncertain or traj.confidence_level == "LOW")
        margin = uncertain_margin if is_unreliable else base_margin
        target_radius = max(obj.length, obj.width) * 0.5 + (8.0 if is_unreliable else 2.0)
        req_clearance = veh_radius + target_radius + margin

        # Find closest point along path to this obstacle
        min_path_d = float("inf")
        best_s_dist = 0.0
        best_pt = (obj.estimated_x, obj.estimated_y)

        for s_i in range(num_samples + 1):
            s_dist = min(total_len, s_i * sample_step)
            px, py, _ = get_point_along_path(s_dist)
            d = math.hypot(px - obj.estimated_x, py - obj.estimated_y)
            if d < min_path_d:
                min_path_d = d
                best_s_dist = s_dist
                best_pt = (px, py)

        # If obstacle is sitting on or very close to the route ahead (within lookahead 350px)
        if min_path_d < req_clearance and best_s_dist <= 350.0:
            sev = "CRITICAL" if best_s_dist < config.EMERGENCY_STOP_THRESHOLD else "WARNING"
            ctype = "UNCERTAINTY" if is_unreliable else "DYNAMIC_DETECTED"
            conflicts.append(
                CollisionConflict(
                    conflict_type=ctype,
                    obstacle_id=obj.track_id,
                    time_offset=best_s_dist / effective_speed,
                    conflict_point=(obj.estimated_x, obj.estimated_y),
                    distance_along_path=best_s_dist,
                    severity=sev,
                    details=f"{obj.category.value} #{obj.track_id} on route at dist={best_s_dist:.1f}px (clearance={min_path_d:.1f}px < {req_clearance:.1f}px)",
                )
            )
            # Find closest waypoint index
            for w_idx in range(len(waypoints)):
                if cum_dists[w_idx] >= best_s_dist:
                    if first_unsafe_idx is None or w_idx < first_unsafe_idx:
                        first_unsafe_idx = max(0, w_idx)
                    break

    # 3. Temporal Future Steps t in [0, horizon] (for crossing / intercepting obstacles)
    time_steps = max(2, sub_steps)
    dt_step = horizon / float(time_steps)

    for s in range(time_steps + 1):
        t = s * dt_step
        ego_dist = effective_speed * t
        ego_x, ego_y, ego_h = get_point_along_path(ego_dist)

        for obj in perceived_objects:
            traj = pred_map.get(obj.track_id)
            is_unreliable = (traj is None or traj.is_uncertain or traj.confidence_level == "LOW")
            margin = uncertain_margin if is_unreliable else base_margin

            if t == 0.0 or traj is None or not traj.states:
                target_x = obj.estimated_x
                target_y = obj.estimated_y
                target_radius = max(obj.length, obj.width) * 0.5 + (8.0 if is_unreliable else 2.0)
            else:
                best_st = traj.states[0]
                min_tdiff = abs(best_st.time_offset - t)
                for st in traj.states:
                    diff = abs(st.time_offset - t)
                    if diff < min_tdiff:
                        min_tdiff = diff
                        best_st = st
                target_x = best_st.x
                target_y = best_st.y
                target_radius = best_st.uncertainty_radius

            center_dist = math.hypot(ego_x - target_x, ego_y - target_y)
            required_clearance = veh_radius + target_radius + margin

            if center_dist < required_clearance:
                sev = "CRITICAL" if (t <= 0.4 or ego_dist < config.EMERGENCY_STOP_THRESHOLD) else "WARNING"
                ctype = "UNCERTAINTY" if is_unreliable else ("DYNAMIC_DETECTED" if t == 0.0 else "PREDICTED")
                # Avoid duplicate conflicts for same object
                if not any(c.obstacle_id == obj.track_id and c.severity == sev for c in conflicts):
                    conflicts.append(
                        CollisionConflict(
                            conflict_type=ctype,
                            obstacle_id=obj.track_id,
                            time_offset=t,
                            conflict_point=(target_x, target_y),
                            distance_along_path=ego_dist,
                            severity=sev,
                            details=f"{obj.category.value} #{obj.track_id} intercept at t={t:.1f}s (dist={center_dist:.1f}px)",
                        )
                    )

    return conflicts, first_unsafe_idx


# ---------------------------------------------------------------------------
# Safety Supervisor FSM
# ---------------------------------------------------------------------------
class SafetySupervisor:
    """
    Phase 7 Safety Supervisor:
    - Maintains state machine: SAFE -> CAUTION -> REPLANNING -> EMERGENCY_STOP -> NO_SAFE_PATH
    - Regulates speed multiplier based on evaluated risk level
    - Triggers A* dynamic replanning when path is compromised
    - Maintains a comprehensive timestamped audit log of events and transitions
    """

    def __init__(
        self,
        replan_cooldown: float = config.REPLAN_COOLDOWN_TIME,
        emergency_distance: float = config.EMERGENCY_STOP_THRESHOLD,
    ) -> None:
        self.state: SafetyState = SafetyState.SAFE
        self.speed_multiplier: float = 1.0
        self.replan_cooldown: float = 0.0
        self.max_cooldown: float = replan_cooldown
        self.emergency_distance: float = emergency_distance

        self.active_conflicts: List[CollisionConflict] = []
        self.unsafe_waypoint_index: Optional[int] = None
        self.event_log: deque = deque(maxlen=40)

        # Performance counters
        self.total_replans_attempted: int = 0
        self.total_replans_succeeded: int = 0
        self.total_emergency_stops: int = 0

        self.log_event(0.0, SafetyState.SAFE, "INITIALIZE", "Safety Supervisor initialized in SAFE mode")

    @property
    def replan_count(self) -> int:
        return self.total_replans_succeeded

    @property
    def emergency_stop_count(self) -> int:
        return self.total_emergency_stops

    @property
    def conflict_history(self) -> list:
        return [ev for ev in self.event_log if ev.event_type in ("CONFLICT", "BRAKE", "STATE_CHANGE")]

    def log_event(self, timestamp: float, state: SafetyState, event_type: str, message: str) -> None:
        """Record an entry in the safety supervisor audit trail."""
        event = SafetyLogEvent(
            timestamp=round(timestamp, 2),
            state=state,
            event_type=event_type,
            message=message,
        )
        self.event_log.append(event)

    def reset(self) -> None:
        """Reset supervisor state back to SAFE."""
        self.state = SafetyState.SAFE
        self.speed_multiplier = 1.0
        self.replan_cooldown = 0.0
        self.active_conflicts.clear()
        self.unsafe_waypoint_index = None

    def update(
        self,
        sim_time: float,
        dt: float,
        vehicle: Vehicle,
        grid: OccupancyGrid,
        planner: Planner,
        perceived_objects: List[PerceivedObject],
        predicted_trajectories: List[PredictedTrajectory],
    ) -> SafetyState:
        """
        Execute safety supervisor control loop:
        1. Decrease replanning cooldown timer
        2. Evaluate path against static and dynamic conflicts
        3. Transition states, modulate vehicle speed, and execute replanning if necessary
        """
        if self.replan_cooldown > 0.0:
            self.replan_cooldown = max(0.0, self.replan_cooldown - dt)

        # If vehicle is not moving or has arrived, perform passive monitoring
        if vehicle.state in (VehicleState.IDLE, VehicleState.ARRIVED):
            if self.state != SafetyState.SAFE:
                self.state = SafetyState.SAFE
                self.speed_multiplier = 1.0
            self.active_conflicts.clear()
            self.unsafe_waypoint_index = None
            return self.state

        # Remaining waypoints from current target waypoint forward
        remaining_waypoints = [vehicle.pixel_center()]
        if vehicle.waypoints and vehicle.target_waypoint_index < len(vehicle.waypoints):
            remaining_waypoints.extend(vehicle.waypoints[vehicle.target_waypoint_index:])

        # Evaluate route for conflicts
        conflicts, unsafe_idx = evaluate_path_conflicts(
            waypoints=remaining_waypoints,
            vehicle_x=vehicle.x,
            vehicle_y=vehicle.y,
            vehicle_speed=vehicle.speed,
            vehicle_heading=vehicle.heading,
            grid=grid,
            perceived_objects=perceived_objects,
            predicted_trajectories=predicted_trajectories,
        )

        self.active_conflicts = conflicts
        # Adjust unsafe index relative to full path
        if unsafe_idx is not None and vehicle.waypoints:
            self.unsafe_waypoint_index = min(len(vehicle.waypoints) - 1, vehicle.target_waypoint_index + unsafe_idx)
        else:
            self.unsafe_waypoint_index = None

        has_critical = any(c.severity == "CRITICAL" for c in conflicts)
        has_warning = any(c.severity == "WARNING" for c in conflicts)

        prev_state = self.state

        # -------------------------------------------------------------
        # Decision Matrix & FSM Transitions
        # -------------------------------------------------------------
        if has_critical:
            # 1. IMMINENT COLLISION -> EMERGENCY STOP
            self.state = SafetyState.EMERGENCY_STOP
            self.speed_multiplier = 0.0
            vehicle.speed = 0.0
            self.total_emergency_stops += 1

            if prev_state != SafetyState.EMERGENCY_STOP:
                crit_c = next(c for c in conflicts if c.severity == "CRITICAL")
                self.log_event(
                    sim_time,
                    SafetyState.EMERGENCY_STOP,
                    "BRAKE",
                    f"EMERGENCY BRAKE: Imminent conflict ({crit_c.details})",
                )

        elif has_warning:
            # 2. FUTURE CONFLICT DETECTED -> REPLANNING OR CAUTION
            if self.replan_cooldown <= 0.0:
                # Trigger dynamic replanning
                self.state = SafetyState.REPLANNING
                self.speed_multiplier = config.CAUTION_SPEED_FACTOR
                self.total_replans_attempted += 1

                replan_success = self._attempt_replanning(
                    sim_time=sim_time,
                    vehicle=vehicle,
                    grid=grid,
                    planner=planner,
                    perceived_objects=perceived_objects,
                    predicted_trajectories=predicted_trajectories,
                )

                self.replan_cooldown = self.max_cooldown

                if replan_success:
                    self.total_replans_succeeded += 1
                    self.state = SafetyState.CAUTION
                    self.log_event(
                        sim_time,
                        SafetyState.CAUTION,
                        "REPLAN_SUCCESS",
                        "New collision-free detour path verified & adopted",
                    )
                else:
                    self.state = SafetyState.NO_SAFE_PATH
                    self.speed_multiplier = 0.0
                    vehicle.speed = 0.0
                    self.log_event(
                        sim_time,
                        SafetyState.NO_SAFE_PATH,
                        "REPLAN_FAILED",
                        "No alternative traversable route found around blockage",
                    )
            else:
                # Still in cooldown: reduce speed and proceed with caution
                if self.state not in (SafetyState.NO_SAFE_PATH, SafetyState.EMERGENCY_STOP):
                    self.state = SafetyState.CAUTION
                    self.speed_multiplier = config.CAUTION_SPEED_FACTOR
                    if prev_state != SafetyState.CAUTION:
                        self.log_event(
                            sim_time,
                            SafetyState.CAUTION,
                            "STATE_CHANGE",
                            f"Speed reduced to {int(config.CAUTION_SPEED_FACTOR*100)}% due to anticipated obstacle",
                        )

        else:
            # 3. NO CONFLICTS -> RECOVERY / SAFE
            if self.state in (SafetyState.EMERGENCY_STOP, SafetyState.NO_SAFE_PATH, SafetyState.CAUTION, SafetyState.REPLANNING):
                self.state = SafetyState.SAFE
                self.speed_multiplier = 1.0
                self.log_event(
                    sim_time,
                    SafetyState.SAFE,
                    "RECOVERY",
                    "Obstacle cleared: path confirmed safe, resuming nominal speed",
                )
                if vehicle.state == VehicleState.BLOCKED:
                    vehicle.state = VehicleState.MOVING
            else:
                self.state = SafetyState.SAFE
                self.speed_multiplier = 1.0

        return self.state

    def _attempt_replanning(
        self,
        sim_time: float,
        vehicle: Vehicle,
        grid: OccupancyGrid,
        planner: Planner,
        perceived_objects: List[PerceivedObject],
        predicted_trajectories: List[PredictedTrajectory],
    ) -> bool:
        """
        Generate and validate an alternative path around perceived obstacles.
        """
        # Create augmented grid with dynamic obstacle footprints marked
        augmented_grid = OccupancyGrid()
        if hasattr(grid, "_road_mask") and grid._road_mask is not None:
            augmented_grid.set_road_mask(grid._road_mask)
        # Copy static obstacles
        for r in range(config.GRID_ROWS):
            for c in range(config.GRID_COLS):
                if grid.is_occupied(c, r):
                    augmented_grid.add_obstacle(c, r)

        # Pad grid with current and short-term predicted positions of perceived objects
        for obj in perceived_objects:
            # Mark current cell
            oc, orow = world_to_grid(obj.estimated_x, obj.estimated_y)
            augmented_grid.add_obstacle(oc, orow)
            # Extra buffer for uncertain or broad entities
            if obj.category in (ObstacleType.CAR, ObstacleType.CATTLE, ObstacleType.AUTORICKSHAW):
                for dc, dr in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    augmented_grid.add_obstacle(oc + dc, orow + dr)

        # Plan from vehicle's current cell to destination
        start_cell = world_to_grid(vehicle.x, vehicle.y)
        dest_cell = getattr(grid, "dest_pos", (config.DEST_GRID_COL, config.DEST_GRID_ROW))
        augmented_grid.start_pos = start_cell
        augmented_grid.dest_pos = dest_cell

        # Ensure start cell is marked free on augmented grid so A* can begin
        if augmented_grid.is_occupied(start_cell[0], start_cell[1]):
            augmented_grid.remove_obstacle(start_cell[0], start_cell[1])

        # Plan candidate route
        candidate_result = planner.plan(augmented_grid, start=start_cell, goal=dest_cell)

        if not candidate_result.success or not candidate_result.path:
            return False

        # Validate candidate path before acceptance
        cand_waypoints = [grid_to_world(c, r) for c, r in candidate_result.path]
        cand_conflicts, _ = evaluate_path_conflicts(
            waypoints=cand_waypoints,
            vehicle_x=vehicle.x,
            vehicle_y=vehicle.y,
            vehicle_speed=config.VEHICLE_MAX_SPEED * config.CAUTION_SPEED_FACTOR,
            vehicle_heading=vehicle.heading,
            grid=grid,
            perceived_objects=perceived_objects,
            predicted_trajectories=predicted_trajectories,
            horizon=1.5,
        )

        # Accept path only if free of critical conflicts
        if any(c.severity == "CRITICAL" for c in cand_conflicts):
            return False

        # Adopt new path in vehicle
        vehicle.set_path(candidate_result.path)
        vehicle.state = VehicleState.MOVING
        return True


# ---------------------------------------------------------------------------
# Visualizations for Phase 7
# ---------------------------------------------------------------------------
def draw_collision_zones(
    surface: pygame.Surface,
    conflicts: List[CollisionConflict],
    font: pygame.font.Font,
) -> None:
    """Render warning disks and conflict indicators for active collision predictions."""
    for conf in conflicts:
        cx, cy = int(round(conf.conflict_point[0])), int(round(conf.conflict_point[1]))
        r = 16 if conf.severity == "CRITICAL" else 12

        zone_surf = pygame.Surface((2 * r + 2, 2 * r + 2), pygame.SRCALPHA)
        color_fill = (255, 50, 50, 110) if conf.severity == "CRITICAL" else (255, 180, 50, 80)
        color_border = (255, 80, 80, 220) if conf.severity == "CRITICAL" else (255, 210, 80, 180)

        pygame.draw.circle(zone_surf, color_fill, (r + 1, r + 1), r)
        pygame.draw.circle(zone_surf, color_border, (r + 1, r + 1), r, width=2)
        surface.blit(zone_surf, (cx - r - 1, cy - r - 1))

        # Tag label
        tag = f"! {conf.severity[:4]} {conf.time_offset:.1f}s"
        tag_surf = font.render(tag, True, (255, 220, 220) if conf.severity == "CRITICAL" else (255, 240, 180))
        surface.blit(tag_surf, (cx + 8, cy - 8))


def draw_unsafe_path_highlight(
    surface: pygame.Surface,
    waypoints: List[Tuple[float, float]],
    unsafe_start_index: Optional[int],
) -> None:
    """Highlight unsafe path segment in bright red with hazard nodes."""
    if unsafe_start_index is None or not waypoints or unsafe_start_index >= len(waypoints):
        return

    pts = [(int(round(x)), int(round(y))) for x, y in waypoints[unsafe_start_index:]]
    if len(pts) >= 2:
        pygame.draw.lines(surface, config.C_PATH_UNSAFE, False, pts, width=4)

    for px, py in pts:
        pygame.draw.circle(surface, config.C_PATH_UNSAFE, (px, py), 5)
        pygame.draw.circle(surface, (255, 255, 255), (px, py), 2)
