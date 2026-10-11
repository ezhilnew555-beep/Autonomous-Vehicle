"""
src/vehicle.py
==============
Autonomous Vehicle implementation for Phase 4: Movement and Path Following.

Features:
- Full 2D kinematic vehicle state:
  - World position (x, y) in simulation pixels
  - Heading in radians (0 = pointing right along the +X axis)
  - Speed with acceleration/deceleration constraints
  - Angular velocity with maximum turning rate constraints
- Discrete states: IDLE, MOVING, PAUSED, ARRIVED, BLOCKED
- Pure pursuit / waypoint tracker following the A* path
- Intermediate continuous collision checks (footprint sampling) to prevent
  tunneling or skipping narrow obstacles
- Decoupled from Pygame rendering (pure physics / controller logic)
- Rendering helper `draw_vehicle(surface)` drawing top-down stylized vehicle
  with cabin, headlights, taillights, and forward orientation indicator
"""

from enum import Enum
import math
from typing import List, Tuple, Optional
import pygame

import config


class VehicleState(Enum):
    IDLE = "IDLE"
    MOVING = "MOVING"
    PAUSED = "PAUSED"
    ARRIVED = "ARRIVED"
    BLOCKED = "BLOCKED"


def grid_to_world(col: int, row: int) -> Tuple[float, float]:
    """Convert grid (col, row) to world pixel center (x, y)."""
    half = config.GRID_CELL_SIZE / 2.0
    x = col * config.GRID_CELL_SIZE + half
    y = row * config.GRID_CELL_SIZE + config.SIM_AREA_TOP + half
    return float(x), float(y)


def world_to_grid(x: float, y: float) -> Tuple[int, int]:
    """Convert world pixel coordinate to grid (col, row)."""
    col = int(x // config.GRID_CELL_SIZE)
    row = int((y - config.SIM_AREA_TOP) // config.GRID_CELL_SIZE)
    return col, row


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class Vehicle:
    """Simulated autonomous vehicle with smooth kinematic movement."""

    def __init__(
        self,
        start_col: int = config.START_GRID_COL,
        start_row: int = config.START_GRID_ROW,
        max_speed: float = config.VEHICLE_MAX_SPEED,
        acceleration: float = config.VEHICLE_ACCEL,
        deceleration: float = config.VEHICLE_DECEL,
        max_turn_rate: float = config.VEHICLE_MAX_TURN,
    ) -> None:
        self.start_col = start_col
        self.start_row = start_row

        # World position (pixels)
        start_x, start_y = grid_to_world(start_col, start_row)
        self.x: float = start_x
        self.y: float = start_y

        # Kinematics
        self.heading: float = 0.0  # radians (pointing rightwards towards destination)
        self.speed: float = 0.0  # pixels / second
        self.max_speed: float = max_speed
        self.acceleration: float = acceleration
        self.deceleration: float = deceleration
        self.max_turn_rate: float = max_turn_rate

        # Dimensions & collision footprint radius
        self.length: float = config.VEHICLE_LENGTH
        self.width: float = config.VEHICLE_WIDTH
        self.collision_radius: float = min(self.length, self.width) / 2.0 + 2.0

        # Path following state
        self.state: VehicleState = VehicleState.IDLE
        self.path: List[Tuple[int, int]] = []
        self.waypoints: List[Tuple[float, float]] = []
        self.target_waypoint_index: int = 0
        self.elapsed_time: float = 0.0
        self.speed_multiplier: float = 1.0

        # For backward compatibility
        self.label: str = "AV"

    # ------------------------------------------------------------------ #
    # Compatibility properties for Phase 1 & 2 tests
    # ------------------------------------------------------------------ #
    @property
    def grid_col(self) -> int:
        col, _ = world_to_grid(self.x, self.y)
        return col

    @property
    def grid_row(self) -> int:
        _, row = world_to_grid(self.x, self.y)
        return row

    def pixel_position(self) -> Tuple[int, int]:
        x = self.grid_col * config.GRID_CELL_SIZE
        y = self.grid_row * config.GRID_CELL_SIZE + config.SIM_AREA_TOP
        return x, y

    def pixel_center(self) -> Tuple[int, int]:
        return int(round(self.x)), int(round(self.y))

    # ------------------------------------------------------------------ #
    # Mission & Route Assignment
    # ------------------------------------------------------------------ #
    def set_path(self, path: List[Tuple[int, int]]) -> bool:
        """
        Assign an ordered A* path to the vehicle.
        Converts grid coordinates to world waypoints.
        """
        if not path:
            self.path = []
            self.waypoints = []
            self.target_waypoint_index = 0
            return False

        self.path = list(path)
        self.waypoints = [grid_to_world(c, r) for c, r in path]

        # Reset vehicle to start of path if currently IDLE
        if self.state in (VehicleState.IDLE, VehicleState.ARRIVED, VehicleState.BLOCKED):
            start_x, start_y = self.waypoints[0]
            self.x = start_x
            self.y = start_y
            self.speed = 0.0
            # Orient towards second waypoint if available
            if len(self.waypoints) > 1:
                dx = self.waypoints[1][0] - self.x
                dy = self.waypoints[1][1] - self.y
                self.heading = math.atan2(dy, dx)
            else:
                self.heading = 0.0
            self.target_waypoint_index = 1 if len(self.waypoints) > 1 else 0
        else:
            # Active / moving vehicle replanning: steer towards next waypoint on replanned path
            self.target_waypoint_index = 1 if len(self.waypoints) > 1 else 0

        return True

    def start(self) -> bool:
        """Start or resume vehicle navigation along the path."""
        if not self.waypoints or len(self.waypoints) < 1:
            return False

        if self.state == VehicleState.ARRIVED:
            # Already at destination
            return False

        self.state = VehicleState.MOVING
        return True

    def pause(self) -> None:
        """Pause vehicle movement without resetting progress."""
        if self.state == VehicleState.MOVING:
            self.state = VehicleState.PAUSED

    def resume(self) -> None:
        """Resume movement from paused state."""
        if self.state == VehicleState.PAUSED:
            self.state = VehicleState.MOVING

    def reset(self) -> None:
        """Reset vehicle back to starting position and IDLE state."""
        start_x, start_y = grid_to_world(self.start_col, self.start_row)
        self.x = start_x
        self.y = start_y
        self.heading = 0.0
        self.speed = 0.0
        self.state = VehicleState.IDLE
        self.target_waypoint_index = 0
        self.elapsed_time = 0.0
        self.speed_multiplier = 1.0
        if self.waypoints and len(self.waypoints) > 1:
            self.target_waypoint_index = 1
            dx = self.waypoints[1][0] - self.x
            dy = self.waypoints[1][1] - self.y
            self.heading = math.atan2(dy, dx)

    # ------------------------------------------------------------------ #
    # Telemetry and Status
    # ------------------------------------------------------------------ #
    @property
    def heading_degrees(self) -> float:
        """Heading in degrees in [0, 360)."""
        deg = math.degrees(self.heading) % 360.0
        return round(deg, 1)

    @property
    def total_waypoints(self) -> int:
        return len(self.waypoints)

    @property
    def remaining_distance(self) -> float:
        """Euclidean distance remaining along remaining waypoints to destination."""
        if not self.waypoints or self.target_waypoint_index >= len(self.waypoints):
            return 0.0

        # Distance from current position to next waypoint
        tx, ty = self.waypoints[self.target_waypoint_index]
        dist = math.hypot(tx - self.x, ty - self.y)

        # Plus distances between subsequent waypoints
        for i in range(self.target_waypoint_index, len(self.waypoints) - 1):
            p1 = self.waypoints[i]
            p2 = self.waypoints[i + 1]
            dist += math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        return round(dist, 1)

    # ------------------------------------------------------------------ #
    # Collision Verification
    # ------------------------------------------------------------------ #
    def check_collision_at(self, x: float, y: float, grid) -> bool:
        """
        Check if placing the vehicle footprint center at (x, y) intersects an occupied cell.
        Samples center and corners of vehicle bounding box.
        """
        # Test center cell
        c, r = world_to_grid(x, y)
        if not grid.in_bounds(c, r) or grid.is_occupied(c, r) or not grid.is_road(c, r):
            return True

        # Test footprint corner offsets along vehicle orientation
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)
        half_l = self.length * 0.45
        half_w = self.width * 0.45

        corners = [
            (half_l, half_w),
            (half_l, -half_w),
            (-half_l, half_w),
            (-half_l, -half_w),
        ]

        for dx_local, dy_local in corners:
            cx = x + (dx_local * cos_h - dy_local * sin_h)
            cy = y + (dx_local * sin_h + dy_local * cos_h)
            cc, cr = world_to_grid(cx, cy)
            if not grid.in_bounds(cc, cr) or grid.is_occupied(cc, cr):
                return True

        return False

    def is_route_blocked(self, grid) -> bool:
        """Check if any upcoming waypoint on the active route is occupied."""
        if not self.path:
            return False
        # Check all cells from current target waypoint to destination
        for idx in range(self.target_waypoint_index, len(self.path)):
            c, r = self.path[idx]
            if grid.is_occupied(c, r):
                return True
        return False

    # ------------------------------------------------------------------ #
    # Kinematic Update Loop
    # ------------------------------------------------------------------ #
    def update(self, dt: float, grid) -> None:
        """
        Advance vehicle physics by delta time (dt seconds).
        Pure pursuit kinematic controller with acceleration and turning limits.
        """
        if self.state != VehicleState.MOVING:
            return

        self.elapsed_time += dt

        # Route safety check: if an obstacle was placed on upcoming route, stop immediately
        if self.is_route_blocked(grid):
            self.speed = 0.0
            self.state = VehicleState.BLOCKED
            return

        if not self.waypoints or self.target_waypoint_index >= len(self.waypoints):
            self.state = VehicleState.ARRIVED
            self.speed = 0.0
            return

        # Current target waypoint
        tx, ty = self.waypoints[self.target_waypoint_index]
        dist_to_target = math.hypot(tx - self.x, ty - self.y)

        # Check if current waypoint is reached
        if dist_to_target < config.WAYPOINT_THRESHOLD:
            # Advance to next waypoint
            self.target_waypoint_index += 1
            if self.target_waypoint_index >= len(self.waypoints):
                # Arrived at final destination
                self.x = tx
                self.y = ty
                self.speed = 0.0
                self.state = VehicleState.ARRIVED
                return
            tx, ty = self.waypoints[self.target_waypoint_index]
            dist_to_target = math.hypot(tx - self.x, ty - self.y)

        # Steering: rotate towards target waypoint
        desired_angle = math.atan2(ty - self.y, tx - self.x)
        angle_diff = normalize_angle(desired_angle - self.heading)

        max_turn_step = self.max_turn_rate * dt
        if abs(angle_diff) <= max_turn_step:
            self.heading = desired_angle
        else:
            self.heading += math.copysign(max_turn_step, angle_diff)
        self.heading = normalize_angle(self.heading)

        # Speed control: accelerate, but slow down when turning sharply or approaching destination
        is_last_waypoint = (self.target_waypoint_index == len(self.waypoints) - 1)
        stopping_distance = (self.speed ** 2) / (2.0 * self.deceleration + 1e-5)

        target_speed = self.max_speed * max(0.0, self.speed_multiplier)

        # Sharp turn penalty: slow down if not facing target
        if abs(angle_diff) > math.radians(45):
            target_speed *= 0.4
        elif abs(angle_diff) > math.radians(20):
            target_speed *= 0.7

        # Deceleration as vehicle nears final waypoint
        if is_last_waypoint and dist_to_target < stopping_distance + config.WAYPOINT_THRESHOLD:
            target_speed = max(15.0, self.max_speed * (dist_to_target / (stopping_distance + config.WAYPOINT_THRESHOLD)))

        # Accelerate / decelerate towards target speed
        if self.speed < target_speed:
            self.speed = min(target_speed, self.speed + self.acceleration * dt)
        else:
            self.speed = max(target_speed, self.speed - self.deceleration * dt)

        # Sub-stepping continuous collision detection
        step_distance = self.speed * dt
        if step_distance <= 0.0:
            return

        # Split step into small increments (max 4 pixels per sub-step)
        sub_steps = max(1, int(math.ceil(step_distance / 4.0)))
        sub_dist = step_distance / sub_steps

        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)

        curr_x, curr_y = self.x, self.y
        blocked = False

        for _ in range(sub_steps):
            next_x = curr_x + cos_h * sub_dist
            next_y = curr_y + sin_h * sub_dist

            if self.check_collision_at(next_x, next_y, grid):
                blocked = True
                break
            curr_x, curr_y = next_x, next_y

        if blocked:
            self.speed = 0.0
            self.state = VehicleState.BLOCKED
        else:
            self.x = curr_x
            self.y = curr_y


# ---------------------------------------------------------------------------
# Vehicle Rendering
# ---------------------------------------------------------------------------
def draw_vehicle(surface: pygame.Surface, vehicle: Vehicle) -> None:
    """
    Render a top-down stylized autonomous vehicle oriented along its heading.
    Includes chassis, cabin roof, headlights, taillights, and heading vector.
    """
    x, y = vehicle.x, vehicle.y
    length = vehicle.length
    width = vehicle.width
    heading = vehicle.heading

    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    def to_world(lx: float, ly: float) -> Tuple[int, int]:
        wx = x + (lx * cos_h - ly * sin_h)
        wy = y + (lx * sin_h + ly * cos_h)
        return int(round(wx)), int(round(wy))

    half_l = length / 2.0
    half_w = width / 2.0

    # 1. Main body chassis (rounded rectangle polygon)
    body_corners = [
        to_world(half_l, -half_w),
        to_world(half_l, half_w),
        to_world(-half_l, half_w),
        to_world(-half_l, -half_w),
    ]
    pygame.draw.polygon(surface, config.C_VEHICLE_BODY, body_corners)
    pygame.draw.polygon(surface, config.C_VEHICLE_OUTLINE, body_corners, width=2)

    # 2. Cabin / Roof (centered slightly backwards)
    cabin_l = half_l * 0.7
    cabin_w = half_w * 0.75
    cabin_offset = -half_l * 0.1
    cabin_corners = [
        to_world(cabin_offset + cabin_l, -cabin_w),
        to_world(cabin_offset + cabin_l, cabin_w),
        to_world(cabin_offset - cabin_l, cabin_w),
        to_world(cabin_offset - cabin_l, -cabin_w),
    ]
    pygame.draw.polygon(surface, config.C_VEHICLE_ROOF, cabin_corners)

    # 3. Headlights (front two corners, warm yellow)
    hl_offset = half_l * 0.85
    hl_spread = half_w * 0.65
    pygame.draw.circle(surface, config.C_VEHICLE_HEADLIGHT, to_world(hl_offset, -hl_spread), 3)
    pygame.draw.circle(surface, config.C_VEHICLE_HEADLIGHT, to_world(hl_offset, hl_spread), 3)

    # 4. Taillights (rear two corners, red)
    tl_offset = -half_l * 0.85
    tl_spread = half_w * 0.65
    pygame.draw.circle(surface, config.C_VEHICLE_TAILLIGHT, to_world(tl_offset, -tl_spread), 2)
    pygame.draw.circle(surface, config.C_VEHICLE_TAILLIGHT, to_world(tl_offset, tl_spread), 2)

    # 5. Front heading pointer (vector indicating direction of travel)
    tip_pt = to_world(half_l + 8.0, 0.0)
    front_center = to_world(half_l, 0.0)
    pygame.draw.line(surface, config.C_VEHICLE_HEADING, front_center, tip_pt, 2)
