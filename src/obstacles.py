"""
src/obstacles.py
================
Obstacle types, simulated dynamic/static road user entities, and rendering helpers.
Phase 5: Extended to model diverse Indian road users:
- Cars
- Motorcycles
- Auto-rickshaws
- Pedestrians
- Cattle
- Static road barriers

Features:
- Unique, persistent ID for each ground-truth object
- Category, world coordinates (x, y), dimensions (width, height), heading, velocity (vx, vy)
- Active / inactive status
- Physics update for moving objects within road boundary constraints
- Clean separation between ground truth and perception
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Tuple, List, Optional
import pygame

import config


class ObstacleType(Enum):
    CAR = "Car"
    MOTORCYCLE = "Motorcycle"
    AUTORICKSHAW = "Auto-rickshaw"
    PEDESTRIAN = "Pedestrian"
    CATTLE = "Cattle"
    BARRIER = "Barrier"

    # Backward compatibility with Phase 1-3
    PARKED_CAR = "Car"
    ROADBLOCK = "Barrier"
    BLOCKED = "Barrier"


# Dimensions: (length/height along heading, width across heading, default_speed)
OBJECT_DIMENSIONS = {
    ObstacleType.CAR: (26.0, 16.0, 30.0),
    ObstacleType.MOTORCYCLE: (18.0, 8.0, 45.0),
    ObstacleType.AUTORICKSHAW: (20.0, 14.0, 35.0),
    ObstacleType.PEDESTRIAN: (10.0, 10.0, 15.0),
    ObstacleType.CATTLE: (24.0, 12.0, 10.0),
    ObstacleType.BARRIER: (config.GRID_CELL_SIZE - 6.0, config.GRID_CELL_SIZE - 6.0, 0.0),
}

# Color palette mapped per object category
_TYPE_COLOURS = {
    ObstacleType.CAR: (config.C_OBJ_CAR, (255, 170, 90)),
    ObstacleType.MOTORCYCLE: (config.C_OBJ_MOTORCYCLE, (255, 230, 120)),
    ObstacleType.AUTORICKSHAW: (config.C_OBJ_AUTORICKSHAW, (100, 240, 200)),
    ObstacleType.PEDESTRIAN: (config.C_OBJ_PEDESTRIAN, (255, 120, 180)),
    ObstacleType.CATTLE: (config.C_OBJ_CATTLE, (210, 150, 100)),
    ObstacleType.BARRIER: (config.C_OBJ_BARRIER, (220, 100, 100)),
}


_NEXT_OBJ_ID = 1


def generate_object_id() -> int:
    """Generate globally unique, persistent integer object ID."""
    global _NEXT_OBJ_ID
    oid = _NEXT_OBJ_ID
    _NEXT_OBJ_ID += 1
    return oid


def reset_object_id_counter() -> None:
    """Reset counter for reproducible testing."""
    global _NEXT_OBJ_ID
    _NEXT_OBJ_ID = 1


@dataclass
class DynamicObstacle:
    """
    Ground-truth simulated obstacle entity.
    Can be stationary (barrier, parked car) or moving (pedestrian, cow, moving vehicle).
    """
    id: int = field(default_factory=generate_object_id)
    category: ObstacleType = ObstacleType.CAR
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0  # radians (0 = pointing right +X)
    speed: float = 0.0  # pixels / second along heading
    length: float = 24.0
    width: float = 14.0
    is_active: bool = True

    # Trajectory bounds for movement ping-pong / patrol
    min_x: float = 0.0
    max_x: float = float(config.SIM_AREA_WIDTH)
    bound_y: bool = True
    min_y: float = float(config.SIM_AREA_TOP)
    max_y: float = float(config.SCREEN_HEIGHT)

    def __post_init__(self):
        default_len, default_w, _ = OBJECT_DIMENSIONS.get(self.category, (24.0, 14.0, 0.0))
        if self.length == 24.0 and self.width == 14.0:
            self.length = default_len
            self.width = default_w

    @property
    def velocity(self) -> Tuple[float, float]:
        """Ground truth velocity vector (vx, vy)."""
        if abs(self.speed) < 1e-3:
            return 0.0, 0.0
        return self.speed * math.cos(self.heading), self.speed * math.sin(self.heading)

    @property
    def is_stationary(self) -> bool:
        return abs(self.speed) < 1e-3

    def update(self, dt: float) -> None:
        """Advance obstacle physics over delta time (dt seconds)."""
        if not self.is_active or self.is_stationary:
            return

        vx, vy = self.velocity
        new_x = self.x + vx * dt
        new_y = self.y + vy * dt

        # Bounce or turnaround if hitting horizontal bounds
        if new_x <= self.min_x:
            new_x = self.min_x
            self.heading = math.pi - self.heading
        elif new_x >= self.max_x:
            new_x = self.max_x
            self.heading = math.pi - self.heading

        # Vertical bounds check
        if self.bound_y:
            pad = self.width / 2.0
            road_min_y = config.ROAD_TOP_Y + pad
            road_max_y = config.ROAD_BOTTOM_Y - pad
            if road_min_y <= self.y <= road_max_y:
                if new_y <= road_min_y:
                    new_y = road_min_y
                    self.heading = -self.heading
                elif new_y >= road_max_y:
                    new_y = road_max_y
                    self.heading = -self.heading
        else:
            # Unconstrained crossing (e.g. crossing cattle or intersection cross-traffic)
            if new_y <= self.min_y or new_y >= self.max_y:
                # Turnaround at screen boundaries
                self.heading = -self.heading

        self.x = new_x
        self.y = new_y


# ---------------------------------------------------------------------------
# Backward Compatibility Wrappers
# ---------------------------------------------------------------------------
@dataclass
class Obstacle:
    """Legacy cell-level obstacle representation for Phase 1-3 tests."""
    grid_col: int = 0
    grid_row: int = 0
    obstacle_type: ObstacleType = ObstacleType.BARRIER
    color: Tuple[int, int, int] = field(default=None)
    border_color: Tuple[int, int, int] = field(default=None)

    def __post_init__(self):
        cat = self.obstacle_type if self.obstacle_type in _TYPE_COLOURS else ObstacleType.BARRIER
        if self.color is None:
            self.color = _TYPE_COLOURS[cat][0]
        if self.border_color is None:
            self.border_color = _TYPE_COLOURS[cat][1]

    @property
    def label(self) -> str:
        return self.obstacle_type.value


def default_obstacles() -> List[Obstacle]:
    """Return default set of obstacles."""
    return []


# ---------------------------------------------------------------------------
# Visual Rendering
# ---------------------------------------------------------------------------
def draw_obstacle_cell(
    surface: pygame.Surface,
    col: int,
    row: int,
    obs_type: ObstacleType = ObstacleType.BARRIER,
) -> None:
    """Draw a static cell obstacle (for occupancy grid)."""
    cell = config.GRID_CELL_SIZE
    pad = 3
    x = col * cell + pad
    y = row * cell + config.SIM_AREA_TOP + pad
    w = cell - 2 * pad
    h = cell - 2 * pad

    cat = obs_type if obs_type in _TYPE_COLOURS else ObstacleType.BARRIER
    fill, border = _TYPE_COLOURS[cat]

    rect = pygame.Rect(x, y, w, h)
    pygame.draw.rect(surface, fill, rect, border_radius=4)
    pygame.draw.rect(surface, border, rect, width=2, border_radius=4)

    inset = 8
    pygame.draw.line(surface, config.C_OBSTACLE_X, (x + inset, y + inset), (x + w - inset, y + h - inset), 2)
    pygame.draw.line(surface, config.C_OBSTACLE_X, (x + w - inset, y + inset), (x + inset, y + h - inset), 2)


def draw_all_obstacles(surface: pygame.Surface, grid, obstacle_registry: dict) -> None:
    """Render every occupied cell from the occupancy grid."""
    for col, row in grid.occupied_positions():
        obs_type = obstacle_registry.get((col, row), ObstacleType.BARRIER)
        draw_obstacle_cell(surface, col, row, obs_type)


def draw_ground_truth_object(
    surface: pygame.Surface,
    obj: DynamicObstacle,
    show_label: bool = True,
    font: Optional[pygame.font.Font] = None,
) -> None:
    """
    Render top-down ground-truth road user with distinct geometric symbols:
    - Car: elongated chassis + roof
    - Motorcycle: narrow two-wheeler
    - Auto-rickshaw: three-wheeler tapered front
    - Pedestrian: small circle / torso
    - Cattle: broad torso with horn markings
    - Barrier: striped rectangle
    """
    if not obj.is_active:
        return

    x, y = obj.x, obj.y
    cat = obj.category
    heading = obj.heading
    cos_h = math.cos(heading)
    sin_h = math.sin(heading)

    def rot(lx: float, ly: float) -> Tuple[int, int]:
        wx = x + (lx * cos_h - ly * sin_h)
        wy = y + (lx * sin_h + ly * cos_h)
        return int(round(wx)), int(round(wy))

    hl = obj.length / 2.0
    hw = obj.width / 2.0
    fill_col, border_col = _TYPE_COLOURS.get(cat, ((180, 180, 180), (220, 220, 220)))

    if cat == ObstacleType.CAR:
        poly = [rot(hl, -hw), rot(hl, hw), rot(-hl, hw), rot(-hl, -hw)]
        pygame.draw.polygon(surface, fill_col, poly)
        pygame.draw.polygon(surface, border_col, poly, width=2)
        # Windshield line
        pygame.draw.line(surface, (255, 255, 255), rot(hl * 0.3, -hw * 0.8), rot(hl * 0.3, hw * 0.8), 2)

    elif cat == ObstacleType.MOTORCYCLE:
        # Narrow elongated chassis with two wheels
        poly = [rot(hl, -hw), rot(hl, hw), rot(-hl, hw), rot(-hl, -hw)]
        pygame.draw.polygon(surface, fill_col, poly)
        pygame.draw.polygon(surface, border_col, poly, width=2)
        pygame.draw.circle(surface, (30, 30, 30), rot(hl * 0.7, 0.0), 3)
        pygame.draw.circle(surface, (30, 30, 30), rot(-hl * 0.7, 0.0), 3)

    elif cat == ObstacleType.AUTORICKSHAW:
        # Triangular / tapered front for 3-wheeler
        poly = [
            rot(hl, 0.0),  # front wheel tip
            rot(hl * 0.4, -hw),
            rot(-hl, -hw),
            rot(-hl, hw),
            rot(hl * 0.4, hw),
        ]
        pygame.draw.polygon(surface, fill_col, poly)
        pygame.draw.polygon(surface, border_col, poly, width=2)
        # Yellow canopy roof
        canopy = [rot(hl * 0.1, -hw * 0.7), rot(hl * 0.1, hw * 0.7), rot(-hl * 0.8, hw * 0.7), rot(-hl * 0.8, -hw * 0.7)]
        pygame.draw.polygon(surface, (240, 220, 60), canopy)

    elif cat == ObstacleType.PEDESTRIAN:
        # Circle head and torso
        pygame.draw.circle(surface, fill_col, (int(round(x)), int(round(y))), int(round(hl)))
        pygame.draw.circle(surface, border_col, (int(round(x)), int(round(y))), int(round(hl)), width=2)
        # Heading nose
        pygame.draw.line(surface, (255, 255, 255), (int(round(x)), int(round(y))), rot(hl + 4.0, 0.0), 2)

    elif cat == ObstacleType.CATTLE:
        # Ellipse-like body with horns
        poly = [rot(hl, -hw), rot(hl, hw), rot(-hl, hw), rot(-hl, -hw)]
        pygame.draw.polygon(surface, fill_col, poly)
        pygame.draw.polygon(surface, border_col, poly, width=2)
        # Horn markings on head
        pygame.draw.line(surface, (240, 240, 240), rot(hl, -hw * 0.8), rot(hl + 4.0, -hw * 1.2), 2)
        pygame.draw.line(surface, (240, 240, 240), rot(hl, hw * 0.8), rot(hl + 4.0, hw * 1.2), 2)

    else:  # BARRIER
        rect = pygame.Rect(int(x - hl), int(y - hw), int(obj.length), int(obj.width))
        pygame.draw.rect(surface, fill_col, rect, border_radius=3)
        pygame.draw.rect(surface, border_col, rect, width=2, border_radius=3)
        # Stripes
        pygame.draw.line(surface, (255, 255, 255), rect.topleft, rect.bottomright, 2)

    # Object tag label: "[ID] Cat"
    if show_label and font is not None:
        label_text = f"#{obj.id} {cat.value}"
        text_surf = font.render(label_text, True, config.C_TEXT_PRIMARY)
        surface.blit(text_surf, (int(x - text_surf.get_width() / 2), int(y - hw - 14)))
