"""
src/scenarios.py
================
Phase 8: Five Realistic Indian Road Scenarios
Provides complete scenario definitions, road masks, static/dynamic obstacle layouts,
scheduled event timelines, and deterministic reproduction.

Scenarios:
1. Unmarked Village Road (narrow rural road, uneven edges, roadside stones, pedestrians)
2. Unsignalized Urban Intersection (4-way intersection, cross-traffic, corner blind spots)
3. Highway Merging with Slow-Moving Vehicles (merging on-ramp, gap assessment, speed adjustments)
4. Dense Market Area with Mixed Traffic (crowded corridor, shops, auto-rickshaws, pedestrians)
5. Sudden Cattle Crossing (rural road, scheduled cow crossing path unexpectedly)
"""

from dataclasses import dataclass, field
import math
import random
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

import config
from src.obstacles import ObstacleType, DynamicObstacle, reset_object_id_counter


@dataclass
class ScheduledEvent:
    """An event scheduled to trigger at a specific simulation timestamp."""
    trigger_time: float                 # seconds into simulation
    action_type: str                    # "SPAWN_OBSTACLE", "ACTIVATE_MOVEMENT", "CUSTOM"
    description: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    has_triggered: bool = False

    def reset(self) -> None:
        self.has_triggered = False


@dataclass
class Scenario:
    """Complete specification of a simulated driving scenario."""
    id: str
    name: str
    description: str
    road_type: str = "straight"         # "village", "intersection", "merge", "market", "straight", "cattle"
    road_mask: Optional[np.ndarray] = None
    start_pos: Tuple[int, int] = (config.START_GRID_COL, config.ROAD_ROW_CENTER)
    dest_pos: Tuple[int, int] = (config.DEST_GRID_COL, config.ROAD_ROW_CENTER)
    obstacle_positions: List[Tuple[int, int]] = field(default_factory=list)
    dynamic_obstacles: List[DynamicObstacle] = field(default_factory=list)
    scheduled_events: List[ScheduledEvent] = field(default_factory=list)
    timeout: float = config.SCENARIO_TIMEOUT_DEFAULT
    seed: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)

    def reset_events(self) -> None:
        """Reset triggers for all scheduled events."""
        for ev in self.scheduled_events:
            ev.reset()

    def get_due_events(self, sim_time: float) -> List[ScheduledEvent]:
        """Return list of events whose trigger time has elapsed and have not triggered yet."""
        due = []
        for ev in self.scheduled_events:
            if not ev.has_triggered and sim_time >= ev.trigger_time:
                due.append(ev)
        return due


# ---------------------------------------------------------------------------
# Helper: Road Mask Generators
# ---------------------------------------------------------------------------
def _build_village_mask() -> np.ndarray:
    """Narrow rural road: 2 lanes wide."""
    mask = np.zeros((config.GRID_ROWS, config.GRID_COLS), dtype=bool)
    center = config.ROAD_ROW_CENTER
    # 2 rows in the middle
    mask[center - 1 : center + 1, :] = True
    return mask


def _build_intersection_mask() -> np.ndarray:
    """4-way urban intersection: horizontal road + vertical crossing street."""
    mask = np.zeros((config.GRID_ROWS, config.GRID_COLS), dtype=bool)
    # Horizontal road
    top_r = config.ROAD_ROW_CENTER - config.ROAD_HALF_WIDTH
    bot_r = config.ROAD_ROW_CENTER + config.ROAD_HALF_WIDTH
    mask[top_r:bot_r, :] = True

    # Vertical cross street (4 columns wide around center)
    mid_c = config.GRID_COLS // 2
    mask[:, mid_c - 2 : mid_c + 2] = True
    return mask


def _build_merge_mask() -> np.ndarray:
    """Highway with on-ramp merging lane entering from bottom-left."""
    mask = np.zeros((config.GRID_ROWS, config.GRID_COLS), dtype=bool)
    top_r = config.ROAD_ROW_CENTER - config.ROAD_HALF_WIDTH
    bot_r = config.ROAD_ROW_CENTER + config.ROAD_HALF_WIDTH
    mask[top_r:bot_r, :] = True

    # Merging on-ramp (cols 1 to 9, rows bot_r to bot_r + 2)
    ramp_rows = min(config.GRID_ROWS, bot_r + 2)
    mask[bot_r:ramp_rows, 1:9] = True
    return mask


def _build_market_mask() -> np.ndarray:
    """Standard road with corridor."""
    mask = np.zeros((config.GRID_ROWS, config.GRID_COLS), dtype=bool)
    top_r = config.ROAD_ROW_CENTER - config.ROAD_HALF_WIDTH
    bot_r = config.ROAD_ROW_CENTER + config.ROAD_HALF_WIDTH
    mask[top_r:bot_r, :] = True
    return mask


# ---------------------------------------------------------------------------
# Scenario 1: Unmarked Village Road
# ---------------------------------------------------------------------------
def create_village_scenario(seed: int = 42) -> Scenario:
    """
    Scenario 1: Narrow rural road without lane markings.
    Uneven edges, roadside stones, parked auto-rickshaw, oncoming motorcycle.
    """
    random.seed(seed)
    np.random.seed(seed)
    reset_object_id_counter()

    center_r = config.ROAD_ROW_CENTER
    mask = _build_village_mask()

    # Stones and roadside hazards narrowing driving space
    static_stones = [
        (6, center_r - 1),   # stone on north shoulder edge
        (13, center_r),      # parked obstruction in north lane
        (17, center_r - 1),  # stone
        (20, center_r),      # roadside structure
    ]

    dyn_obs = [
        # Oncoming motorcycle in south lane
        DynamicObstacle(
            category=ObstacleType.MOTORCYCLE,
            x=650.0,
            y=float(center_r * config.GRID_CELL_SIZE + config.SIM_AREA_TOP + 15),
            heading=math.pi,  # heading west
            speed=28.0,
            min_x=250.0,
            max_x=850.0,
        ),
        # Pedestrian walking along road edge
        DynamicObstacle(
            category=ObstacleType.PEDESTRIAN,
            x=320.0,
            y=float((center_r - 1) * config.GRID_CELL_SIZE + config.SIM_AREA_TOP + 10),
            heading=0.0,
            speed=12.0,
            min_x=200.0,
            max_x=550.0,
        ),
    ]

    return Scenario(
        id="village_road",
        name="1. Unmarked Village Road",
        description="Narrow rural road without lane markings, uneven gravel shoulders, roadside stones, and oncoming traffic.",
        road_type="village",
        road_mask=mask,
        start_pos=(2, center_r),
        dest_pos=(config.GRID_COLS - 3, center_r - 1),
        obstacle_positions=static_stones,
        dynamic_obstacles=dyn_obs,
        timeout=28.0,
        seed=seed,
        metadata={"lane_markings": False, "road_width_lanes": 2},
    )


# ---------------------------------------------------------------------------
# Scenario 2: Unsignalized Urban Intersection
# ---------------------------------------------------------------------------
def create_intersection_scenario(seed: int = 101) -> Scenario:
    """
    Scenario 2: 4-way unsignalized urban intersection.
    Crossing vehicles and motorcycles approaching from cross streets with corner blind spots.
    """
    random.seed(seed)
    np.random.seed(seed)
    reset_object_id_counter()

    mask = _build_intersection_mask()
    mid_c = config.GRID_COLS // 2
    center_r = config.ROAD_ROW_CENTER

    # Corner structures creating blind spots near intersection mouth
    corner_blocks = [
        (mid_c - 3, center_r - 3),
        (mid_c - 3, center_r - 2),
        (mid_c + 2, center_r - 3),
        (mid_c + 2, center_r - 2),
        (mid_c - 3, center_r + 2),
        (mid_c + 2, center_r + 2),
    ]

    # Cross street traffic moving vertically through intersection
    mid_x = mid_c * config.GRID_CELL_SIZE
    dyn_obs = [
        # Car heading south across intersection
        DynamicObstacle(
            category=ObstacleType.CAR,
            x=float(mid_x - 20),
            y=float(config.SIM_AREA_TOP + 30),
            heading=math.pi / 2.0,  # south
            speed=32.0,
            bound_y=False,
            min_y=float(config.SIM_AREA_TOP),
            max_y=float(config.SCREEN_HEIGHT),
        ),
        # Motorcycle heading north across intersection
        DynamicObstacle(
            category=ObstacleType.MOTORCYCLE,
            x=float(mid_x + 25),
            y=float(config.SCREEN_HEIGHT - 40),
            heading=-math.pi / 2.0,  # north
            speed=36.0,
            bound_y=False,
            min_y=float(config.SIM_AREA_TOP),
            max_y=float(config.SCREEN_HEIGHT),
        ),
    ]

    return Scenario(
        id="urban_intersection",
        name="2. Unsignalized Urban Intersection",
        description="4-way intersection without traffic signals. Crossing vehicles and corner blind spots requiring gap anticipation.",
        road_type="intersection",
        road_mask=mask,
        start_pos=(2, center_r),
        dest_pos=(config.GRID_COLS - 3, center_r),
        obstacle_positions=corner_blocks,
        dynamic_obstacles=dyn_obs,
        timeout=30.0,
        seed=seed,
        metadata={"intersection_type": "4-way", "traffic_signals": False},
    )


# ---------------------------------------------------------------------------
# Scenario 3: Highway Merging with Slow-Moving Vehicles
# ---------------------------------------------------------------------------
def create_merge_scenario(seed: int = 202) -> Scenario:
    """
    Scenario 3: Highway merging with slow-moving vehicles.
    AV enters via on-ramp behind a slow vehicle and must merge into faster highway traffic.
    """
    random.seed(seed)
    np.random.seed(seed)
    reset_object_id_counter()

    mask = _build_merge_mask()
    center_r = config.ROAD_ROW_CENTER
    bot_r = center_r + config.ROAD_HALF_WIDTH

    # Ramp shoulder barricade
    ramp_barriers = [
        (9, bot_r),  # End of merge lane barrier
        (10, bot_r),
    ]

    # Slow vehicle in merge lane + faster cruising traffic on highway
    dyn_obs = [
        # Slow-moving auto-rickshaw ahead in merge lane
        DynamicObstacle(
            category=ObstacleType.AUTORICKSHAW,
            x=240.0,
            y=float(bot_r * config.GRID_CELL_SIZE + config.SIM_AREA_TOP + 15),
            heading=0.0,
            speed=16.0,
            min_x=150.0,
            max_x=340.0,
        ),
        # Fast car on main highway lane
        DynamicObstacle(
            category=ObstacleType.CAR,
            x=180.0,
            y=float(center_r * config.GRID_CELL_SIZE + config.SIM_AREA_TOP),
            heading=0.0,
            speed=45.0,
            min_x=100.0,
            max_x=850.0,
        ),
    ]

    return Scenario(
        id="highway_merge",
        name="3. Highway Merging with Slow Traffic",
        description="Merging entry lane with slow-moving auto-rickshaw ahead and faster highway traffic in adjacent lanes.",
        road_type="merge",
        road_mask=mask,
        start_pos=(2, bot_r),  # start on on-ramp
        dest_pos=(config.GRID_COLS - 3, center_r - 1),  # destination on cruising lane
        obstacle_positions=ramp_barriers,
        dynamic_obstacles=dyn_obs,
        timeout=32.0,
        seed=seed,
        metadata={"merge_type": "on-ramp", "gap_assessment_required": True},
    )


# ---------------------------------------------------------------------------
# Scenario 4: Dense Market Area with Mixed Traffic
# ---------------------------------------------------------------------------
def create_market_scenario(seed: int = 303) -> Scenario:
    """
    Scenario 4: Crowded market street flanked by vendor stalls and mixed road users.
    """
    random.seed(seed)
    np.random.seed(seed)
    reset_object_id_counter()

    mask = _build_market_mask()
    center_r = config.ROAD_ROW_CENTER
    top_r = center_r - config.ROAD_HALF_WIDTH
    bot_r = center_r + config.ROAD_HALF_WIDTH

    # Vendor stalls and parked carts lining the road edges
    market_stalls = [
        (4, top_r),
        (5, top_r),
        (8, bot_r - 1),  # protrusion
        (11, top_r),
        (15, bot_r - 1),
        (16, top_r),
        (19, bot_r - 1),
    ]

    dyn_obs = [
        # Wandering pedestrian 1
        DynamicObstacle(
            category=ObstacleType.PEDESTRIAN,
            x=280.0,
            y=float(center_r * config.GRID_CELL_SIZE + config.SIM_AREA_TOP + 10),
            heading=-0.2,
            speed=12.0,
            min_x=220.0,
            max_x=420.0,
        ),
        # Wandering pedestrian 2
        DynamicObstacle(
            category=ObstacleType.PEDESTRIAN,
            x=540.0,
            y=float((center_r - 1) * config.GRID_CELL_SIZE + config.SIM_AREA_TOP + 15),
            heading=0.3,
            speed=10.0,
            min_x=480.0,
            max_x=650.0,
        ),
        # Agile auto-rickshaw negotiating market crowd
        DynamicObstacle(
            category=ObstacleType.AUTORICKSHAW,
            x=420.0,
            y=float((center_r) * config.GRID_CELL_SIZE + config.SIM_AREA_TOP - 10),
            heading=0.0,
            speed=20.0,
            min_x=350.0,
            max_x=650.0,
        ),
    ]

    return Scenario(
        id="dense_market",
        name="4. Dense Market with Mixed Traffic",
        description="Crowded market street with vendor stalls, pedestrians, auto-rickshaws, and limited free navigation space.",
        road_type="market",
        road_mask=mask,
        start_pos=(2, center_r),
        dest_pos=(config.GRID_COLS - 3, center_r),
        obstacle_positions=market_stalls,
        dynamic_obstacles=dyn_obs,
        timeout=35.0,
        seed=seed,
        metadata={"density": "high", "speed_moderation": "conservative"},
    )


# ---------------------------------------------------------------------------
# Scenario 5: Sudden Cattle Crossing
# ---------------------------------------------------------------------------
def create_cattle_scenario(seed: int = 404) -> Scenario:
    """
    Scenario 5: Rural road with an unexpected cattle crossing event.
    Cattle starts near verge and steps out into the vehicle's path at t = 1.2s.
    """
    random.seed(seed)
    np.random.seed(seed)
    reset_object_id_counter()

    center_r = config.ROAD_ROW_CENTER

    # Create cattle obstacle starting off-road on northern verge (initially stationary)
    cattle_crosser = DynamicObstacle(
        id=77,
        category=ObstacleType.CATTLE,
        x=450.0,
        y=float(config.ROAD_TOP_Y - 24.0),
        heading=math.pi / 2.0,  # facing south towards road
        speed=0.0,              # stationary until scheduled trigger
        bound_y=False,          # can cross entire road
        min_y=float(config.SIM_AREA_TOP),
        max_y=float(config.SCREEN_HEIGHT),
    )

    # Scheduled dynamic event: cow suddenly walks into roadway at t = 1.2 seconds
    cross_event = ScheduledEvent(
        trigger_time=1.2,
        action_type="ACTIVATE_MOVEMENT",
        description="Cattle suddenly steps out onto roadway crossing AV path",
        payload={"obstacle_id": 77, "speed": 22.0, "heading": math.pi / 2.0},
    )

    return Scenario(
        id="cattle_crossing",
        name="5. Sudden Cattle Crossing",
        description="Rural highway where cattle unexpectedly walks into the roadway, testing imminent collision braking & safe recovery.",
        road_type="cattle",
        road_mask=None,  # default road
        start_pos=(2, center_r),
        dest_pos=(config.GRID_COLS - 3, center_r),
        obstacle_positions=[],
        dynamic_obstacles=[cattle_crosser],
        scheduled_events=[cross_event],
        timeout=28.0,
        seed=seed,
        metadata={"event_trigger_time": 1.2, "hazard": "erratic_animal"},
    )


# ---------------------------------------------------------------------------
# Scenario Registry
# ---------------------------------------------------------------------------
def get_all_scenarios(seed_base: int = 42) -> List[Scenario]:
    """Return all five Phase 8 selectable scenarios."""
    return [
        create_village_scenario(seed=seed_base),
        create_intersection_scenario(seed=seed_base + 1),
        create_merge_scenario(seed=seed_base + 2),
        create_market_scenario(seed=seed_base + 3),
        create_cattle_scenario(seed=seed_base + 4),
    ]


def get_scenario_by_id(scenario_id: str, seed: int = 42) -> Optional[Scenario]:
    """Retrieve scenario by its unique identifier."""
    registry = {
        "village_road": lambda: create_village_scenario(seed),
        "urban_intersection": lambda: create_intersection_scenario(seed),
        "highway_merge": lambda: create_merge_scenario(seed),
        "dense_market": lambda: create_market_scenario(seed),
        "cattle_crossing": lambda: create_cattle_scenario(seed),
    }
    factory = registry.get(scenario_id)
    return factory() if factory else None


# ---------------------------------------------------------------------------
# Backward Compatibility Wrappers (for Phase 7 Tests)
# ---------------------------------------------------------------------------
def load_default_scenario() -> Scenario:
    return Scenario(id="default", name="Default", description="Default demonstration scenario")

def load_safe_route_scenario() -> Scenario:
    s = Scenario(id="safe_route", name="Safe Route", description="Unobstructed road")
    s.dynamic_obstacles = [
        DynamicObstacle(category=ObstacleType.CAR, x=500.0, y=config.ROAD_TOP_Y + 12.0, speed=0.0)
    ]
    return s

def load_blocked_route_scenario() -> Scenario:
    s = Scenario(id="blocked_route", name="Blocked Route", description="Direct path blocked")
    mid_col = config.GRID_COLS // 2
    mid_row = config.ROAD_ROW_CENTER
    s.obstacle_positions = [(mid_col, mid_row)]
    s.dynamic_obstacles = [
        DynamicObstacle(
            category=ObstacleType.BARRIER,
            x=float(mid_col * config.GRID_CELL_SIZE + config.GRID_CELL_SIZE / 2),
            y=float(mid_row * config.GRID_CELL_SIZE + config.SIM_AREA_TOP + config.GRID_CELL_SIZE / 2),
            speed=0.0,
        )
    ]
    return s

def load_crossing_obstacle_scenario() -> Scenario:
    s = Scenario(id="crossing_obstacle", name="Crossing Obstacle", description="Crossing pedestrian")
    s.dynamic_obstacles = [
        DynamicObstacle(
            category=ObstacleType.PEDESTRIAN,
            x=320.0,
            y=config.ROAD_TOP_Y + 10.0,
            heading=math.pi / 2.0,
            speed=22.0,
            min_x=300.0,
            max_x=340.0,
        )
    ]
    return s

def load_no_safe_route_scenario() -> Scenario:
    s = Scenario(id="no_safe_route", name="No Safe Route", description="Complete roadblock")
    mid_col = config.GRID_COLS // 2
    top_row = config.ROAD_ROW_CENTER - config.ROAD_HALF_WIDTH
    bot_row = config.ROAD_ROW_CENTER + config.ROAD_HALF_WIDTH
    s.obstacle_positions = [(mid_col, r) for r in range(top_row, bot_row)]
    return s

def load_recovery_scenario() -> Scenario:
    s = Scenario(id="recovery", name="Recovery", description="Route clears")
    s.dynamic_obstacles = [
        DynamicObstacle(category=ObstacleType.CATTLE, x=450.0, y=config.ROAD_CENTER_Y - 35.0, speed=8.0)
    ]
    return s
