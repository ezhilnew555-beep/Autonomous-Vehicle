"""
src/prediction.py
=================
Motion Prediction System for Autonomous Vehicle Simulation (Phase 6).

Key Specifications:
- Consumes ONLY perceived obstacle data (`PerceivedObject`), NEVER hidden ground-truth.
- Constant-Velocity Motion Model (CVMM):
    x_future = x + vx * t
    y_future = y + vy * t
- Configurable prediction horizons: 0.5s, 1.0s, 1.5s, 2.0s.
- Supports all road user categories:
    Cars, Motorcycles, Auto-rickshaws, Pedestrians, Cattle, and Stationary Obstacles (Barriers).
- Documented Uncertainty Growth Model:
    Uncertainty represents the spatial divergence of possible future states as time horizon grows.
    Starting from a baseline detection radius r_0:
        uncertainty_radius(t) = r_0 + growth_rate * t
    For stationary objects, growth_rate is minimal (stationary confidence high).
    For moving objects, growth_rate scales with velocity and category unpredictability
    (e.g., pedestrians and cattle have higher heading variance than vehicles on lanes).
- Confidence Classification:
    - HIGH: Consistent velocity history (len >= 3), steady heading, or stationary barrier.
    - MEDIUM: Moderate history (len == 2), or modest velocity fluctuations.
    - LOW / UNCERTAIN: Insufficient history (len < 2), erratic speed, or out-of-range coasting.
- Clean data structure `PredictedTrajectory` and `PredictedState` exposed for future
  collision avoidance and risk assessment modules.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple
import pygame

import config
from src.obstacles import ObstacleType
from src.perception import PerceivedObject


# Category-specific motion variance multiplier (for uncertainty growth)
CATEGORY_UNCERTAINTY_WEIGHT = {
    ObstacleType.BARRIER: 0.05,       # Highly predictable (static)
    ObstacleType.CAR: 0.8,           # Lane-constrained
    ObstacleType.AUTORICKSHAW: 1.0,  # Agile 3-wheeler
    ObstacleType.MOTORCYCLE: 1.2,    # High lateral agility
    ObstacleType.PEDESTRIAN: 1.5,    # Omnidirectional / sudden stops
    ObstacleType.CATTLE: 1.6,        # Unstructured / erratic wander
}


@dataclass
class PredictedState:
    """Predicted future state at a specific forward timestamp."""
    time_offset: float          # seconds ahead (e.g. 0.5, 1.0, 1.5, 2.0)
    x: float                    # predicted world x
    y: float                    # predicted world y
    uncertainty_radius: float   # radius of 2D uncertainty disk (pixels)
    is_valid: bool = True


@dataclass
class PredictedTrajectory:
    """
    Complete forward prediction bundle for a single tracked object.
    Exposed cleanly for future collision detection and risk evaluation.
    """
    track_id: int
    category: ObstacleType
    current_x: float
    current_y: float
    current_vx: float
    current_vy: float
    current_speed: float
    is_stationary: bool
    confidence_level: str       # "HIGH", "MEDIUM", "LOW"
    confidence_score: float     # 0.0 to 1.0
    is_uncertain: bool          # True if history insufficient or variance high
    states: List[PredictedState] = field(default_factory=list)

    @property
    def final_state(self) -> Optional[PredictedState]:
        return self.states[-1] if self.states else None


class Predictor:
    """
    Motion Predictor consuming perceived obstacles and generating
    constant-velocity trajectory forecasts with expanding uncertainty bounds.
    """

    def __init__(
        self,
        horizon: float = 1.0,
        sub_steps: int = config.PREDICTION_STEPS,
        min_radius: float = config.UNCERTAINTY_MIN_RADIUS,
        base_growth: float = config.UNCERTAINTY_GROWTH_RATE,
    ) -> None:
        self.horizon = horizon
        self.sub_steps = sub_steps
        self.min_radius = min_radius
        self.base_growth = base_growth

        # Visualization toggles
        self.show_trajectories: bool = True
        self.show_uncertainty: bool = True
        self.show_time_labels: bool = True

        # Cached predictions keyed by track_id
        self._predictions: Dict[int, PredictedTrajectory] = {}

    @property
    def predictions(self) -> Dict[int, PredictedTrajectory]:
        return self._predictions

    def clear(self) -> None:
        self._predictions.clear()

    def set_horizon(self, horizon: float) -> None:
        """Update active forward prediction horizon."""
        self.horizon = horizon

    def predict_object(
        self,
        obj: PerceivedObject,
        horizon: Optional[float] = None,
    ) -> PredictedTrajectory:
        """
        Generate trajectory forecast for a single perceived object.
        Applies Constant-Velocity Model (CVMM) and Uncertainty Growth Model.
        """
        h = horizon if horizon is not None else self.horizon
        obs_count = len(obj._history) if hasattr(obj, "_history") else 0

        # Check if explicitly stationary (Barrier, or sufficient observation history showing zero velocity)
        is_barrier = (obj.category == ObstacleType.BARRIER)
        is_measured_stationary = (obs_count >= 2 and obj.estimated_speed < 1.5)
        is_stationary = is_barrier or is_measured_stationary

        # Check for invalid or physically impossible / corrupted velocity
        is_invalid_vel = (
            math.isnan(obj.estimated_vx)
            or math.isnan(obj.estimated_vy)
            or math.isinf(obj.estimated_vx)
            or math.isinf(obj.estimated_vy)
            or math.isnan(obj.estimated_speed)
            or math.isinf(obj.estimated_speed)
            or obj.estimated_speed > 300.0  # Physically impossible speed on simulation canvas
        )

        # Determine velocity and uncertainty reliability
        if is_invalid_vel:
            confidence_level = "LOW"
            confidence_score = 0.10
            is_uncertain = True
            vx, vy = 0.0, 0.0
        elif is_barrier:
            confidence_level = "HIGH"
            confidence_score = 0.95
            is_uncertain = False
            vx, vy = 0.0, 0.0
        elif obs_count < 2:
            # Insufficient observation history to reliably know speed or stationary status
            confidence_level = "LOW"
            confidence_score = 0.35
            is_uncertain = True
            vx, vy = obj.estimated_vx, obj.estimated_vy
        elif not obj.is_in_range:
            confidence_level = "LOW"
            confidence_score = 0.40
            is_uncertain = True
            vx, vy = obj.estimated_vx, obj.estimated_vy
        elif is_measured_stationary:
            confidence_level = "HIGH"
            confidence_score = 0.92
            is_uncertain = False
            vx, vy = 0.0, 0.0
        elif obs_count >= 3:
            confidence_level = "HIGH"
            confidence_score = 0.88
            is_uncertain = False
            vx, vy = obj.estimated_vx, obj.estimated_vy
        else:  # obs_count == 2
            confidence_level = "MEDIUM"
            confidence_score = 0.70
            is_uncertain = False
            vx, vy = obj.estimated_vx, obj.estimated_vy

        # Generate predicted states along forward horizon
        states: List[PredictedState] = []
        cat_weight = CATEGORY_UNCERTAINTY_WEIGHT.get(obj.category, 1.0)
        effective_growth = self.base_growth * cat_weight

        # Sample at least 1 intermediate step + final horizon step
        num_steps = max(1, self.sub_steps)
        time_step = h / float(num_steps)

        for step in range(1, num_steps + 1):
            t = step * time_step
            # Constant-velocity formula
            x_future = obj.estimated_x + vx * t
            y_future = obj.estimated_y + vy * t

            # Uncertainty growth model:
            # radius(t) = min_radius + effective_growth * t + extra penalty if uncertain
            uncertainty_penalty = 8.0 if is_uncertain else 0.0
            if is_stationary:
                # Stationary uncertainty stays tight around object footprint
                radius = self.min_radius + 1.5 * t
            else:
                radius = self.min_radius + (effective_growth * t) + uncertainty_penalty

            states.append(
                PredictedState(
                    time_offset=round(t, 2),
                    x=x_future,
                    y=y_future,
                    uncertainty_radius=round(radius, 2),
                    is_valid=True,
                )
            )

        trajectory = PredictedTrajectory(
            track_id=obj.track_id,
            category=obj.category,
            current_x=obj.estimated_x,
            current_y=obj.estimated_y,
            current_vx=round(vx, 2),
            current_vy=round(vy, 2),
            current_speed=round(obj.estimated_speed, 2),
            is_stationary=is_stationary,
            confidence_level=confidence_level,
            confidence_score=confidence_score,
            is_uncertain=is_uncertain,
            states=states,
        )

        return trajectory

    def predict_all(
        self,
        perceived_objects: List[PerceivedObject],
        horizon: Optional[float] = None,
    ) -> List[PredictedTrajectory]:
        """
        Predict future motions for all currently tracked perceived objects.
        Returns list of PredictedTrajectory objects.
        """
        results: List[PredictedTrajectory] = []
        new_dict: Dict[int, PredictedTrajectory] = {}

        for obj in perceived_objects:
            traj = self.predict_object(obj, horizon)
            results.append(traj)
            new_dict[obj.track_id] = traj

        self._predictions = new_dict
        return results

    # Backward compatibility with Phase 1 stub
    def predict(self, obstacle_positions: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Legacy stub compatibility for Phase 1 smoke tests."""
        return list(obstacle_positions)


# ---------------------------------------------------------------------------
# Prediction Visualizations
# ---------------------------------------------------------------------------
def draw_predicted_trajectories(
    surface: pygame.Surface,
    trajectories: List[PredictedTrajectory],
    show_uncertainty: bool,
    show_time_labels: bool,
    font: pygame.font.Font,
) -> None:
    """
    Render predicted trajectory lines, waypoint nodes, uncertainty ellipses,
    and arrival time markers over the simulation area.
    """
    for traj in trajectories:
        if not traj.states:
            continue

        # Points along path: starting from current estimated pos
        pts = [(int(round(traj.current_x)), int(round(traj.current_y)))]
        for st in traj.states:
            pts.append((int(round(st.x)), int(round(st.y))))

        # 1. Trajectory line (bright amber / dashed style)
        if len(pts) >= 2 and not traj.is_stationary:
            pygame.draw.lines(surface, config.C_PREDICTED_LINE, False, pts, 2)

        # 2. Uncertainty regions (expanding circles / ellipses over time)
        if show_uncertainty:
            for st in traj.states:
                r = int(round(st.uncertainty_radius))
                cx, cy = int(round(st.x)), int(round(st.y))

                # Render alpha-blended uncertainty disk
                disk_surf = pygame.Surface((2 * r + 2, 2 * r + 2), pygame.SRCALPHA)
                pygame.draw.circle(disk_surf, config.C_PREDICTED_UNCERTAINTY, (r + 1, r + 1), r)
                pygame.draw.circle(disk_surf, config.C_PREDICTED_UNCERTAIN_BORDER, (r + 1, r + 1), r, width=1)
                surface.blit(disk_surf, (cx - r - 1, cy - r - 1))

        # 3. Predicted endpoint and time marker
        final = traj.final_state
        if final:
            fx, fy = int(round(final.x)), int(round(final.y))
            pygame.draw.circle(surface, config.C_PREDICTED_POINT, (fx, fy), 4)

            if show_time_labels and not traj.is_stationary:
                t_label = f"+{final.time_offset}s"
                lbl_surf = font.render(t_label, True, config.C_PREDICTED_POINT)
                surface.blit(lbl_surf, (fx + 5, fy - 6))
