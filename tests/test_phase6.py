"""
tests/test_phase6.py
====================
Automated test suite for Phase 6: Motion Prediction System
- Trajectory calculations using Constant-Velocity Model (CVMM)
- Stationary obstacles (Barriers & parked vehicles) forecasting
- Configurable prediction horizons (0.5s, 1.0s, 1.5s, 2.0s)
- Handling insufficient or unreliable history (low confidence, uncertain flag)
- Uncertainty radius growth over time
- High, Medium, Low confidence level definitions
- Separation from ground truth (uses perceived object estimates only)
- Output format validation for future collision module consumption
"""

import sys
import os
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.obstacles import ObstacleType, DynamicObstacle
from src.perception import PerceivedObject, Observation
from src.prediction import (
    Predictor,
    PredictedTrajectory,
    PredictedState,
    CATEGORY_UNCERTAINTY_WEIGHT,
)


def _make_perceived_object(
    track_id: int = 1,
    category: ObstacleType = ObstacleType.CAR,
    x: float = 100.0,
    y: float = 200.0,
    vx: float = 20.0,
    vy: float = 0.0,
    history_count: int = 3,
    is_in_range: bool = True,
) -> PerceivedObject:
    """Helper creating a test PerceivedObject with specified observation history."""
    obj = PerceivedObject(
        track_id=track_id,
        category=category,
        estimated_x=x,
        estimated_y=y,
        estimated_heading=math.atan2(vy, vx) if abs(vx) > 1e-3 or abs(vy) > 1e-3 else 0.0,
        estimated_vx=vx,
        estimated_vy=vy,
        estimated_speed=math.hypot(vx, vy),
        is_in_range=is_in_range,
    )

    # Populate dummy observation history
    for i in range(history_count):
        obj._history.append(
            Observation(
                timestamp=i * 0.1,
                x=x - (history_count - 1 - i) * vx * 0.1,
                y=y - (history_count - 1 - i) * vy * 0.1,
                heading=obj.estimated_heading,
                category=category,
                length=24.0,
                width=14.0,
            )
        )
    return obj


def test_constant_velocity_prediction_math():
    """Verify CVMM calculates x_future = x + vx * t and y_future = y + vy * t."""
    predictor = Predictor(horizon=1.0, sub_steps=2)
    obj = _make_perceived_object(x=100.0, y=200.0, vx=30.0, vy=10.0, history_count=3)

    traj = predictor.predict_object(obj)
    assert traj.track_id == 1
    assert len(traj.states) == 2

    # Step 1: t = 0.5s -> x = 100 + 30*0.5 = 115, y = 200 + 10*0.5 = 205
    st1 = traj.states[0]
    assert math.isclose(st1.time_offset, 0.5, abs_tol=1e-2)
    assert math.isclose(st1.x, 115.0, abs_tol=1e-2)
    assert math.isclose(st1.y, 205.0, abs_tol=1e-2)

    # Step 2: t = 1.0s -> x = 100 + 30*1.0 = 130, y = 200 + 10*1.0 = 210
    st2 = traj.states[1]
    assert math.isclose(st2.time_offset, 1.0, abs_tol=1e-2)
    assert math.isclose(st2.x, 130.0, abs_tol=1e-2)
    assert math.isclose(st2.y, 210.0, abs_tol=1e-2)


def test_stationary_obstacles_forecast():
    """Verify stationary barriers and parked objects forecast zero displacement with high confidence."""
    predictor = Predictor(horizon=1.5)
    barrier = _make_perceived_object(
        track_id=2, category=ObstacleType.BARRIER, x=250.0, y=300.0, vx=0.0, vy=0.0, history_count=1
    )

    traj = predictor.predict_object(barrier)
    assert traj.is_stationary is True
    assert traj.confidence_level == "HIGH"
    assert traj.is_uncertain is False

    # Predicted states must remain fixed at origin
    for st in traj.states:
        assert math.isclose(st.x, 250.0, abs_tol=1e-2)
        assert math.isclose(st.y, 300.0, abs_tol=1e-2)


def test_configurable_horizons():
    """Verify predictor respects configured horizons: 0.5s, 1.0s, 1.5s, 2.0s."""
    obj = _make_perceived_object(x=100.0, y=100.0, vx=20.0, vy=0.0, history_count=3)
    predictor = Predictor(sub_steps=4)

    for h in [0.5, 1.0, 1.5, 2.0]:
        predictor.set_horizon(h)
        traj = predictor.predict_object(obj)
        final_state = traj.final_state
        assert final_state is not None
        assert math.isclose(final_state.time_offset, h, abs_tol=1e-2)
        expected_final_x = 100.0 + 20.0 * h
        assert math.isclose(final_state.x, expected_final_x, abs_tol=1e-2)


def test_insufficient_history_marks_uncertain():
    """Verify objects with < 2 observations are marked UNCERTAIN and LOW confidence."""
    predictor = Predictor(horizon=1.0)
    # Only 1 observation -> Cannot reliably compute velocity
    obj_raw = _make_perceived_object(history_count=1, vx=0.0, vy=0.0)

    traj = predictor.predict_object(obj_raw)
    assert traj.is_uncertain is True
    assert traj.confidence_level == "LOW"
    assert traj.confidence_score <= 0.50


def test_uncertainty_growth_over_time():
    """Verify the uncertainty radius expands monotonically with time horizon."""
    predictor = Predictor(horizon=2.0, sub_steps=4)
    ped = _make_perceived_object(
        category=ObstacleType.PEDESTRIAN, x=100.0, y=100.0, vx=10.0, vy=0.0, history_count=3
    )

    traj = predictor.predict_object(ped)
    radii = [st.uncertainty_radius for st in traj.states]

    # Must be monotonically increasing
    for i in range(1, len(radii)):
        assert radii[i] > radii[i - 1], f"Radius at step {i} did not expand: {radii}"


def test_category_uncertainty_weights():
    """Verify pedestrians and cattle expand uncertainty faster than lane-bound cars."""
    predictor = Predictor(horizon=1.0, sub_steps=1)
    car = _make_perceived_object(category=ObstacleType.CAR, vx=20.0, vy=0.0, history_count=3)
    cow = _make_perceived_object(category=ObstacleType.CATTLE, vx=20.0, vy=0.0, history_count=3)

    traj_car = predictor.predict_object(car)
    traj_cow = predictor.predict_object(cow)

    assert traj_cow.final_state.uncertainty_radius > traj_car.final_state.uncertainty_radius


def test_separation_from_ground_truth():
    """Verify predictor operates strictly on PerceivedObject, not DynamicObstacle."""
    gt_obj = DynamicObstacle(id=88, category=ObstacleType.AUTORICKSHAW, x=300.0, y=200.0, speed=40.0)
    perceived = _make_perceived_object(track_id=88, category=ObstacleType.AUTORICKSHAW, x=302.0, y=201.0, vx=38.0)

    predictor = Predictor(horizon=1.0)
    traj = predictor.predict_object(perceived)

    # Output matches perceived estimates, not ground truth
    assert traj.current_x == 302.0
    assert traj.current_vx == 38.0
    assert not isinstance(traj, DynamicObstacle)


def test_data_bundle_exposure_for_collision_module():
    """Verify predicted trajectory structure exposes all required attributes for future collision checking."""
    predictor = Predictor(horizon=1.0)
    obj = _make_perceived_object(track_id=7, x=100.0, y=100.0, vx=15.0, vy=5.0)

    traj = predictor.predict_object(obj)
    assert hasattr(traj, "track_id")
    assert hasattr(traj, "category")
    assert hasattr(traj, "current_x")
    assert hasattr(traj, "current_y")
    assert hasattr(traj, "current_vx")
    assert hasattr(traj, "current_vy")
    assert hasattr(traj, "current_speed")
    assert hasattr(traj, "confidence_level")
    assert hasattr(traj, "confidence_score")
    assert hasattr(traj, "is_uncertain")
    assert hasattr(traj, "states")
    assert len(traj.states) > 0


def test_invalid_velocity_marks_uncertain():
    """Verify NaN, Inf, and physically unrealistic velocities are marked UNCERTAIN and LOW confidence."""
    predictor = Predictor(horizon=1.0)

    # 1. NaN velocity
    nan_obj = _make_perceived_object(vx=float("nan"), vy=0.0, history_count=3)
    traj_nan = predictor.predict_object(nan_obj)
    assert traj_nan.is_uncertain is True
    assert traj_nan.confidence_level == "LOW"
    assert traj_nan.confidence_score <= 0.20

    # 2. Infinite velocity
    inf_obj = _make_perceived_object(vx=float("inf"), vy=10.0, history_count=3)
    traj_inf = predictor.predict_object(inf_obj)
    assert traj_inf.is_uncertain is True
    assert traj_inf.confidence_level == "LOW"

    # 3. Physically excessive speed jump
    crazy_obj = _make_perceived_object(vx=999.0, vy=0.0, history_count=3)
    traj_crazy = predictor.predict_object(crazy_obj)
    assert traj_crazy.is_uncertain is True
    assert traj_crazy.confidence_level == "LOW"


def test_supported_road_user_categories():
    """Verify CVMM motion prediction supports all 6 required road user types."""
    predictor = Predictor(horizon=1.0)
    all_categories = [
        ObstacleType.CAR,
        ObstacleType.MOTORCYCLE,
        ObstacleType.AUTORICKSHAW,
        ObstacleType.PEDESTRIAN,
        ObstacleType.CATTLE,
        ObstacleType.BARRIER,
    ]

    for cat in all_categories:
        obj = _make_perceived_object(category=cat, vx=10.0, vy=0.0, history_count=3)
        traj = predictor.predict_object(obj)
        assert traj.category == cat
        assert len(traj.states) > 0

