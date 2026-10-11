"""
tests/test_phase5.py
====================
Automated test suite for Phase 5: Simulated Obstacle Detection and Tracking
- Support for all simulated object categories (Car, Motorcycle, Auto-rickshaw, Pedestrian, Cattle, Barrier)
- Unique and persistent object IDs across creations and observations
- Detection range filtering (objects inside vs outside range)
- Simulated detection latency delay (observations buffer delivery)
- Simulated missing detections (stochastic dropout)
- Object movement: objects entering and leaving detection range
- Velocity and speed estimation from successive observations
- Removal of inactive objects / track pruning after timeout
- Strict separation between ground-truth objects and perceived objects
"""

import sys
import os
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from src.obstacles import ObstacleType, DynamicObstacle, reset_object_id_counter
from src.perception import PerceptionSystem, PerceivedObject, Observation


def test_supported_object_types_and_dimensions():
    """Verify all 6 Indian road user categories are supported with dimensions."""
    categories = [
        ObstacleType.CAR,
        ObstacleType.MOTORCYCLE,
        ObstacleType.AUTORICKSHAW,
        ObstacleType.PEDESTRIAN,
        ObstacleType.CATTLE,
        ObstacleType.BARRIER,
    ]

    for cat in categories:
        obj = DynamicObstacle(category=cat, x=100.0, y=200.0)
        assert obj.category == cat
        assert obj.length > 0
        assert obj.width > 0
        assert obj.is_active is True


def test_unique_persistent_object_ids():
    """Verify ground truth objects receive unique, monotonic, persistent IDs."""
    reset_object_id_counter()
    obj1 = DynamicObstacle(category=ObstacleType.CAR)
    obj2 = DynamicObstacle(category=ObstacleType.MOTORCYCLE)
    obj3 = DynamicObstacle(category=ObstacleType.CATTLE)

    assert obj1.id != obj2.id
    assert obj2.id != obj3.id
    assert obj1.id == 1
    assert obj2.id == 2
    assert obj3.id == 3


def test_detection_range_filtering():
    """Verify perception only detects objects within detection range."""
    ego_x, ego_y = 100.0, 100.0
    perception = PerceptionSystem(detection_range=150.0, detection_delay=0.0)

    # Object 1: 50px away (inside range)
    obj_inside = DynamicObstacle(id=10, category=ObstacleType.CAR, x=150.0, y=100.0)
    # Object 2: 250px away (outside range)
    obj_outside = DynamicObstacle(id=20, category=ObstacleType.PEDESTRIAN, x=350.0, y=100.0)

    tracks = perception.update(
        sim_time=0.1,
        ego_x=ego_x,
        ego_y=ego_y,
        ground_truth_objects=[obj_inside, obj_outside],
    )

    detected_ids = [t.track_id for t in tracks]
    assert 10 in detected_ids
    assert 20 not in detected_ids


def test_detection_delay_buffer():
    """Verify observations are delayed by configured sensor latency."""
    ego_x, ego_y = 100.0, 100.0
    delay = 0.20  # 200ms delay
    perception = PerceptionSystem(detection_range=200.0, detection_delay=delay)

    obj = DynamicObstacle(id=5, category=ObstacleType.AUTORICKSHAW, x=120.0, y=100.0)

    # At t = 0.05s, delivery time is 0.05 + 0.20 = 0.25s -> Should not yet be visible
    tracks_t1 = perception.update(sim_time=0.05, ego_x=ego_x, ego_y=ego_y, ground_truth_objects=[obj])
    assert len(tracks_t1) == 0

    # At t = 0.15s -> Still delayed
    tracks_t2 = perception.update(sim_time=0.15, ego_x=ego_x, ego_y=ego_y, ground_truth_objects=[obj])
    assert len(tracks_t2) == 0

    # At t = 0.25s -> Latency elapsed, observation matures
    tracks_t3 = perception.update(sim_time=0.25, ego_x=ego_x, ego_y=ego_y, ground_truth_objects=[obj])
    assert len(tracks_t3) == 1
    assert tracks_t3[0].track_id == 5


def test_missing_detections_dropout():
    """Verify stochastic sensor dropout omits detections when miss_rate > 0."""
    ego_x, ego_y = 100.0, 100.0
    # 100% miss rate
    perception_blind = PerceptionSystem(detection_range=200.0, detection_delay=0.0, miss_rate=1.0)
    obj = DynamicObstacle(id=7, category=ObstacleType.CATTLE, x=120.0, y=100.0)

    tracks = perception_blind.update(sim_time=0.1, ego_x=ego_x, ego_y=ego_y, ground_truth_objects=[obj])
    assert len(tracks) == 0


def test_object_entering_and_leaving_detection_range():
    """Verify perception tracks an object entering view, flags out-of-range when leaving, and prunes on timeout."""
    ego_x, ego_y = 100.0, 100.0
    perception = PerceptionSystem(detection_range=100.0, detection_delay=0.0, track_timeout=0.5)

    # Starts outside range
    obj = DynamicObstacle(id=99, category=ObstacleType.MOTORCYCLE, x=250.0, y=100.0)
    tracks = perception.update(0.1, ego_x, ego_y, [obj])
    assert len(tracks) == 0

    # Enters range
    obj.x = 150.0
    tracks = perception.update(0.2, ego_x, ego_y, [obj])
    assert len(tracks) == 1
    assert tracks[0].track_id == 99
    assert tracks[0].is_in_range is True

    # Leaves range
    obj.x = 300.0
    tracks = perception.update(0.3, ego_x, ego_y, [obj])
    assert len(tracks) == 1
    assert tracks[0].is_in_range is False  # Flagged out of range

    # After timeout > 0.5s with no observation, track is pruned
    tracks = perception.update(0.9, ego_x, ego_y, [obj])
    assert len(tracks) == 0


def test_velocity_and_speed_estimation():
    """Verify perception accurately estimates object velocity from successive observations."""
    ego_x, ego_y = 100.0, 100.0
    perception = PerceptionSystem(detection_range=300.0, detection_delay=0.0)

    # Object moving at true speed = 50.0 px/s along +X
    obj = DynamicObstacle(id=42, category=ObstacleType.CAR, x=100.0, y=120.0, heading=0.0, speed=50.0)

    # Observation 1 at t = 0.0s
    perception.update(0.0, ego_x, ego_y, [obj])
    track = perception.tracks[42]
    assert track.estimated_speed == 0.0  # Only 1 observation, cannot infer speed yet

    # Advance object physics over dt = 0.1s -> dx = 5.0px
    obj.update(0.1)
    perception.update(0.1, ego_x, ego_y, [obj])

    # Advance another dt = 0.1s
    obj.update(0.1)
    perception.update(0.2, ego_x, ego_y, [obj])

    # Estimated speed should converge close to 50.0 px/s
    assert math.isclose(track.estimated_speed, 50.0, abs_tol=5.0)
    assert track.estimated_vx > 40.0
    assert abs(track.estimated_vy) < 2.0


def test_removal_of_inactive_objects():
    """Verify inactive objects are skipped and pruned by perception."""
    ego_x, ego_y = 100.0, 100.0
    perception = PerceptionSystem(detection_range=200.0, detection_delay=0.0, track_timeout=0.2)

    obj = DynamicObstacle(id=15, category=ObstacleType.BARRIER, x=120.0, y=100.0, is_active=True)
    perception.update(0.1, ego_x, ego_y, [obj])
    assert len(perception.tracks) == 1

    # Deactivate object
    obj.is_active = False
    perception.update(0.4, ego_x, ego_y, [obj])
    assert len(perception.tracks) == 0


def test_ground_truth_perceived_separation():
    """Verify ground truth objects and perceived objects remain separate instances."""
    gt_obj = DynamicObstacle(id=77, category=ObstacleType.PEDESTRIAN, x=100.0, y=100.0)
    perception = PerceptionSystem(detection_range=200.0, detection_delay=0.0)

    tracks = perception.update(0.1, 100.0, 100.0, [gt_obj])
    assert len(tracks) == 1
    perceived = tracks[0]

    assert isinstance(gt_obj, DynamicObstacle)
    assert isinstance(perceived, PerceivedObject)
    assert gt_obj is not perceived
