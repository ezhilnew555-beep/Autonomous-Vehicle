"""
tests/test_smoke.py
===================
Smoke tests for Phase 1.
Verify that all core modules can be imported without error.
"""

import sys
import os

# Ensure the project root is on sys.path so "import config" and
# "from src.xxx" work regardless of how pytest is invoked.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def test_import_config():
    import config
    assert hasattr(config, "SCREEN_WIDTH")
    assert hasattr(config, "SCREEN_HEIGHT")
    assert hasattr(config, "FPS")


def test_import_vehicle():
    from src.vehicle import Vehicle
    v = Vehicle()
    assert v.label == "AV"


def test_import_obstacles():
    from src.obstacles import Obstacle, default_obstacles
    obs_list = default_obstacles()
    assert isinstance(obs_list, list)


def test_import_planner():
    from src.planner import Planner
    p = Planner()
    path = p.compute_path((0, 0), (5, 5))
    assert isinstance(path, list)


def test_import_prediction():
    from src.prediction import Predictor
    pred = Predictor()
    result = pred.predict([(1, 2)])
    assert result == [(1, 2)]


def test_import_collision():
    from src.collision import check_collision
    assert check_collision((1, 1), [(2, 2)]) is False
    assert check_collision((1, 1), [(1, 1)]) is True


def test_import_scenarios():
    from src.scenarios import Scenario, load_default_scenario
    s = load_default_scenario()
    assert s.name == "Default"


def test_import_metrics():
    from src.metrics import Metrics
    m = Metrics()
    m.reset()
    assert m.collision_count == 0
