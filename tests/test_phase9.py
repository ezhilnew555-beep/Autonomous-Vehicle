"""
tests/test_phase9.py
====================
Test suite for Phase 9: Performance Evaluation and Comparative Analysis.

Verifies:
1. Planner supports A*, Dijkstra, and Greedy Best-First Search under identical conditions.
2. Dijkstra expands >= A* nodes for identical optimal paths on identical grids.
3. Metric containers, units, and unavailable value marking (None / N/A, not fabricated).
4. Statistical calculations: mean, median, sample std-dev (Bessel's N-1), min, max, zero denominators.
5. Success, failure, and timeout rates with documented denominators.
6. Repeatable runner with deterministic seeds.
7. Headless dynamic simulation runner capturing realistic kinematics and safety events.
8. CSV and JSON export functionality under reports/.
9. Matplotlib chart generation generating real PNG files.
10. Benchmark UI modal card geometry and bounds.
"""

import math
import os
import tempfile
import pytest

import config
from src.grid import OccupancyGrid
from src.planner import Planner, PathResult
from src.scenarios import (
    Scenario,
    get_all_scenarios,
    create_village_scenario,
    create_cattle_scenario,
)
from src.metrics import (
    BenchmarkRunRecord,
    ScenarioMetrics,
    calculate_sample_statistics,
    calculate_benchmark_summary,
    export_runs_to_csv,
    export_summary_to_csv,
    export_metadata_to_json,
)
from src.benchmarks import (
    run_static_planner_benchmark,
    run_single_simulation_run,
    BenchmarkSuite,
)
from src.charts import (
    generate_all_performance_charts,
    plot_planning_time_comparison,
    plot_expanded_nodes_comparison,
)
from src.ui import get_benchmark_screen_card_rect


# ---------------------------------------------------------------------------
# Test 1: Algorithmic Comparison Under Identical Conditions
# ---------------------------------------------------------------------------
def test_planner_algorithm_comparison_optimal_paths():
    """Verify A* and Dijkstra find the exact same optimal path cost, but Dijkstra expands more nodes."""
    grid = OccupancyGrid()
    center_r = config.ROAD_ROW_CENTER
    start = (2, center_r)
    goal = (20, center_r)

    planner_astar = Planner(allow_diagonal=True, algorithm="astar")
    planner_dijkstra = Planner(allow_diagonal=True, algorithm="dijkstra")
    planner_gbfs = Planner(allow_diagonal=True, algorithm="gbfs")

    res_astar = planner_astar.plan(grid, start=start, goal=goal)
    res_dijkstra = planner_dijkstra.plan(grid, start=start, goal=goal)
    res_gbfs = planner_gbfs.plan(grid, start=start, goal=goal)

    assert res_astar.success is True
    assert res_dijkstra.success is True
    assert res_gbfs.success is True

    # Both A* and Dijkstra guarantee the same optimal path cost
    assert res_astar.path_cost == pytest.approx(res_dijkstra.path_cost, abs=1e-3)

    # Dijkstra explores uniformly in all directions (h=0), expanding >= nodes than A*
    assert res_dijkstra.nodes_expanded >= res_astar.nodes_expanded

    # Verify algorithm tags recorded in PathResult
    assert res_astar.algorithm == "astar"
    assert res_dijkstra.algorithm == "dijkstra"
    assert res_gbfs.algorithm == "gbfs"


def test_planner_algorithm_with_obstacles():
    """Verify algorithms navigate around obstacles identically."""
    grid = OccupancyGrid()
    center_r = config.ROAD_ROW_CENTER
    # Place obstacle on row center, col 10 (leaving lanes above/below open to detour)
    grid.add_obstacle(10, center_r)

    start = (2, center_r)
    goal = (18, center_r)

    planner_astar = Planner(allow_diagonal=True, algorithm="astar")
    planner_dijkstra = Planner(allow_diagonal=True, algorithm="dijkstra")

    res_astar = planner_astar.plan(grid, start=start, goal=goal)
    res_dijkstra = planner_dijkstra.plan(grid, start=start, goal=goal)

    assert res_astar.success is True
    assert res_dijkstra.success is True
    assert res_astar.path_cost == pytest.approx(res_dijkstra.path_cost, abs=1e-3)
    assert res_dijkstra.nodes_expanded >= res_astar.nodes_expanded


# ---------------------------------------------------------------------------
# Test 2: Metrics and Unavailable Values
# ---------------------------------------------------------------------------
def test_metrics_marking_unavailable_values():
    """Verify metrics mark unavailable values as None, rather than fabricating data."""
    rec = BenchmarkRunRecord(
        run_id="test_run_01",
        scenario_id="village_road",
        scenario_name="Village Road",
        algorithm="astar",
        seed=42,
        success=False,
        status="NO_PATH",
    )

    d = rec.to_dict()
    assert d["destination_time_s"] is None
    assert d["path_length_px"] is None
    assert d["min_obstacle_clearance_px"] is None
    assert d["actual_simulated_collisions"] == 0
    assert d["collision_risk_events"] == 0


def test_risk_events_distinct_from_actual_collisions():
    """Verify predicted collision risk events are strictly distinct from actual physical collisions."""
    sm = ScenarioMetrics()
    sm.record_conflict()
    sm.record_conflict()
    assert sm.collision_risk_events == 2
    assert sm.actual_simulated_collisions == 0

    sm.record_actual_collision()
    assert sm.actual_simulated_collisions == 1
    assert sm.collision_risk_events == 2


# ---------------------------------------------------------------------------
# Test 3: Statistical Calculations & Denominator Rules
# ---------------------------------------------------------------------------
def test_sample_statistics_calculations():
    # Empty list
    empty_stats = calculate_sample_statistics([])
    assert empty_stats["mean"] is None
    assert empty_stats["median"] is None
    assert empty_stats["std_dev"] is None
    assert empty_stats["sample_size"] == 0

    # Single value (N=1)
    single_stats = calculate_sample_statistics([42.0])
    assert single_stats["mean"] == 42.0
    assert single_stats["median"] == 42.0
    assert single_stats["std_dev"] == 0.0
    assert single_stats["min"] == 42.0
    assert single_stats["max"] == 42.0
    assert single_stats["sample_size"] == 1

    # Multiple values (N=4): [10, 20, 30, 40]
    multi_stats = calculate_sample_statistics([10.0, 20.0, 30.0, 40.0])
    assert multi_stats["mean"] == 25.0
    assert multi_stats["median"] == 25.0
    assert multi_stats["min"] == 10.0
    assert multi_stats["max"] == 40.0
    assert multi_stats["sample_size"] == 4
    # Sample variance: ((15^2 + 5^2 + 5^2 + 15^2) / 3) = 500 / 3 = 166.6667 -> std_dev ~= 12.91
    assert multi_stats["std_dev"] == pytest.approx(12.91, abs=0.01)


def test_benchmark_summary_denominators():
    """Verify summary only computes travel time on runs that actually reached destination."""
    runs = [
        BenchmarkRunRecord(
            run_id="run_1",
            scenario_id="scen_a",
            scenario_name="Scenario A",
            algorithm="astar",
            seed=1,
            success=True,
            status="COMPLETED",
            planning_time_ms=5.0,
            destination_time_s=12.4,
            expanded_nodes=40,
        ),
        BenchmarkRunRecord(
            run_id="run_2",
            scenario_id="scen_a",
            scenario_name="Scenario A",
            algorithm="astar",
            seed=2,
            success=False,
            status="TIMEOUT",
            is_timeout=True,
            planning_time_ms=6.0,
            destination_time_s=None,  # Did not reach destination!
            expanded_nodes=42,
        ),
    ]

    summary = calculate_benchmark_summary(runs)
    assert len(summary) == 1
    row = summary[0]

    assert row["total_runs"] == 2
    assert row["success_count"] == 1
    assert row["success_rate"] == 0.5
    assert row["timeout_count"] == 1
    assert row["timeout_rate"] == 0.5
    assert row["failure_count"] == 0

    # Travel time mean must ONLY come from the single successful run (12.4), not divided by 2!
    assert row["travel_time_mean_s"] == 12.4
    assert row["travel_time_sample_size"] == 1

    # Planning time includes both runs
    assert row["plan_time_mean_ms"] == 5.5


# ---------------------------------------------------------------------------
# Test 4: Static Planner Repeatable Runner
# ---------------------------------------------------------------------------
def test_static_planner_benchmark_runner():
    sc = create_village_scenario()
    runs = run_static_planner_benchmark(
        scenarios=[sc],
        algorithms=["astar", "dijkstra"],
        repetitions=2,
        seed_base=100,
    )

    assert len(runs) == 4  # 1 scenario * 2 algorithms * 2 reps
    for r in runs:
        assert r.benchmark_type == "STATIC_PLANNER"
        assert r.success is True
        assert r.planning_time_ms >= 0.0
        assert r.expanded_nodes is not None and r.expanded_nodes > 0


# ---------------------------------------------------------------------------
# Test 5: Headless Dynamic Simulation Runner
# ---------------------------------------------------------------------------
def test_single_simulation_run_execution():
    rec = run_single_simulation_run(
        scenario_factory=create_village_scenario,
        algorithm="astar",
        seed=42,
        run_index=1,
        max_steps=200,  # Run initial 10s of simulation
    )

    assert rec.benchmark_type == "DYNAMIC_SIMULATION"
    assert rec.simulation_duration_s > 0.0
    assert rec.distance_traveled_px > 0.0
    assert rec.actual_simulated_collisions == 0  # Safe navigation without crashes
    assert rec.min_obstacle_clearance_px is not None


# ---------------------------------------------------------------------------
# Test 6: CSV and JSON Exports
# ---------------------------------------------------------------------------
def test_csv_and_json_exports(tmp_path):
    runs = [
        BenchmarkRunRecord(
            run_id="run_csv_1",
            scenario_id="village_road",
            scenario_name="1. Unmarked Village Road",
            algorithm="astar",
            seed=42,
            success=True,
            status="COMPLETED",
            planning_time_ms=4.2,
            path_length_px=420.0,
            path_cost_grid=14.0,
            grid_cell_count=15,
            expanded_nodes=35,
        )
    ]

    # Test individual runs CSV export
    runs_file = os.path.join(str(tmp_path), "test_runs.csv")
    assert export_runs_to_csv(runs, runs_file) is True
    assert os.path.exists(runs_file)
    with open(runs_file, "r") as f:
        content = f.read()
        assert "run_csv_1" in content
        assert "village_road" in content

    # Test summary CSV export
    summary = calculate_benchmark_summary(runs)
    summary_file = os.path.join(str(tmp_path), "test_summary.csv")
    assert export_summary_to_csv(summary, summary_file) is True
    assert os.path.exists(summary_file)

    # Test metadata JSON export
    meta_file = os.path.join(str(tmp_path), "test_meta.json")
    meta_data = {"test_run": True, "total_runs": 1}
    assert export_metadata_to_json(meta_data, meta_file) is True
    assert os.path.exists(meta_file)


# ---------------------------------------------------------------------------
# Test 7: Performance Charts Generation
# ---------------------------------------------------------------------------
def test_chart_generation_produces_png_files(tmp_path):
    runs = [
        BenchmarkRunRecord(
            run_id=f"r_{i}_{algo}",
            scenario_id=f"scen_{i}",
            scenario_name=f"Scenario {i}",
            algorithm=algo,
            seed=42 + i,
            success=True,
            status="COMPLETED",
            planning_time_ms=2.5 + i,
            path_cost_grid=15.0 + i,
            expanded_nodes=30 + i * 5,
            destination_time_s=15.0 + i,
            collision_risk_events=1,
            actual_simulated_collisions=0,
            benchmark_type="DYNAMIC_SIMULATION",
        )
        for i in range(1, 3)
        for algo in ["astar", "dijkstra", "gbfs"]
    ]

    summary = calculate_benchmark_summary(runs)
    output_dir = str(tmp_path / "reports")

    chart_paths = generate_all_performance_charts(runs, summary, output_dir=output_dir)
    assert len(chart_paths) >= 4

    for p in chart_paths:
        assert os.path.exists(p)
        assert os.path.getsize(p) > 1000  # Valid non-empty PNG file


# ---------------------------------------------------------------------------
# Test 8: UI Modal Card Geometry
# ---------------------------------------------------------------------------
def test_benchmark_screen_card_geometry():
    card_rect = get_benchmark_screen_card_rect()
    assert card_rect.width > 600
    assert card_rect.height > 400
    assert card_rect.right <= config.SCREEN_WIDTH
    assert card_rect.bottom <= config.SCREEN_HEIGHT
