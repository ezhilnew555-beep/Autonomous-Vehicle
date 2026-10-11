"""
src/benchmarks.py
=================
Repeatable benchmark runner for Phase 9 performance evaluation.

Supports:
1. Static Planner Benchmarks: Controlled comparison of A*, Dijkstra, and GBFS
   under identical grid boundaries, endpoints, movement rules, and obstacle configurations.
2. Full Dynamic Simulation Benchmarks: End-to-end evaluation with kinematics,
   dynamic obstacles, perception delays, motion prediction, safety supervisor,
   and automatic replanning.
3. Complete reset between runs and deterministic random seeds.
4. Exporting data to CSV, JSON metadata, and performance charts.
"""

import copy
import datetime
import math
import os
import platform
import time
from typing import List, Dict, Any, Optional, Callable, Tuple

import config
from src.grid import OccupancyGrid
from src.obstacles import ObstacleType, DynamicObstacle, reset_object_id_counter
from src.planner import Planner, PathResult
from src.vehicle import Vehicle, VehicleState, grid_to_world, world_to_grid
from src.perception import PerceptionSystem
from src.prediction import Predictor
from src.collision import SafetySupervisor, SafetyState, evaluate_path_conflicts
from src.scenarios import (
    Scenario,
    ScheduledEvent,
    get_all_scenarios,
    get_scenario_by_id,
    create_village_scenario,
    create_intersection_scenario,
    create_merge_scenario,
    create_market_scenario,
    create_cattle_scenario,
)
from src.metrics import (
    BenchmarkRunRecord,
    calculate_benchmark_summary,
    export_runs_to_csv,
    export_summary_to_csv,
    export_metadata_to_json,
)


# ---------------------------------------------------------------------------
# 1. Static Planner Benchmark (A* vs Dijkstra vs GBFS)
# ---------------------------------------------------------------------------
def run_static_planner_benchmark(
    scenarios: Optional[List[Scenario]] = None,
    algorithms: Optional[List[str]] = None,
    repetitions: int = 3,
    seed_base: int = 42,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[BenchmarkRunRecord]:
    """
    Run controlled algorithmic path planning benchmarks under equivalent conditions:
    - Same road masks and obstacle positions.
    - Same start and goal endpoints.
    - Same movement constraints (8-directional with corner-cutting prevention).
    - Tests: A*, Dijkstra (h=0), Greedy Best-First Search (f=h).
    """
    if scenarios is None:
        scenarios = get_all_scenarios(seed_base=seed_base)
    if algorithms is None:
        algorithms = ["astar", "dijkstra", "gbfs"]

    total_tasks = len(scenarios) * len(algorithms) * repetitions
    task_idx = 0
    records: List[BenchmarkRunRecord] = []

    for sc in scenarios:
        for algo in algorithms:
            for rep in range(1, repetitions + 1):
                task_idx += 1
                run_seed = seed_base + (rep - 1) * 17
                if progress_callback:
                    progress_callback(task_idx, total_tasks, f"Planning {sc.name} [{algo.upper()}] run {rep}/{repetitions}")

                # Setup fresh grid for this scenario
                grid = OccupancyGrid()
                if sc.road_mask is not None:
                    grid.set_road_mask(sc.road_mask)
                grid.start_pos = sc.start_pos
                grid.dest_pos = sc.dest_pos

                for col, row in sc.obstacle_positions:
                    grid.add_obstacle(col, row)

                # Execute planner with selected algorithm
                planner = Planner(allow_diagonal=True, algorithm=algo)
                res = planner.plan(grid, start=sc.start_pos, goal=sc.dest_pos, algorithm=algo)

                run_id = f"static_{sc.id}_{algo}_s{run_seed}_r{rep}"
                record = BenchmarkRunRecord(
                    run_id=run_id,
                    scenario_id=sc.id,
                    scenario_name=sc.name,
                    algorithm=algo,
                    seed=run_seed,
                    run_index=rep,
                    benchmark_type="STATIC_PLANNER",
                    success=res.success,
                    status="COMPLETED" if res.success else "NO_PATH",
                    termination_reason=res.message if not res.success else "GOAL_REACHED",
                    planning_time_ms=res.execution_time_ms,
                    path_length_px=res.path_cost * config.GRID_CELL_SIZE if res.success else None,
                    path_cost_grid=res.path_cost if res.success else None,
                    grid_cell_count=res.cell_count if res.success else None,
                    expanded_nodes=res.nodes_expanded,
                )
                records.append(record)

    return records


def check_oriented_boxes_intersect(
    x1: float, y1: float, h1: float, l1: float, w1: float,
    x2: float, y2: float, h2: float, l2: float, w2: float,
) -> bool:
    """
    Separating Axis Theorem (SAT) test for two 2D oriented bounding boxes.
    Returns True strictly if boxes physically overlap.
    """
    # Quick circle rejection test with bounding radius
    r1 = math.hypot(l1 * 0.5, w1 * 0.5)
    r2 = math.hypot(l2 * 0.5, w2 * 0.5)
    if math.hypot(x1 - x2, y1 - y2) > (r1 + r2):
        return False

    cos1, sin1 = math.cos(h1), math.sin(h1)
    cos2, sin2 = math.cos(h2), math.sin(h2)

    # 4 normal axes
    axes = [
        (cos1, sin1),
        (-sin1, cos1),
        (cos2, sin2),
        (-sin2, cos2),
    ]

    corners1 = [
        (x1 + dx * cos1 - dy * sin1, y1 + dx * sin1 + dy * cos1)
        for dx, dy in [(-l1*0.5, -w1*0.5), (l1*0.5, -w1*0.5), (l1*0.5, w1*0.5), (-l1*0.5, w1*0.5)]
    ]
    corners2 = [
        (x2 + dx * cos2 - dy * sin2, y2 + dx * sin2 + dy * cos2)
        for dx, dy in [(-l2*0.5, -w2*0.5), (l2*0.5, -w2*0.5), (l2*0.5, w2*0.5), (-l2*0.5, w2*0.5)]
    ]

    for ax, ay in axes:
        p1 = [c[0] * ax + c[1] * ay for c in corners1]
        p2 = [c[0] * ax + c[1] * ay for c in corners2]
        if min(p1) > max(p2) or min(p2) > max(p1):
            return False

    return True


# ---------------------------------------------------------------------------
# 2. Dynamic Simulation Benchmark (End-to-End Autonomous Driving)
# ---------------------------------------------------------------------------
def run_single_simulation_run(
    scenario_factory: Callable[[int], Scenario],
    algorithm: str = "astar",
    seed: int = 42,
    run_index: int = 1,
    step_dt: float = 0.05,
    max_steps: int = 1200,  # 60 seconds max at dt=0.05
) -> BenchmarkRunRecord:
    """
    Execute a single, completely headless simulation run from start to finish.
    Simulates kinematics, perception, prediction, collision checking,
    and automatic replanning.
    """
    reset_object_id_counter()
    sc = scenario_factory(seed)

    # 1. Environment & Planner setup
    grid = OccupancyGrid()
    if sc.road_mask is not None:
        grid.set_road_mask(sc.road_mask)
    grid.start_pos = sc.start_pos
    grid.dest_pos = sc.dest_pos

    for col, row in sc.obstacle_positions:
        grid.add_obstacle(col, row)

    planner = Planner(allow_diagonal=True, algorithm=algorithm)

    # 2. Vehicle setup
    start_c, start_r = sc.start_pos
    dest_c, dest_r = sc.dest_pos
    veh = Vehicle(start_col=start_c, start_row=start_r)

    # 3. Perception & Safety Systems
    perception = PerceptionSystem(
        detection_range=config.DETECTION_RANGE_OPTIONS[1],
        detection_delay=config.DETECTION_DELAY_OPTIONS[1],
    )
    predictor = Predictor(horizon=config.PREDICTION_HORIZONS[config.DEFAULT_HORIZON_INDEX])
    supervisor = SafetySupervisor()

    # Dynamic obstacles
    ground_truth = [copy.deepcopy(o) for o in sc.dynamic_obstacles]
    sc.reset_events()

    # Initial route planning
    init_plan_start = time.perf_counter()
    path_res = planner.plan(grid, start=sc.start_pos, goal=sc.dest_pos, algorithm=algorithm)
    init_plan_ms = (time.perf_counter() - init_plan_start) * 1000.0

    if not path_res.success:
        return BenchmarkRunRecord(
            run_id=f"dyn_{sc.id}_{algorithm}_s{seed}_r{run_index}",
            scenario_id=sc.id,
            scenario_name=sc.name,
            algorithm=algorithm,
            seed=seed,
            run_index=run_index,
            benchmark_type="DYNAMIC_SIMULATION",
            success=False,
            status="NO_SAFE_PATH",
            termination_reason="NO_INITIAL_ROUTE",
            planning_time_ms=init_plan_ms,
            is_failed=True,
        )

    veh.set_path(path_res.path)
    veh.start()

    # Tracking variables
    sim_time = 0.0
    distance_traveled = 0.0
    prev_x, prev_y = veh.x, veh.y
    max_observed_speed = 0.0
    min_clearance = float("inf")
    actual_collisions = 0
    run_status = "RUNNING"
    termination_reason = ""
    replan_total_time_ms = 0.0
    accepted_paths = 0
    rejected_paths = 0

    # Simulation Physics Loop
    for _ in range(max_steps):
        sim_time += step_dt

        # Check timeout
        if sim_time >= sc.timeout:
            run_status = "TIMEOUT"
            termination_reason = f"TIMEOUT_{sc.timeout:.0f}s"
            break

        # A. Scheduled dynamic events
        due = sc.get_due_events(sim_time)
        for ev in due:
            ev.has_triggered = True
            if ev.action_type == "ACTIVATE_MOVEMENT":
                tid = ev.payload.get("obstacle_id")
                for obj in ground_truth:
                    if tid is None or obj.id == tid or (tid == 77 and obj.category == ObstacleType.CATTLE):
                        if "speed" in ev.payload:
                            obj.speed = ev.payload["speed"]
                        if "heading" in ev.payload:
                            obj.heading = ev.payload["heading"]
                        if "bound_y" in ev.payload:
                            obj.bound_y = ev.payload["bound_y"]
                        break
            elif ev.action_type == "SPAWN_OBSTACLE":
                if "obstacle" in ev.payload:
                    ground_truth.append(copy.deepcopy(ev.payload["obstacle"]))

        # B. Dynamic obstacles physics
        for obj in ground_truth:
            obj.update(step_dt)

        # C. Perception & Motion Prediction
        perceived = perception.update(
            sim_time=sim_time,
            ego_x=veh.x,
            ego_y=veh.y,
            ground_truth_objects=ground_truth,
        )
        trajectories = predictor.predict_all(perceived)

        # D. Safety Supervisor update (measure replanning time)
        replan_before = supervisor.replan_count
        t_sup_start = time.perf_counter()
        sup_state = supervisor.update(
            sim_time=sim_time,
            dt=step_dt,
            vehicle=veh,
            grid=grid,
            planner=planner,
            perceived_objects=perception.perceived_objects,
            predicted_trajectories=trajectories,
        )
        if supervisor.replan_count > replan_before:
            replan_total_time_ms += (time.perf_counter() - t_sup_start) * 1000.0
            accepted_paths += 1

        veh.speed_multiplier = supervisor.speed_multiplier

        # E. Vehicle kinematics update
        veh.update(step_dt, grid)

        step_dist = math.hypot(veh.x - prev_x, veh.y - prev_y)
        distance_traveled += step_dist
        prev_x, prev_y = veh.x, veh.y

        if veh.speed > max_observed_speed:
            max_observed_speed = veh.speed

        # F. Clearance & Physical Collision Verification
        # Check distance to all ground truth obstacles
        for obj in ground_truth:
            d = math.hypot(obj.x - veh.x, obj.y - veh.y)
            if d < min_clearance:
                min_clearance = round(d, 1)

            # Check actual footprint overlap (exact 2D Oriented Bounding Box SAT test)
            if check_oriented_boxes_intersect(
                veh.x, veh.y, veh.heading, config.VEHICLE_LENGTH, config.VEHICLE_WIDTH,
                obj.x, obj.y, obj.heading, obj.length, obj.width,
            ):
                actual_collisions += 1
                run_status = "COLLISION"
                termination_reason = f"COLLISION_WITH_{obj.category.value.upper()}"
                break

        if run_status == "COLLISION":
            break

        # G. Destination Arrival Check
        veh_c, veh_r = veh.grid_col, veh.grid_row
        if veh.state == VehicleState.ARRIVED or (veh_c == dest_c and veh_r == dest_r):
            run_status = "COMPLETED"
            termination_reason = "DESTINATION_REACHED"
            break

        # H. No Safe Path Check
        if sup_state == SafetyState.NO_SAFE_PATH:
            run_status = "NO_SAFE_PATH"
            termination_reason = "BLOCKED_NO_SAFE_PATH"
            break

    # Determine final success
    is_success = (run_status == "COMPLETED")
    avg_speed = (distance_traveled / sim_time) if sim_time > 0.0 else None

    return BenchmarkRunRecord(
        run_id=f"dyn_{sc.id}_{algorithm}_s{seed}_r{run_index}",
        scenario_id=sc.id,
        scenario_name=sc.name,
        algorithm=algorithm,
        seed=seed,
        run_index=run_index,
        benchmark_type="DYNAMIC_SIMULATION",
        success=is_success,
        status=run_status,
        termination_reason=termination_reason,
        planning_time_ms=round(init_plan_ms, 2),
        path_length_px=path_res.path_cost * config.GRID_CELL_SIZE,
        path_cost_grid=path_res.path_cost,
        grid_cell_count=path_res.cell_count,
        expanded_nodes=path_res.nodes_expanded,
        simulation_duration_s=round(sim_time, 2),
        destination_time_s=round(sim_time, 2) if is_success else None,
        distance_traveled_px=round(distance_traveled, 1),
        average_speed_px_s=round(avg_speed, 2) if avg_speed is not None else None,
        max_speed_px_s=round(max_observed_speed, 1),
        replanning_count=supervisor.replan_count,
        replanning_time_ms=round(replan_total_time_ms, 2),
        accepted_alternative_paths=accepted_paths,
        rejected_alternative_paths=rejected_paths,
        stops_count=1 if not is_success else 0,
        emergency_stops=supervisor.emergency_stop_count,
        emergency_braking_events=supervisor.emergency_stop_count,
        collision_risk_events=len(supervisor.conflict_history),
        actual_simulated_collisions=actual_collisions,
        min_obstacle_clearance_px=min_clearance if min_clearance != float("inf") else None,
        is_timeout=(run_status == "TIMEOUT"),
        is_failed=(run_status in ("COLLISION", "NO_SAFE_PATH", "FAILED")),
    )


def run_dynamic_simulation_benchmark(
    scenario_ids: Optional[List[str]] = None,
    algorithms: Optional[List[str]] = None,
    repetitions: int = 1,
    seed_base: int = 42,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> List[BenchmarkRunRecord]:
    """Execute dynamic simulation benchmarks across selected scenarios and algorithms."""
    scenario_map = {
        "village_road": create_village_scenario,
        "urban_intersection": create_intersection_scenario,
        "highway_merge": create_merge_scenario,
        "dense_market": create_market_scenario,
        "cattle_crossing": create_cattle_scenario,
    }

    if scenario_ids is None:
        scenario_ids = list(scenario_map.keys())
    if algorithms is None:
        algorithms = ["astar"]

    total_tasks = len(scenario_ids) * len(algorithms) * repetitions
    task_idx = 0
    records: List[BenchmarkRunRecord] = []

    for sid in scenario_ids:
        factory = scenario_map.get(sid)
        if not factory:
            continue

        for algo in algorithms:
            for rep in range(1, repetitions + 1):
                task_idx += 1
                seed = seed_base + (rep - 1) * 23
                if progress_callback:
                    progress_callback(task_idx, total_tasks, f"Simulating {sid} [{algo.upper()}] run {rep}/{repetitions}")

                rec = run_single_simulation_run(
                    scenario_factory=factory,
                    algorithm=algo,
                    seed=seed,
                    run_index=rep,
                )
                records.append(rec)

    return records


# ---------------------------------------------------------------------------
# 3. Comprehensive Benchmark Suite
# ---------------------------------------------------------------------------
class BenchmarkSuite:
    """Coordinates benchmark execution, data collection, and report generation."""

    def __init__(self, output_dir: str = config.REPORTS_DIR) -> None:
        self.output_dir = output_dir
        self.static_runs: List[BenchmarkRunRecord] = []
        self.dynamic_runs: List[BenchmarkRunRecord] = []
        self.last_run_timestamp: Optional[str] = None

    @property
    def all_runs(self) -> List[BenchmarkRunRecord]:
        return self.static_runs + self.dynamic_runs

    def run_benchmarks(
        self,
        include_static: bool = True,
        include_dynamic: bool = True,
        static_algorithms: Optional[List[str]] = None,
        dynamic_algorithms: Optional[List[str]] = None,
        repetitions: int = config.BENCHMARK_DEFAULT_REPS,
        seed_base: int = config.BENCHMARK_DEFAULT_SEED,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> List[BenchmarkRunRecord]:
        """Execute full benchmark suite."""
        self.static_runs.clear()
        self.dynamic_runs.clear()
        self.last_run_timestamp = datetime.datetime.now().isoformat()

        if include_static:
            if static_algorithms is None:
                static_algorithms = ["astar", "dijkstra", "gbfs"]
            self.static_runs = run_static_planner_benchmark(
                algorithms=static_algorithms,
                repetitions=repetitions,
                seed_base=seed_base,
                progress_callback=progress_callback,
            )

        if include_dynamic:
            if dynamic_algorithms is None:
                dynamic_algorithms = ["astar"]
            self.dynamic_runs = run_dynamic_simulation_benchmark(
                algorithms=dynamic_algorithms,
                repetitions=max(1, min(repetitions, 3)),
                seed_base=seed_base,
                progress_callback=progress_callback,
            )

        return self.all_runs

    def get_summary(self) -> List[Dict[str, Any]]:
        """Compute aggregated statistical summary over all runs."""
        return calculate_benchmark_summary(self.all_runs)

    def export_all(self) -> Dict[str, str]:
        """
        Export individual runs to CSV, summary to CSV, metadata to JSON,
        and generate all performance charts.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exported_files: Dict[str, str] = {}

        # 1. Export CSV of individual runs
        runs_csv = os.path.join(self.output_dir, "benchmark_runs.csv")
        if export_runs_to_csv(self.all_runs, runs_csv):
            exported_files["runs_csv"] = runs_csv

        # 2. Export CSV of aggregated summary
        summary = self.get_summary()
        summary_csv = os.path.join(self.output_dir, "benchmark_summary.csv")
        if export_summary_to_csv(summary, summary_csv):
            exported_files["summary_csv"] = summary_csv

        # 3. Export JSON metadata
        metadata = {
            "timestamp": self.last_run_timestamp or datetime.datetime.now().isoformat(),
            "total_runs": len(self.all_runs),
            "static_runs_count": len(self.static_runs),
            "dynamic_runs_count": len(self.dynamic_runs),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "configuration": {
                "grid_size": f"{config.GRID_COLS}x{config.GRID_ROWS}",
                "cell_size": config.GRID_CELL_SIZE,
                "detection_range": config.DETECTION_RANGE_OPTIONS[1],
                "prediction_horizon": config.PREDICTION_HORIZONS[config.DEFAULT_HORIZON_INDEX],
                "caution_speed_factor": config.CAUTION_SPEED_FACTOR,
                "safety_margin_base": config.SAFETY_MARGIN_BASE,
            },
        }
        meta_json = os.path.join(self.output_dir, "benchmark_metadata.json")
        if export_metadata_to_json(metadata, meta_json):
            exported_files["metadata_json"] = meta_json

        # 4. Generate Performance Charts (Phase 9)
        try:
            from src.charts import generate_all_performance_charts
            chart_files = generate_all_performance_charts(self.all_runs, summary, output_dir=self.output_dir)
            for idx, cpath in enumerate(chart_files):
                exported_files[f"chart_{idx+1}"] = cpath
        except Exception as exc:
            print(f"[Warning] Chart generation encountered error: {exc}")

        return exported_files
