"""
src/metrics.py
==============
Performance metrics collection, statistical analysis, and data export for Phase 9.

Units and Metric Definitions:
- planning_time_ms: CPU execution time of path planning search algorithm in milliseconds (ms).
- path_length_px: Total Euclidean length along planned waypoints in simulation pixels (px).
- path_cost_grid: Accumulated step cost on grid (grid units, straight=1.0, diagonal=sqrt(2)).
- grid_cell_count: Total number of discrete grid waypoints in planned path.
- expanded_nodes: Total number of grid states expanded from priority queue during search.
- replanning_count: Number of automatic replanning events triggered by collision supervisor.
- replanning_time_ms: Total CPU execution time spent across all replanning attempts in milliseconds (ms).
- simulation_duration_s: Total elapsed simulation physics time in seconds (s).
- destination_time_s: Simulation timestamp when ego vehicle arrived at destination (s), or None if not arrived.
- distance_traveled_px: Cumulative distance traversed by ego vehicle in pixels (px).
- average_speed_px_s: Average operating speed = distance_traveled / duration (px/s), or None if duration is 0.
- max_speed_px_s: Peak instantaneous speed achieved by ego vehicle (px/s).
- stops_count: Total count of normal pauses/stops.
- emergency_stops: Count of supervisor emergency stop transitions.
- emergency_braking_events: Count of emergency braking activations triggered by imminent collision risks.
- collision_risk_events: Count of predicted future trajectory conflicts detected (warning level).
- actual_simulated_collisions: Count of actual physical vehicle footprint intersections with obstacles (strictly 0 for safe runs).
- min_obstacle_clearance_px: Minimum Euclidean distance observed between ego vehicle bounding box and any obstacle (px), or None if no obstacles observed.
- accepted_alternative_paths: Count of candidate replanned paths accepted after safety validation.
- rejected_alternative_paths: Count of candidate replanned paths rejected due to detected hazard.
"""

from dataclasses import dataclass, field, asdict
import csv
import json
import math
import os
from pathlib import Path as FilePath
from typing import List, Dict, Any, Optional


# ---------------------------------------------------------------------------
# Individual Benchmark Run Record (Phase 9)
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkRunRecord:
    """
    Granular measurement record for an individual benchmark run.
    Preserves raw data and marks unavailable metrics as None rather than inventing data.
    """
    run_id: str
    scenario_id: str
    scenario_name: str
    algorithm: str                      # "astar", "dijkstra", "gbfs"
    seed: int
    run_index: int = 1
    benchmark_type: str = "STATIC_PLANNER"  # "STATIC_PLANNER" or "DYNAMIC_SIMULATION"
    success: bool = False
    status: str = "NOT_STARTED"         # "COMPLETED", "TIMEOUT", "NO_SAFE_PATH", "COLLISION", "FAILED"
    termination_reason: str = ""

    # Static & Planning Metrics
    planning_time_ms: float = 0.0
    path_length_px: Optional[float] = None
    path_cost_grid: Optional[float] = None
    grid_cell_count: Optional[int] = None
    expanded_nodes: Optional[int] = None

    # Dynamic Simulation Metrics (None for static planner runs)
    simulation_duration_s: Optional[float] = None
    destination_time_s: Optional[float] = None
    distance_traveled_px: Optional[float] = None
    average_speed_px_s: Optional[float] = None
    max_speed_px_s: Optional[float] = None
    replanning_count: int = 0
    replanning_time_ms: float = 0.0
    accepted_alternative_paths: int = 0
    rejected_alternative_paths: int = 0
    stops_count: int = 0
    emergency_stops: int = 0
    emergency_braking_events: int = 0
    collision_risk_events: int = 0
    actual_simulated_collisions: int = 0
    min_obstacle_clearance_px: Optional[float] = None
    is_timeout: bool = False
    is_failed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a JSON/CSV serializable dictionary."""
        d = asdict(self)
        # Format floating points nicely
        for k, v in d.items():
            if isinstance(v, float) and v is not None:
                d[k] = round(v, 3)
        return d


# ---------------------------------------------------------------------------
# Scenario Metrics Container (Phases 7, 8 & 9)
# ---------------------------------------------------------------------------
@dataclass
class ScenarioMetrics:
    """Live telemetry and audit container for a scenario run."""
    scenario_name: str = "Default"
    random_seed: int = 42
    status: str = "NOT_STARTED"
    simulation_duration: float = 0.0
    path_length: float = 0.0
    planning_time_ms: float = 0.0
    replanning_events: int = 0
    emergency_stops: int = 0
    collision_risk_events: int = 0
    min_obstacle_clearance: float = float("inf")
    destination_reached: bool = False
    termination_reason: str = ""

    # Phase 9 Extensions
    grid_cell_count: Optional[int] = None
    expanded_nodes: Optional[int] = None
    replanning_time_ms: float = 0.0
    destination_time_s: Optional[float] = None
    distance_traveled_px: float = 0.0
    average_speed_px_s: Optional[float] = None
    max_speed_px_s: float = 0.0
    stops: int = 0
    emergency_braking_events: int = 0
    actual_simulated_collisions: int = 0
    accepted_alternative_paths: int = 0
    rejected_alternative_paths: int = 0
    is_timeout: bool = False
    is_failed: bool = False

    def update_clearance(self, clearance: float) -> None:
        """Update minimum observed distance between vehicle and obstacles."""
        if clearance > 0.0 and clearance < self.min_obstacle_clearance:
            self.min_obstacle_clearance = round(clearance, 1)

    def record_replan(self) -> None:
        self.replanning_events += 1

    def record_emergency_stop(self) -> None:
        self.emergency_stops += 1
        self.emergency_braking_events += 1

    def record_conflict(self) -> None:
        self.collision_risk_events += 1

    def record_actual_collision(self) -> None:
        self.actual_simulated_collisions += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_name": self.scenario_name,
            "random_seed": self.random_seed,
            "status": self.status,
            "simulation_duration_s": round(self.simulation_duration, 2),
            "destination_time_s": round(self.destination_time_s, 2) if self.destination_time_s is not None else None,
            "path_length_px": round(self.path_length, 1),
            "planning_time_ms": round(self.planning_time_ms, 2),
            "expanded_nodes": self.expanded_nodes,
            "replanning_events": self.replanning_events,
            "replanning_time_ms": round(self.replanning_time_ms, 2),
            "emergency_stops": self.emergency_stops,
            "emergency_braking_events": self.emergency_braking_events,
            "collision_risk_events": self.collision_risk_events,
            "actual_simulated_collisions": self.actual_simulated_collisions,
            "min_obstacle_clearance_px": self.min_obstacle_clearance if self.min_obstacle_clearance != float("inf") else None,
            "distance_traveled_px": round(self.distance_traveled_px, 1),
            "average_speed_px_s": round(self.average_speed_px_s, 2) if self.average_speed_px_s is not None else None,
            "max_speed_px_s": round(self.max_speed_px_s, 2),
            "accepted_alternative_paths": self.accepted_alternative_paths,
            "rejected_alternative_paths": self.rejected_alternative_paths,
            "destination_reached": self.destination_reached,
            "termination_reason": self.termination_reason,
        }

    def summary(self) -> str:
        clearance_str = f"{self.min_obstacle_clearance:.1f}px" if self.min_obstacle_clearance != float("inf") else "N/A"
        return (
            f"Scenario: {self.scenario_name} (Seed: {self.random_seed})\n"
            f"Status: {self.status} | Time: {self.simulation_duration:.1f}s\n"
            f"Arrived: {self.destination_reached} | Reason: {self.termination_reason or 'Nominal'}\n"
            f"Replans: {self.replanning_events} | E-Stops: {self.emergency_stops} | Conflicts: {self.collision_risk_events} | Collisions: {self.actual_simulated_collisions}\n"
            f"Min Clearance: {clearance_str} | Path Length: {self.path_length:.1f}px"
        )


# ---------------------------------------------------------------------------
# Legacy Metrics Container (Phase 1 Smoke Test Compatibility)
# ---------------------------------------------------------------------------
@dataclass
class Metrics:
    """Container for legacy simulation performance metrics (Phase 1 compatibility)."""
    path_lengths: List[float] = field(default_factory=list)
    computation_times: List[float] = field(default_factory=list)
    collision_count: int = 0
    active_scenario_metrics: Optional[ScenarioMetrics] = None

    def reset(self) -> None:
        """Clear all recorded metrics."""
        self.path_lengths.clear()
        self.computation_times.clear()
        self.collision_count = 0
        self.active_scenario_metrics = None

    def summary(self) -> str:
        if self.active_scenario_metrics:
            return self.active_scenario_metrics.summary()
        return "Metrics tracker active."


# ---------------------------------------------------------------------------
# Statistical Calculations
# ---------------------------------------------------------------------------
def calculate_sample_statistics(values: List[float]) -> Dict[str, Optional[float]]:
    """
    Calculate sample mean, median, sample standard deviation, min, and max.

    Rules for robust analysis:
    - Empty list: all statistical indicators return None.
    - Single item (N=1): mean=median=min=max=val, std_dev=0.0.
    - N >= 2: sample standard deviation using Bessel's correction (N - 1 denominator).
    """
    valid_vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    n = len(valid_vals)

    if n == 0:
        return {
            "mean": None,
            "median": None,
            "std_dev": None,
            "min": None,
            "max": None,
            "sample_size": 0,
        }

    sorted_vals = sorted(valid_vals)
    mean_val = sum(sorted_vals) / float(n)

    if n % 2 == 1:
        median_val = sorted_vals[n // 2]
    else:
        median_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0

    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]

    if n == 1:
        std_dev = 0.0
    else:
        variance = sum((x - mean_val) ** 2 for x in sorted_vals) / float(n - 1)
        std_dev = math.sqrt(variance)

    return {
        "mean": round(mean_val, 3),
        "median": round(median_val, 3),
        "std_dev": round(std_dev, 3),
        "min": round(min_val, 3),
        "max": round(max_val, 3),
        "sample_size": n,
    }


def calculate_benchmark_summary(runs: List[BenchmarkRunRecord]) -> List[Dict[str, Any]]:
    """
    Aggregate benchmark runs grouped by (scenario_id, algorithm).
    Computes success rates, failure rates, and sample distributions
    with explicit documented denominators.
    """
    if not runs:
        return []

    # Group runs by (scenario_id, algorithm)
    groups: Dict[Tuple[str, str], List[BenchmarkRunRecord]] = {}
    for r in runs:
        key = (r.scenario_id, r.algorithm)
        groups.setdefault(key, []).append(r)

    summary_list = []
    for (scen_id, algo), group in sorted(groups.items()):
        total_runs = len(group)
        scen_name = group[0].scenario_name if group else scen_id

        # Categorical counts
        success_runs = [r for r in group if r.success]
        timeout_runs = [r for r in group if r.is_timeout or r.status == "TIMEOUT"]
        failed_runs  = [r for r in group if r.is_failed or r.status in ("FAILED", "NO_SAFE_PATH", "COLLISION")]

        success_count = len(success_runs)
        timeout_count = len(timeout_runs)
        fail_count    = len(failed_runs)

        success_rate = round(success_count / float(total_runs), 3) if total_runs > 0 else 0.0
        timeout_rate = round(timeout_count / float(total_runs), 3) if total_runs > 0 else 0.0
        failure_rate = round(fail_count / float(total_runs), 3) if total_runs > 0 else 0.0

        # Numeric distributions
        plan_times = [r.planning_time_ms for r in group if r.planning_time_ms is not None]
        path_costs = [r.path_cost_grid for r in group if r.path_cost_grid is not None]
        path_lens  = [r.path_length_px for r in group if r.path_length_px is not None]
        nodes_exp  = [float(r.expanded_nodes) for r in group if r.expanded_nodes is not None]

        # Travel time: ONLY computed on runs that safely reached the destination!
        travel_times = [r.destination_time_s for r in success_runs if r.destination_time_s is not None]

        # Dynamic metrics
        replan_cnts  = [float(r.replanning_count) for r in group]
        clearances   = [r.min_obstacle_clearance_px for r in group if r.min_obstacle_clearance_px is not None]
        risk_events  = sum(r.collision_risk_events for r in group)
        collisions   = sum(r.actual_simulated_collisions for r in group)
        e_stops      = sum(r.emergency_stops for r in group)

        pt_stats = calculate_sample_statistics(plan_times)
        pc_stats = calculate_sample_statistics(path_costs)
        pl_stats = calculate_sample_statistics(path_lens)
        ne_stats = calculate_sample_statistics(nodes_exp)
        tt_stats = calculate_sample_statistics(travel_times)
        cl_stats = calculate_sample_statistics(clearances)
        rp_stats = calculate_sample_statistics(replan_cnts)

        summary_list.append({
            "scenario_id": scen_id,
            "scenario_name": scen_name,
            "algorithm": algo,
            "total_runs": total_runs,
            "success_count": success_count,
            "success_rate": success_rate,
            "timeout_count": timeout_count,
            "timeout_rate": timeout_rate,
            "failure_count": fail_count,
            "failure_rate": failure_rate,

            # Planning Time (ms)
            "plan_time_mean_ms": pt_stats["mean"],
            "plan_time_median_ms": pt_stats["median"],
            "plan_time_std_dev_ms": pt_stats["std_dev"],
            "plan_time_min_ms": pt_stats["min"],
            "plan_time_max_ms": pt_stats["max"],

            # Expanded Nodes
            "nodes_expanded_mean": ne_stats["mean"],
            "nodes_expanded_median": ne_stats["median"],
            "nodes_expanded_std_dev": ne_stats["std_dev"],
            "nodes_expanded_min": ne_stats["min"],
            "nodes_expanded_max": ne_stats["max"],

            # Path Cost / Length
            "path_cost_mean": pc_stats["mean"],
            "path_length_px_mean": pl_stats["mean"],

            # Travel Time (s) [Denominator: Successful destination arrivals only]
            "travel_time_mean_s": tt_stats["mean"],
            "travel_time_median_s": tt_stats["median"],
            "travel_time_std_dev_s": tt_stats["std_dev"],
            "travel_time_sample_size": tt_stats["sample_size"],

            # Dynamic & Safety Totals
            "replanning_count_mean": rp_stats["mean"],
            "total_emergency_stops": e_stops,
            "total_collision_risk_events": risk_events,
            "total_actual_collisions": collisions,
            "min_clearance_mean_px": cl_stats["mean"],
        })

    return summary_list


# ---------------------------------------------------------------------------
# Exporters (CSV & JSON)
# ---------------------------------------------------------------------------
def export_runs_to_csv(runs: List[BenchmarkRunRecord], filepath: str) -> bool:
    """Export individual benchmark run records to a CSV file."""
    if not runs:
        return False

    try:
        path = FilePath(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = [r.to_dict() for r in runs]
        fieldnames = list(rows[0].keys())

        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        return True
    except Exception as exc:
        print(f"[Error] Failed to export runs to CSV ({filepath}): {exc}")
        return False


def export_summary_to_csv(summary_rows: List[Dict[str, Any]], filepath: str) -> bool:
    """Export aggregated summary statistics to a CSV file."""
    if not summary_rows:
        return False

    try:
        path = FilePath(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(summary_rows[0].keys())

        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        return True
    except Exception as exc:
        print(f"[Error] Failed to export summary to CSV ({filepath}): {exc}")
        return False


def export_metadata_to_json(metadata: Dict[str, Any], filepath: str) -> bool:
    """Export benchmark configuration and run metadata to a JSON file."""
    try:
        path = FilePath(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, mode="w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        return True
    except Exception as exc:
        print(f"[Error] Failed to export metadata to JSON ({filepath}): {exc}")
        return False
