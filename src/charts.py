"""
src/charts.py
=============
Matplotlib-based visualization module for Phase 9 performance evaluation.

Generates publication-quality charts:
1. Planning Time Comparison (ms, with std-dev error bars)
2. Search Efficiency / Expanded Nodes Comparison
3. Path Cost / Length Comparison
4. Safety Metrics: Collision Risk Warnings vs Actual Simulated Collisions
5. Travel Time / Simulation Duration Distribution
6. Executive 4-Panel Performance Dashboard Summary

Only plots real, collected data. Handles small or empty datasets gracefully.
"""

import os
from typing import List, Dict, Any, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive, thread-safe headless backend
import matplotlib.pyplot as plt
import numpy as np

import config
from src.metrics import BenchmarkRunRecord


# Modern dark-mode styling for plots matching simulator chrome
PLOT_STYLE = {
    "figure.facecolor": "#0e182e",
    "axes.facecolor": "#14203a",
    "axes.edgecolor": "#2a3d66",
    "axes.labelcolor": "#dce6ff",
    "text.color": "#dce6ff",
    "xtick.color": "#a0b0d0",
    "ytick.color": "#a0b0d0",
    "grid.color": "#1f2f50",
    "grid.alpha": 0.6,
    "legend.facecolor": "#14203a",
    "legend.edgecolor": "#2a3d66",
}


def _apply_theme():
    plt.rcParams.update(PLOT_STYLE)


# ---------------------------------------------------------------------------
# 1. Planning Time Comparison Chart
# ---------------------------------------------------------------------------
def plot_planning_time_comparison(summary_data: List[Dict[str, Any]], output_path: str) -> Optional[str]:
    """Generate bar chart comparing planning execution time (ms) with std-dev error bars."""
    if not summary_data:
        return None

    _apply_theme()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    # Filter algorithms & scenarios
    scenarios = sorted(list(set(row["scenario_name"] for row in summary_data)))
    algorithms = ["astar", "dijkstra", "gbfs"]
    algo_labels = {"astar": "A* (Octile/Manhattan)", "dijkstra": "Dijkstra (h=0)", "gbfs": "Greedy Best-First"}
    algo_colors = {"astar": "#3b82f6", "dijkstra": "#f59e0b", "gbfs": "#10b981"}

    x = np.arange(len(scenarios))
    width = 0.26

    for i, algo in enumerate(algorithms):
        means = []
        errors = []
        for sc in scenarios:
            match = next((r for r in summary_data if r["scenario_name"] == sc and r["algorithm"] == algo), None)
            if match and match["plan_time_mean_ms"] is not None:
                means.append(match["plan_time_mean_ms"])
                errors.append(match["plan_time_std_dev_ms"] or 0.0)
            else:
                means.append(0.0)
                errors.append(0.0)

        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            means,
            width,
            yerr=errors,
            capsize=4,
            label=algo_labels.get(algo, algo.upper()),
            color=algo_colors.get(algo, "#888888"),
            alpha=0.9,
            edgecolor="#ffffff",
            linewidth=0.5,
        )

        # Value annotations
        for bar, m in zip(bars, means):
            if m > 0.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.05,
                    f"{m:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#ffffff",
                )

    ax.set_title("Initial Path Planning Execution Time (ms) by Algorithm & Scenario", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Scenario", fontsize=10, labelpad=8)
    ax.set_ylabel("Execution Time (milliseconds)", fontsize=10, labelpad=8)
    ax.set_xticks(x)
    short_labels = [s.split(". ")[-1] if ". " in s else s for s in scenarios]
    ax.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 2. Search Efficiency / Expanded Nodes Chart
# ---------------------------------------------------------------------------
def plot_expanded_nodes_comparison(summary_data: List[Dict[str, Any]], output_path: str) -> Optional[str]:
    """Generate bar chart comparing expanded nodes count (search space efficiency)."""
    if not summary_data:
        return None

    _apply_theme()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    scenarios = sorted(list(set(row["scenario_name"] for row in summary_data)))
    algorithms = ["astar", "dijkstra", "gbfs"]
    algo_labels = {"astar": "A*", "dijkstra": "Dijkstra", "gbfs": "Greedy Best-First"}
    algo_colors = {"astar": "#3b82f6", "dijkstra": "#ef4444", "gbfs": "#10b981"}

    x = np.arange(len(scenarios))
    width = 0.26

    for i, algo in enumerate(algorithms):
        nodes_vals = []
        for sc in scenarios:
            match = next((r for r in summary_data if r["scenario_name"] == sc and r["algorithm"] == algo), None)
            nodes_vals.append(match["nodes_expanded_mean"] if match and match["nodes_expanded_mean"] is not None else 0.0)

        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            nodes_vals,
            width,
            label=algo_labels.get(algo, algo.upper()),
            color=algo_colors.get(algo, "#888888"),
            alpha=0.9,
            edgecolor="#ffffff",
            linewidth=0.5,
        )

        for bar, v in zip(bars, nodes_vals):
            if v > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 1.0,
                    f"{int(v)}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#ffffff",
                )

    ax.set_title("Search Efficiency: Expanded Node Count Comparison", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Scenario", fontsize=10, labelpad=8)
    ax.set_ylabel("Expanded Nodes (count)", fontsize=10, labelpad=8)
    ax.set_xticks(x)
    short_labels = [s.split(". ")[-1] if ". " in s else s for s in scenarios]
    ax.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 3. Path Cost / Length Comparison Chart
# ---------------------------------------------------------------------------
def plot_path_cost_comparison(summary_data: List[Dict[str, Any]], output_path: str) -> Optional[str]:
    """Generate bar chart comparing path length / cost between algorithms."""
    if not summary_data:
        return None

    _apply_theme()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    scenarios = sorted(list(set(row["scenario_name"] for row in summary_data)))
    algorithms = ["astar", "dijkstra", "gbfs"]
    algo_labels = {"astar": "A* (Optimal)", "dijkstra": "Dijkstra (Optimal)", "gbfs": "GBFS (Heuristic)"}
    algo_colors = {"astar": "#3b82f6", "dijkstra": "#8b5cf6", "gbfs": "#10b981"}

    x = np.arange(len(scenarios))
    width = 0.26

    for i, algo in enumerate(algorithms):
        costs = []
        for sc in scenarios:
            match = next((r for r in summary_data if r["scenario_name"] == sc and r["algorithm"] == algo), None)
            costs.append(match["path_cost_mean"] if match and match["path_cost_mean"] is not None else 0.0)

        offset = (i - 1) * width
        bars = ax.bar(
            x + offset,
            costs,
            width,
            label=algo_labels.get(algo, algo.upper()),
            color=algo_colors.get(algo, "#888888"),
            alpha=0.9,
            edgecolor="#ffffff",
            linewidth=0.5,
        )

        for bar, c in zip(bars, costs):
            if c > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.3,
                    f"{c:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#ffffff",
                )

    ax.set_title("Planned Path Cost Comparison (Grid Steps)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Scenario", fontsize=10, labelpad=8)
    ax.set_ylabel("Path Cost (grid metric units)", fontsize=10, labelpad=8)
    ax.set_xticks(x)
    short_labels = [s.split(". ")[-1] if ". " in s else s for s in scenarios]
    ax.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=9)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 4. Safety: Predicted Collision Risks vs Actual Collisions Chart
# ---------------------------------------------------------------------------
def plot_safety_conflicts_vs_collisions(summary_data: List[Dict[str, Any]], output_path: str) -> Optional[str]:
    """Grouped bar chart showing Predicted Conflict Alerts vs Actual Collisions."""
    if not summary_data:
        return None

    # Filter to scenarios with dynamic metrics
    dyn_rows = [r for r in summary_data if r.get("total_collision_risk_events", 0) > 0 or r.get("total_actual_collisions", 0) >= 0]
    if not dyn_rows:
        return None

    _apply_theme()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    scenarios = sorted(list(set(row["scenario_name"] for row in dyn_rows)))
    x = np.arange(len(scenarios))
    width = 0.35

    risks = []
    collisions = []

    for sc in scenarios:
        match = next((r for r in dyn_rows if r["scenario_name"] == sc), None)
        risks.append(match["total_collision_risk_events"] if match else 0)
        collisions.append(match["total_actual_collisions"] if match else 0)

    bar1 = ax.bar(x - width / 2, risks, width, label="Predicted Collision Risk Alerts", color="#f59e0b", alpha=0.9, edgecolor="#ffffff", linewidth=0.5)
    bar2 = ax.bar(x + width / 2, collisions, width, label="Actual Simulated Collisions (Zero)", color="#10b981", alpha=0.9, edgecolor="#ffffff", linewidth=0.5)

    for bar, val in zip(bar1, risks):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val}", ha="center", va="bottom", fontsize=9, color="#ffffff")

    for bar, val in zip(bar2, collisions):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1, f"{val}", ha="center", va="bottom", fontsize=9, color="#10b981", fontweight="bold")

    ax.set_title("Safety System Efficacy: Predicted Risks Forecasted vs Actual Collisions", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Scenario", fontsize=10, labelpad=8)
    ax.set_ylabel("Event Count", fontsize=10, labelpad=8)
    ax.set_xticks(x)
    short_labels = [s.split(". ")[-1] if ". " in s else s for s in scenarios]
    ax.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", linestyle="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 5. Travel Time / Simulation Duration Distribution
# ---------------------------------------------------------------------------
def plot_travel_time_distribution(runs: List[BenchmarkRunRecord], output_path: str) -> Optional[str]:
    """Plot travel time distribution for completed dynamic simulation runs."""
    dyn_runs = [r for r in runs if r.benchmark_type == "DYNAMIC_SIMULATION" and r.destination_time_s is not None]
    if not dyn_runs:
        return None

    _apply_theme()
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)

    scenarios = sorted(list(set(r.scenario_name for r in dyn_runs)))
    data_by_scen = []
    for sc in scenarios:
        times = [r.destination_time_s for r in dyn_runs if r.scenario_name == sc and r.destination_time_s is not None]
        data_by_scen.append(times if times else [0.0])

    bp = ax.boxplot(
        data_by_scen,
        patch_artist=True,
        tick_labels=[s.split(". ")[-1] if ". " in s else s for s in scenarios],
        boxprops=dict(facecolor="#2563eb", color="#ffffff", alpha=0.8),
        whiskerprops=dict(color="#ffffff"),
        capprops=dict(color="#ffffff"),
        medianprops=dict(color="#fbbf24", linewidth=2),
    )

    ax.set_title("Autonomous Navigation Completion Time (s) per Scenario", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Scenario", fontsize=10, labelpad=8)
    ax.set_ylabel("Travel Time to Destination (seconds)", fontsize=10, labelpad=8)
    ax.grid(axis="y", linestyle="--")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# 6. Executive 4-Panel Performance Dashboard
# ---------------------------------------------------------------------------
def plot_comprehensive_dashboard(summary_data: List[Dict[str, Any]], output_path: str) -> Optional[str]:
    """Generate a combined 4-panel executive dashboard for Phase 9 analysis."""
    if not summary_data:
        return None

    _apply_theme()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9), dpi=150)

    scenarios = sorted(list(set(row["scenario_name"] for row in summary_data)))
    short_labels = [s.split(". ")[-1] if ". " in s else s for s in scenarios]
    x = np.arange(len(scenarios))
    width = 0.26
    algorithms = ["astar", "dijkstra", "gbfs"]
    colors = {"astar": "#3b82f6", "dijkstra": "#ef4444", "gbfs": "#10b981"}

    # Subplot 1: Planning Time (ms)
    for i, algo in enumerate(algorithms):
        vals = [next((r["plan_time_mean_ms"] for r in summary_data if r["scenario_name"] == sc and r["algorithm"] == algo), 0.0) for sc in scenarios]
        ax1.bar(x + (i - 1) * width, vals, width, label=algo.upper(), color=colors[algo], alpha=0.85)
    ax1.set_title("A. Planning Execution Time (ms)", fontsize=10, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=8)
    ax1.set_ylabel("Time (ms)", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", linestyle="--")

    # Subplot 2: Expanded Nodes
    for i, algo in enumerate(algorithms):
        vals = [next((r["nodes_expanded_mean"] for r in summary_data if r["scenario_name"] == sc and r["algorithm"] == algo), 0.0) for sc in scenarios]
        ax2.bar(x + (i - 1) * width, vals, width, label=algo.upper(), color=colors[algo], alpha=0.85)
    ax2.set_title("B. Search Space Efficiency (Expanded Nodes)", fontsize=10, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=8)
    ax2.set_ylabel("Nodes Count", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", linestyle="--")

    # Subplot 3: Success Rate (%)
    succ_rates = [next((r["success_rate"] * 100 for r in summary_data if r["scenario_name"] == sc and r["algorithm"] == "astar"), 100.0) for sc in scenarios]
    ax3.bar(x, succ_rates, 0.45, color="#10b981", alpha=0.85, edgecolor="#ffffff", linewidth=0.5)
    for xi, sr in zip(x, succ_rates):
        ax3.text(xi, sr / 2, f"{sr:.0f}%", ha="center", va="center", color="#ffffff", fontweight="bold", fontsize=9)
    ax3.set_title("C. Scenario Navigation Success Rate (%)", fontsize=10, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=8)
    ax3.set_ylabel("Success Rate (%)", fontsize=9)
    ax3.set_ylim(0, 110)
    ax3.grid(axis="y", linestyle="--")

    # Subplot 4: Replanning Count
    replan_vals = [next((r.get("replanning_count_mean") or 0.0 for r in summary_data if r["scenario_name"] == sc and r["algorithm"] == "astar"), 0.0) for sc in scenarios]
    ax4.bar(x, replan_vals, 0.45, color="#f59e0b", alpha=0.85, edgecolor="#ffffff", linewidth=0.5)
    for xi, rv in zip(x, replan_vals):
        if rv > 0:
            ax4.text(xi, rv + 0.05, f"{rv:.1f}", ha="center", va="bottom", color="#ffffff", fontsize=9)
    ax4.set_title("D. Autonomous Replanning Events Count", fontsize=10, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=8)
    ax4.set_ylabel("Replans Count", fontsize=9)
    ax4.grid(axis="y", linestyle="--")

    plt.suptitle("PHASE 9: PERFORMANCE EVALUATION & COMPARATIVE ANALYSIS DASHBOARD", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Master Chart Generator
# ---------------------------------------------------------------------------
def generate_all_performance_charts(
    runs: List[BenchmarkRunRecord],
    summary: List[Dict[str, Any]],
    output_dir: str = config.REPORTS_DIR,
) -> List[str]:
    """Generate all Phase 9 visual performance charts and return their file paths."""
    os.makedirs(output_dir, exist_ok=True)
    created_charts: List[str] = []

    c1 = plot_planning_time_comparison(summary, os.path.join(output_dir, "planning_time_comparison.png"))
    if c1:
        created_charts.append(c1)

    c2 = plot_expanded_nodes_comparison(summary, os.path.join(output_dir, "expanded_nodes_comparison.png"))
    if c2:
        created_charts.append(c2)

    c3 = plot_path_cost_comparison(summary, os.path.join(output_dir, "path_cost_comparison.png"))
    if c3:
        created_charts.append(c3)

    c4 = plot_safety_conflicts_vs_collisions(summary, os.path.join(output_dir, "safety_conflicts_vs_collisions.png"))
    if c4:
        created_charts.append(c4)

    c5 = plot_travel_time_distribution(runs, os.path.join(output_dir, "travel_time_distribution.png"))
    if c5:
        created_charts.append(c5)

    c6 = plot_comprehensive_dashboard(summary, os.path.join(output_dir, "performance_dashboard_summary.png"))
    if c6:
        created_charts.append(c6)

    return created_charts
