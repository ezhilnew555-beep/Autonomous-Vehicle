# Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads

[![Phase](https://img.shields.io/badge/Phase-10%20Final%20Integration-success.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](#)
[![Pygame](https://img.shields.io/badge/Pygame-2.5%2B-orange.svg)](#)
[![Tests](https://img.shields.io/badge/Tests-90%20Passed%20(100%25)-brightgreen.svg)](#)

A simulation platform developed in Python and Pygame modeling autonomous vehicle navigation, perception, multi-horizon motion prediction, safety supervision, dynamic replanning, and comparative benchmarking under realistic, unstructured Indian road conditions.

---

## 1. Problem Overview & Scope

Autonomous driving on Indian roads presents distinct structural challenges rarely captured in standardized Western benchmarks:
- **Absence of Lane Markings**: Narrow rural roads and informal streets lack demarcated lanes, requiring drivable boundary estimation rather than strict lane keeping.
- **Heterogeneous Mixed Traffic**: Roadways are shared concurrently by passenger cars, auto-rickshaws, high-agility motorcycles, unhurried pedestrians, and wandering cattle.
- **Non-Conforming Road Behaviors**: Oncoming wrong-way traffic, roadside market encroachments, and unprompted lateral street crossings are routine.
- **Severe Spatial Blockages**: Complete route blockages require rapid alternative route discovery or safe emergency stops when all detours are obstructed.

This project delivers an end-to-end autonomous navigation stack designed specifically to address these challenges with real-time perception, motion prediction, safety-governed speed regulation, adaptive replanning, and empirical benchmarking.

---

## 2. System Architecture

The software architecture is modularized into distinct, decoupled subsystems communicating across clean interfaces:

```
+-----------------------------------------------------------------------------------------+
|                                SIMULATION ENVIRONMENT                                   |
|   Road Geometries | Static Occupancy Grid (24x17) | Ground Truth Road Users             |
+--------------------------------------------+--------------------------------------------+
                                             |
                                             v
                      +----------------------------------------------+
                      |         PERCEPTION SYSTEM (Phase 5)          |
                      |   Range Frustum | Latency Buffer | Tracking  |
                      +----------------------+-----------------------+
                                             |  PerceivedObject (Decoupled)
                                             v
                      +----------------------------------------------+
                      |        MOTION PREDICTOR (Phase 6)            |
                      |   Constant-Velocity Model | Horizons [0.5-2s]|
                      |   Expanding Uncertainty Ellipses             |
                      +----------------------+-----------------------+
                                             |  PredictedTrajectory
                                             v
                      +----------------------------------------------+
                      |       SAFETY SUPERVISOR (Phase 7)            |
                      |   Spatial & Temporal Conflict Check          |
                      |   FSM: SAFE -> CAUTION -> REPLAN -> E-STOP   |
                      +----------------------+-----------------------+
                                             |
                         +-------------------+-------------------+
                         |                                       |
                         v                                       v
        +----------------------------------+   +----------------------------------+
        |   DYNAMIC A* REPLANNER (Phase 7) |   |   VEHICLE KINEMATICS (Phase 4)   |
        |   Augmented Grid Routing         |   |   Pure Pursuit Pure-Pursuit Steering|
        |   Alternative Path Acceptance    |   |   Acceleration & Emergency Braking  |
        +----------------------------------+   +----------------------------------+
                         |                                       |
                         +-------------------+-------------------+
                                             |
                                             v
                      +----------------------------------------------+
                      |       EVALUATION & BENCHMARKS (Phase 9/10)   |
                      |   A* vs Dijkstra vs GBFS Comparative Suite   |
                      |   CSV & JSON Exporters | Matplotlib Charts   |
                      +----------------------------------------------+
```

---

## 3. Five Realistic Indian Road Scenarios (Phase 8)

The simulator features five selectable, reproducible scenarios modeling unstructured road topographies:

| Scenario | Road Geometry | Hazards & Road Users | Autonomous Safety Challenge |
| :--- | :--- | :--- | :--- |
| **1. Unmarked Village Road** | Narrow 2-lane rural road | Uneven shoulders, roadside stones, parked auto-rickshaw, oncoming motorcycle. | Navigating narrow drivable bounds, steering around gravel piles, yielding safely to oncoming traffic. |
| **2. Unsignalized Urban Intersection** | 4-way unsignalized cross-junction | Fast through-traffic cars, cross-cutting pedestrians, crossing motorcycles. | Estimating cross-traffic trajectories, modulating speed at intersection mouth, crossing clear gaps. |
| **3. Highway Merging with Slow Traffic** | High-speed multi-lane road with on-ramp | Slow auto-rickshaw occupying ramp, high-speed through traffic on highway lanes. | Assessing merge gap intervals, decelerating behind slow vehicles, safely merging without collisions. |
| **4. Dense Market with Mixed Traffic** | Narrow urban street corridor | Roadside vendor stalls, erratically walking pedestrians, stationary handcarts. | Navigating dense clutter under CAUTION speed, executing multi-turn detours, or halting if blocked. |
| **5. Sudden Cattle Crossing** | Rural unlit highway | Cow unexpectedly stepping into roadway from unpaved shoulder at scheduled time. | Perceiving cattle, triggering immediate emergency stop, holding position, and resuming once cleared. |

---

## 4. Algorithmic Formulations

### A. Motion Prediction & Uncertainty Model (Phase 6)
Future positions of perceived road users are forecast using the **Constant-Velocity Motion Model (CVMM)**:

$$x_{future}(t) = x + v_x \cdot t, \quad y_{future}(t) = y + v_y \cdot t$$

Predicted spatial uncertainty expands over time according to category-weighted dispersion:

$$r_{uncertainty}(t) = r_0 + w_{category} \cdot k_{growth} \cdot t + \delta_{penalty}$$

- $r_0 = 6.0\text{ px}$ (base localization error)
- $k_{growth} = 14.0\text{ px/s}$ (dispersion rate)
- $w_{category}$: `PEDESTRIAN` ($1.6$), `CATTLE` ($1.7$), `MOTORCYCLE` ($1.3$), `AUTORICKSHAW` ($1.1$), `CAR` ($1.0$), `BARRIER` ($0.1$)
- $\delta_{penalty} = 10.0\text{ px}$ if observation history is sparse ($\le 2$ observations)

Confidence is classified explicitly:
- **HIGH**: $\ge 4$ observations and velocity variance $< 0.15$
- **MEDIUM**: $2$–$3$ observations or moderate variance
- **LOW**: $< 2$ observations (marked uncertain; expands safety padding)

### B. Safety Supervisor Finite State Machine (Phase 7)
The safety supervisor continuously monitors upcoming route waypoints:

$$\text{SAFE} \xrightarrow{\text{warning conflict}} \text{CAUTION} \xrightarrow{\text{replan trigger}} \text{REPLANNING} \xrightarrow{\text{detour adopted}} \text{CAUTION} \xrightarrow{\text{clear}} \text{SAFE}$$
$$\text{ANY} \xrightarrow{\text{imminent conflict / } t \le 0.4\text{s}} \text{EMERGENCY STOP} \xrightarrow{\text{obstacle cleared}} \text{SAFE}$$
$$\text{REPLANNING} \xrightarrow{\text{no viable detour}} \text{NO SAFE PATH}$$

- **CAUTION**: Speed multiplier throttled to $50\%$ ($config.CAUTION\_SPEED\_FACTOR$).
- **EMERGENCY STOP**: Speed instantaneously forced to $0.0\text{ px/s}$.
- **NO SAFE PATH**: Vehicle halts permanently until an obstacle is cleared or user intervenes.

### C. Comparative Path Planners (Phase 9)
All algorithms operate on the exact same 8-connected grid with corner-cutting prevention:

$$f(n) = g(n) + W_h \cdot h(n)$$

- **A\*** ($W_h = 1$): Evaluates $f(n) = g(n) + h(n)$ using Octile distance. **Guaranteed optimal path length with minimal node expansions.**
- **Dijkstra** ($W_h = 0$): Evaluates $f(n) = g(n)$. **Guaranteed optimal**, but expands 2–3$\times$ more nodes.
- **Greedy Best-First Search (GBFS)** ($g \equiv 0$): Evaluates $f(n) = h(n)$. Expands the fewest nodes, but produces suboptimal path lengths.

---

## 5. Performance Metrics & Physical Units

| Metric Field | Unit | Physical Meaning | Calculation Method |
| :--- | :--- | :--- | :--- |
| `planning_time_ms` | ms | Planner CPU compute duration | `time.perf_counter()` duration inside `plan()` |
| `expanded_nodes` | count | Search space exploration volume | Total states popped from planner open set |
| `path_cost_grid` | cells | Cumulative path edge weights | Cardinal step: $1.0$, Diagonal step: $\sqrt{2}$ |
| `path_length_px` | px | Metric length along path | $\text{path\_cost} \times 40\text{px/cell}$ |
| `simulation_duration_s` | s | Total elapsed simulation time | Simulation kinematic time until termination |
| `destination_time_s` | s | Arrival time at destination | Elapsed time when `state == ARRIVED` (`None` if failed) |
| `average_speed_px_s` | px/s | Mean linear speed | $\text{distance\_traveled} / \text{duration}$ |
| `min_obstacle_clearance_px` | px | Minimum observed clearance | Smallest distance observed to any obstacle |
| `emergency_stops` | count | Total emergency braking events | Supervisor transitions to `EMERGENCY_STOP` |
| `collision_risk_events` | count | Predicted safety conflict warnings | Warnings flagged by supervisor lookahead |
| `actual_simulated_collisions` | count | Physical overlaps | 2D Oriented Bounding Box SAT intersections |

---

## 6. Installation & Prerequisites

### Prerequisites
- Python **3.10** or **3.11** (recommended: Python 3.11)
- Windows, macOS, or Linux

### Setup
```bash
# 1. Clone or navigate to the repository
cd autonomous_vehicle_sim

# 2. (Optional) Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 7. Launch Instructions & User Interface

### Launching the Application
```bash
# Windows
py -3.11 main.py
# or standard python
python main.py
```

### Visual Interface Layout
- **Top Status Bar**: Live Safety state badge (`SAFE`, `CAUTION`, `REPLANNING`, `EMERGENCY_STOP`, `NO_SAFE_PATH`), Scenario title, Obstacle counts, Prediction horizon, Speed multiplier.
- **Central Simulation Viewport ($960 \times 680$px)**: Road layout, start (S) and goal (D) markers, vehicle footprint, sensor detection frustum, motion prediction trajectories with uncertainty ellipses, and conflict warning zones.
- **Side Panel ($320 \times 680$px)**:
  - **Controls**: Interactive buttons for starting simulation, cycling speed, switching algorithms, toggling scenarios, running benchmarks, and triggering demo mode.
  - **Telemetry & Safety**: Real-time vehicle speed, heading, active planner, waypoint index, supervisor state, conflicts, and replans.
  - **Motion Prediction Panel**: Obstacle IDs, categories, confidence levels, positions, velocities, and uncertainty radii.
  - **Legend & Hotkeys**: Color swatches for road elements and quick reference keys.

---

## 8. Keyboard & Mouse Controls Reference

| Control / Key | Action | Description |
| :--- | :--- | :--- |
| <kbd>SPACE</kbd> | **Start / Pause** | Toggle vehicle autonomous navigation. |
| <kbd>D</kbd> | **Demo Mode** | Toggle deterministic 7-stage demonstration walkthrough. |
| <kbd>B</kbd> | **Benchmarks** | Open Phase 9/10 performance evaluation and comparisons screen. |
| <kbd>M</kbd> | **Scenarios Menu** | Open modal scenario selection catalog. |
| <kbd>A</kbd> | **Cycle Planner** | Toggle active planner: A* $\rightarrow$ Dijkstra $\rightarrow$ GBFS. |
| <kbd>P</kbd> | **Plan Route** | Compute path from current vehicle position to destination. |
| <kbd>H</kbd> | **Cycle Horizon** | Cycle prediction lookahead: $0.5$s $\rightarrow$ $1.0$s $\rightarrow$ $1.5$s $\rightarrow$ $2.0$s. |
| <kbd>U</kbd> | **Toggle Uncertainty** | Show or hide prediction uncertainty ellipses. |
| <kbd>1</kbd> – <kbd>6</kbd> | **Obstacle Selector** | 1: Car, 2: Moto, 3: Auto, 4: Pedestrian, 5: Cattle, 6: Barrier. |
| <kbd>S</kbd> | **Spawn Obstacle** | Spawn selected obstacle type at mouse cursor or road center. |
| <kbd>F1</kbd> – <kbd>F5</kbd> | **Quick Load Scenario** | F1: Village, F2: Intersection, F3: Merge, F4: Market, F5: Cattle. |
| <kbd>F6</kbd> / <kbd>R</kbd> | **Restart Scenario** | Re-initialize active scenario to initial conditions. |
| <kbd>X</kbd> | **Reset Vehicle** | Return vehicle to start position without modifying obstacles. |
| <kbd>G</kbd> | **Toggle Grid** | Toggle visual grid overlay lines. |
| <kbd>Left Click</kbd> | **Add Obstacle** | Place static grid obstacle on clicked road cell. |
| <kbd>Right Click</kbd> | **Remove Obstacle** | Remove obstacle under cursor. |
| <kbd>Esc</kbd> | **Exit / Back** | Close active modal dialog or exit application. |

---

## 9. Deterministic Demonstration Mode Guide

Pressing <kbd>D</kbd> or clicking **Demo (D)** triggers the 7-stage guided demonstration:

1. **Stage 1: Scenario Initialization & Global Path Planning** — Loads Scenario 1 (Unmarked Village Road) and computes the baseline collision-free A* route.
2. **Stage 2: Autonomous Kinematic Navigation** — Starts vehicle navigation; vehicle accelerates smoothly along waypoints.
3. **Stage 3: Road Obstruction & Dynamic Replanning** — Injects a static roadblock obstacle on the planned trajectory; the safety supervisor detects the obstruction and executes an immediate collision-free A* detour.
4. **Stage 4: Perception Tracking & Motion Prediction** — Spawns oncoming road users; displays simulated perception with delay filtering, velocity estimation, and multi-horizon predictions with expanding uncertainty ellipses.
5. **Stage 5: Predictive Collision Avoidance & Emergency Braking** — Demonstrates safety supervisor FSM transitions: speed moderation under `CAUTION` and emergency stopping under imminent conflict.
6. **Stage 6: Sudden Cattle Crossing** — Loads Scenario 5; demonstrates reactive yielding to crossing cattle and safe recovery once the roadway clears.
7. **Stage 7: Comparative Benchmarking & Reports Export** — Executes the automated benchmark suite across all scenarios and algorithms, displays the performance evaluation dashboard, and exports all CSVs, JSON metadata, and charts to `reports/`.

*Controls during Demo Mode:*
- <kbd>SPACE</kbd> / <kbd>→</kbd>: Advance to next demonstration stage immediately.
- <kbd>Backspace</kbd> / <kbd>←</kbd>: Step back to previous stage.
- <kbd>R</kbd>: Restart demonstration from Stage 1.
- <kbd>D</kbd> / <kbd>Esc</kbd>: Exit Demonstration Mode.

---

## 10. Reports & Export Directory (`reports/`)

When benchmarks are executed in the simulation or via `src/benchmarks.py`, comprehensive evaluation artifacts are exported directly to `reports/`:

```
reports/
├── benchmark_runs.csv                    # Granular record of every single run executed
├── benchmark_summary.csv                 # Aggregated statistics grouped by scenario & algorithm
├── benchmark_metadata.json               # System configuration, Python version, grid parameters
├── planning_time_comparison.png          # Execution latency comparison bar chart
├── expanded_nodes_comparison.png         # Node expansion search volume comparison chart
├── path_cost_comparison.png              # Path length optimality comparison bar chart
├── safety_conflicts_vs_collisions.png    # Predicted risk warnings vs actual simulated collisions
├── travel_time_distribution.png          # Scenario traversal duration box plot
└── performance_dashboard_summary.png     # 4-panel executive performance dashboard
```

---

## 11. Automated Test Suite

The test suite validates correctness across all project phases:
```bash
# Run complete test suite (90 tests)
py -3.11 -m pytest tests/ -v
```

Test coverage by module:
- `tests/test_phase2.py`: Occupancy grid initialization, bounds, obstacle placement & removal.
- `tests/test_phase3.py`: A* path planning, diagonal corner-cutting prevention, detour generation.
- `tests/test_phase4.py`: Vehicle kinematics, waypoint progression, pure pursuit steering, collision stops.
- `tests/test_phase5.py`: Perception system, range filtering, delay buffer, velocity estimation.
- `tests/test_phase6.py`: Constant-velocity motion prediction, configurable horizons, uncertainty growth math.
- `tests/test_phase7.py`: Collision detection, safety supervisor FSM transitions, automatic replanning.
- `tests/test_phase8.py`: Five Indian road scenarios, scheduled events, seed determinism, menu geometry.
- `tests/test_phase9.py`: A* vs Dijkstra vs GBFS benchmarking, Bessel's correction $N-1$, CSV/JSON exporters, charts.
- `tests/test_phase10.py`: Demonstration controller lifecycle, 7-stage workflow, algorithm cycling, integration pipeline.
- `tests/test_smoke.py`: End-to-end import and configuration smoke checks.

---

## 12. Known Limitations & Future Research

1. **2D Discrete Grid Resolution**: Path planning operates on a $40\text{px}$ occupancy grid. While suitable for real-time simulation, continuous-curvature lattice planners (e.g., Hybrid A* or polynomial splines) would provide smoother trajectories for vehicles with Ackermann steering constraints.
2. **Constant-Velocity Motion Assumption**: Phase 6 uses a linear velocity model. Unpredictable road users (such as turning auto-rickshaws or darting pedestrians) may deviate from straight trajectories; future iterations could incorporate interactive Gaussian Process or intention-aware prediction models.
3. **Simulation Decoupling**: Sensors and delays are simulated via software models rather than raw LiDAR point clouds or camera sensor fusion streams.
4. **Safety Verification Disclaimer**: This software is an academic simulation benchmark platform. The safety mechanisms implemented herein are designed for simulated collision avoidance and must not be interpreted as certified safety guarantees for physical production autonomous driving systems.
