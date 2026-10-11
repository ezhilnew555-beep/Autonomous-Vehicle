"""
config.py
=========
Central configuration for the Autonomous Vehicle Simulation.
All tuneable constants live here so every other module can
import from one authoritative source.
"""

# ---------------------------------------------------------------------------
# Window / Display
# ---------------------------------------------------------------------------
WINDOW_TITLE = "Adaptive Path Planning & Collision Avoidance (Phase 10)"
SCREEN_WIDTH  = 1280
SCREEN_HEIGHT = 720
FPS           = 60

# ---------------------------------------------------------------------------
# Layout zones (pixels)
# ---------------------------------------------------------------------------
STATUS_BAR_HEIGHT = 40          # top bar height
SIM_AREA_TOP      = STATUS_BAR_HEIGHT
SIM_AREA_WIDTH    = 960         # left panel - simulation canvas
PANEL_WIDTH       = SCREEN_WIDTH - SIM_AREA_WIDTH   # right panel - legend / info

# Simulation canvas rectangle (x, y, w, h)
SIM_RECT = (0, SIM_AREA_TOP, SIM_AREA_WIDTH, SCREEN_HEIGHT - SIM_AREA_TOP)

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
GRID_CELL_SIZE = 40             # pixels per grid cell
GRID_COLS = SIM_AREA_WIDTH  // GRID_CELL_SIZE
GRID_ROWS = (SCREEN_HEIGHT - SIM_AREA_TOP) // GRID_CELL_SIZE

# ---------------------------------------------------------------------------
# Colour palette  (R, G, B)
# ---------------------------------------------------------------------------

# Background / structural
C_BACKGROUND        = (15,  25,  50)
C_SIM_BG            = (20,  32,  60)
C_PANEL_BG          = (12,  20,  42)
C_STATUS_BAR_BG     = (10,  16,  36)

# Grid
C_GRID_LINE         = (30,  45,  80)

# Road
C_ROAD_SURFACE      = (55,  55,  65)
C_ROAD_MARKING      = (220, 200,  60)
C_ROAD_EDGE         = (200, 200, 200)
C_ROAD_SHOULDER     = (80,  75,  60)

# Markers
C_START_MARKER      = ( 50, 200,  80)
C_DEST_MARKER       = (220,  50,  50)

# UI chrome
C_TEXT_PRIMARY      = (220, 230, 255)
C_TEXT_SECONDARY    = (160, 175, 210)
C_ACCENT            = ( 80, 160, 255)
C_DIVIDER           = ( 40,  55, 100)
C_SWATCH_BORDER     = ( 60,  80, 130)

# Obstacles (Phase 2)
C_OBSTACLE          = (200,  80,  60)   # warm red-orange – obstacle fill
C_OBSTACLE_BORDER   = (240, 120,  80)   # lighter border for contrast
C_OBSTACLE_X        = (255, 200, 180)   # X marking on obstacle

# Buttons (Phase 2)
C_BUTTON_BG         = ( 30,  48,  90)   # button background
C_BUTTON_HOVER      = ( 45,  65, 120)   # button hovered
C_BUTTON_ACTIVE     = ( 80, 160, 255)   # button active / selected
C_BUTTON_TEXT       = (220, 230, 255)   # button label
C_BUTTON_BORDER     = ( 50,  70, 120)   # button border

# Hover highlight
C_HOVER_VALID       = ( 50, 200,  80, 80)   # semi-transparent green
C_HOVER_INVALID     = (220,  50,  50, 80)   # semi-transparent red

# Path Planning (Phase 3)
C_PATH              = (  0, 220, 255)   # bright cyan - path line
C_PATH_NODE         = ( 40, 180, 230)   # cyan dot on nodes
C_PATH_OUTDATED     = (180, 180, 100)   # yellow-ish dimmed cyan when outdated
C_PATH_TEXT         = (120, 230, 255)   # cyan info text
C_STATUS_SUCCESS    = ( 80, 220, 120)   # green
C_STATUS_FAILED     = (240,  90,  90)   # red
C_STATUS_OUTDATED   = (240, 200,  70)   # yellow/orange

# Vehicle Appearance & Colors (Phase 4)
C_VEHICLE_BODY      = ( 50, 140, 240)   # vibrant blue vehicle body
C_VEHICLE_ROOF      = ( 30,  90, 180)   # darker blue roof/cabin
C_VEHICLE_HEADLIGHT = (255, 240, 100)   # warm yellow headlights
C_VEHICLE_TAILLIGHT = (255,  50,  50)   # red taillights
C_VEHICLE_OUTLINE   = (220, 240, 255)   # crisp border highlight
C_VEHICLE_HEADING   = (255, 255, 255)   # heading line

# ---------------------------------------------------------------------------
# Vehicle Dynamics Settings (Phase 4)
# ---------------------------------------------------------------------------
VEHICLE_LENGTH      = 28.0              # pixels
VEHICLE_WIDTH       = 18.0              # pixels
VEHICLE_MAX_SPEED   = 120.0             # pixels / second (approx 3 cells / sec)
VEHICLE_ACCEL       = 80.0              # pixels / second^2
VEHICLE_DECEL       = 140.0             # pixels / second^2 (braking)
VEHICLE_MAX_TURN    = 4.5               # radians / second (~260 deg/sec)
WAYPOINT_THRESHOLD  = 12.0              # pixels radius to declare waypoint reached

# Simulation Speeds
SIM_SPEED_OPTIONS   = [1.0, 1.5, 2.0, 0.5]

# ---------------------------------------------------------------------------
# Simulated Perception System (Phase 5)
# ---------------------------------------------------------------------------
DEFAULT_DETECTION_RANGE = 220.0         # pixels radius (~5.5 grid cells)
DETECTION_RANGE_OPTIONS = [150.0, 220.0, 300.0, 100.0]
DEFAULT_DETECTION_DELAY = 0.10          # seconds
DETECTION_DELAY_OPTIONS = [0.0, 0.10, 0.25, 0.50]
DEFAULT_MISS_RATE       = 0.0           # 0% simulated sensor dropout

# Perception Visuals & Tracking Colors
C_RANGE_CIRCLE          = ( 60, 180, 255, 35)   # semi-transparent sensor coverage
C_RANGE_BORDER          = ( 80, 200, 255, 120)  # sensor boundary
C_PERCEIVED_BBOX        = (  0, 255, 180)       # bright neon green-cyan bounding box
C_PERCEIVED_LABEL       = (180, 255, 220)       # perception tag
C_OUT_OF_RANGE_TAG      = (140, 140, 160)       # dimmed tag when outside sensor view

# Road User Objects Palette (Ground Truth)
C_OBJ_CAR               = (220, 130,  50)       # orange-amber car
C_OBJ_MOTORCYCLE        = (240, 200,  60)       # golden motorcycle
C_OBJ_AUTORICKSHAW      = ( 50, 200, 160)       # teal-green rickshaw
C_OBJ_PEDESTRIAN        = (230,  80, 140)       # magenta pedestrian
C_OBJ_CATTLE            = (170, 110,  70)       # brown cattle
C_OBJ_BARRIER           = (180,  60,  60)       # dark red barrier

# ---------------------------------------------------------------------------
# Road geometry (grid-cell coordinates)
# ---------------------------------------------------------------------------
ROAD_ROW_CENTER = GRID_ROWS // 2
ROAD_LANE_WIDTH = 2
ROAD_HALF_WIDTH = ROAD_LANE_WIDTH

# Pixel y-coordinates of road edges (derived)
ROAD_TOP_Y    = (ROAD_ROW_CENTER - ROAD_HALF_WIDTH) * GRID_CELL_SIZE + SIM_AREA_TOP
ROAD_BOTTOM_Y = (ROAD_ROW_CENTER + ROAD_HALF_WIDTH) * GRID_CELL_SIZE + SIM_AREA_TOP
ROAD_CENTER_Y = ROAD_ROW_CENTER * GRID_CELL_SIZE + SIM_AREA_TOP

# ---------------------------------------------------------------------------
# Markers (grid-cell coordinates)
# ---------------------------------------------------------------------------
START_GRID_COL = 2
START_GRID_ROW = ROAD_ROW_CENTER
DEST_GRID_COL  = GRID_COLS - 3
DEST_GRID_ROW  = ROAD_ROW_CENTER

# ---------------------------------------------------------------------------
# Font sizes
# ---------------------------------------------------------------------------
FONT_SIZE_TITLE  = 20
FONT_SIZE_BODY   = 15
FONT_SIZE_SMALL  = 12

# ---------------------------------------------------------------------------
# Interaction modes (Phase 2, 3, 4 & 5)
# ---------------------------------------------------------------------------
MODE_ADD_OBSTACLE    = "Add Obstacle"
MODE_REMOVE_OBSTACLE = "Remove Obstacle"
DEFAULT_MODE         = MODE_ADD_OBSTACLE

# ---------------------------------------------------------------------------
# Button layout
# ---------------------------------------------------------------------------
BUTTON_HEIGHT = 28
BUTTON_MARGIN = 6

# ---------------------------------------------------------------------------
# Motion Prediction System (Phase 6)
# ---------------------------------------------------------------------------
PREDICTION_HORIZONS     = [0.5, 1.0, 1.5, 2.0]  # seconds into the future
DEFAULT_HORIZON_INDEX   = 1                     # 1.0s by default
PREDICTION_STEPS        = 4                     # sub-steps for smooth trajectory arc
UNCERTAINTY_MIN_RADIUS  = 6.0                   # base localization uncertainty (pixels)
UNCERTAINTY_GROWTH_RATE = 14.0                  # uncertainty radius expansion per second (pixels/s)

# Prediction Visual Colors
C_PREDICTED_LINE        = (255, 170,   0)       # bright amber predicted trajectory line
C_PREDICTED_POINT       = (255, 210,  50)       # waypoint/horizon marker
C_PREDICTED_UNCERTAINTY = (255, 180,  50, 45)   # semi-transparent uncertainty region
C_PREDICTED_UNCERTAIN_BORDER = (255, 160, 40, 120) # boundary for uncertainty ellipse
C_CONFIDENCE_HIGH       = ( 80, 220, 120)       # green
C_CONFIDENCE_MEDIUM     = (240, 200,  70)       # yellow
C_CONFIDENCE_LOW        = (240,  90,  90)       # red

# ---------------------------------------------------------------------------
# Collision Avoidance & Safety Supervisor (Phase 7)
# ---------------------------------------------------------------------------
SAFETY_MARGIN_BASE          = 8.0               # pixels base clearance around vehicle
SAFETY_MARGIN_UNCERTAIN     = 18.0              # pixels conservative margin for unreliable/uncertain tracks
SAFETY_BRAKING_DISTANCE_FACTOR = 1.2            # multiplier applied to kinematic stopping distance
COLLISION_LOOKAHEAD_STEPS   = 5                 # time steps evaluated along prediction horizon
COLLISION_HORIZON_DEFAULT   = 2.0               # seconds to project future conflicts
CAUTION_SPEED_FACTOR        = 0.5               # speed multiplier during CAUTION state
REPLAN_COOLDOWN_TIME        = 1.0               # seconds before allowing another replan to avoid thrashing
EMERGENCY_STOP_THRESHOLD    = 38.0              # pixels distance threshold for emergency braking

# Phase 7 Safety Colors
C_SAFETY_SAFE               = ( 80, 220, 120)   # green
C_SAFETY_CAUTION            = (240, 200,  70)   # yellow/amber
C_SAFETY_REPLANNING         = ( 80, 190, 255)   # cyan/light blue
C_SAFETY_EMERGENCY          = (240,  60,  60)   # bright red
C_SAFETY_NO_PATH            = (220,  80, 220)   # magenta/purple
C_PATH_UNSAFE               = (240,  60,  60)   # red for compromised path segments
C_CONFLICT_ZONE             = (255,  70,  70, 90) # semi-transparent collision warning disk
C_CONFLICT_BORDER           = (255, 100, 100, 200)

# ---------------------------------------------------------------------------
# Indian Road Scenarios (Phase 8)
# ---------------------------------------------------------------------------
SCENARIO_TIMEOUT_DEFAULT    = 30.0              # seconds
C_MENU_BG                   = (12,  18,  36, 235) # semi-transparent dark menu overlay
C_MENU_CARD_BG              = (20,  30,  58)
C_MENU_CARD_BORDER          = (45,  70, 130)
C_MENU_CARD_ACTIVE          = (35,  60, 110)
C_MENU_BADGE                = (40, 120, 220)

# ---------------------------------------------------------------------------
# Performance Evaluation & Benchmarking (Phase 9)
# ---------------------------------------------------------------------------
REPORTS_DIR                 = "reports"
BENCHMARK_DEFAULT_REPS      = 3
BENCHMARK_DEFAULT_SEED      = 42
C_BENCHMARK_BG              = (10,  16,  32, 248) # modal backdrop for benchmark screen
C_BENCHMARK_TABLE_HEADER    = (28,  44,  84)
C_BENCHMARK_TABLE_ROW_ALT   = (16,  24,  48)

# ---------------------------------------------------------------------------
# Planners (Phase 9 & 10)
# ---------------------------------------------------------------------------
PLANNER_ALGORITHMS          = ["astar", "dijkstra", "gbfs"]
PLANNER_LABELS              = {
    "astar": "A* (Optimal)",
    "dijkstra": "Dijkstra (h=0)",
    "gbfs": "GBFS (Greedy)",
}

# ---------------------------------------------------------------------------
# Demonstration Mode (Phase 10)
# ---------------------------------------------------------------------------
DEMO_STAGE_DURATION         = 6.5               # seconds per demonstration step
C_DEMO_BANNER_BG            = (10,  18,  38, 240) # semi-transparent dark banner overlay
C_DEMO_ACCENT               = (255, 195,  45)   # bright gold accent
C_DEMO_BADGE                = ( 38, 130, 240)   # royal blue stage badge
C_DEMO_PROGRESS_BAR         = ( 60, 210, 130)   # vibrant green timer progress

# ---------------------------------------------------------------------------
# Phase metadata
# ---------------------------------------------------------------------------
CURRENT_PHASE   = 10
PHASE_LABEL     = f"Phase {CURRENT_PHASE} - Final Integration & Demo"

