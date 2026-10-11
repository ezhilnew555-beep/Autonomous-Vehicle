"""
src/ui.py
=========
Draws the non-simulation chrome: status bar, control buttons, path metrics,
live vehicle telemetry, perceived object perception list, motion prediction dashboard,
and legend for Phase 6.
"""

from typing import List, Tuple, Callable, Optional, Dict
import pygame
import config
from src.obstacles import ObstacleType, _TYPE_COLOURS
from src.planner import Path, PathResult
from src.vehicle import Vehicle, VehicleState
from src.perception import PerceivedObject
from src.prediction import PredictedTrajectory


class Button:
    """A clickable UI button."""

    def __init__(self, rect: pygame.Rect, label: str, action: Callable[[], None], is_toggle: bool = False):
        self.rect = rect
        self.label = label
        self.action = action
        self.is_toggle = is_toggle
        self.is_active = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.action()
                return True
        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        mouse_pos = pygame.mouse.get_pos()
        is_hover = self.rect.collidepoint(mouse_pos)

        if self.is_active:
            bg_col = config.C_BUTTON_ACTIVE
            text_col = (10, 20, 40)
        elif is_hover:
            bg_col = config.C_BUTTON_HOVER
            text_col = config.C_TEXT_PRIMARY
        else:
            bg_col = config.C_BUTTON_BG
            text_col = config.C_BUTTON_TEXT

        pygame.draw.rect(surface, bg_col, self.rect, border_radius=4)
        pygame.draw.rect(surface, config.C_BUTTON_BORDER, self.rect, width=1, border_radius=4)

        text_surf = font.render(self.label, True, text_col)
        tx = self.rect.centerx - text_surf.get_width() // 2
        ty = self.rect.centery - text_surf.get_height() // 2
        surface.blit(text_surf, (tx, ty))


def draw_status_bar(
    surface: pygame.Surface,
    font: pygame.font.Font,
    mode: str,
    obstacle_count: int,
    detected_count: int,
    veh_state: str,
    sim_speed: float,
    horizon: float,
    safety_state: str = "SAFE",
    scenario_name: Optional[str] = None,
    scenario_status: Optional[str] = None,
) -> None:
    """Draw top status bar with mode, obstacles, detections, vehicle state, sim speed, horizon, safety state, and active scenario."""
    bar_rect = pygame.Rect(0, 0, config.SCREEN_WIDTH, config.STATUS_BAR_HEIGHT)
    pygame.draw.rect(surface, config.C_STATUS_BAR_BG, bar_rect)

    pygame.draw.line(
        surface,
        config.C_DIVIDER,
        (0, config.STATUS_BAR_HEIGHT - 1),
        (config.SCREEN_WIDTH, config.STATUS_BAR_HEIGHT - 1),
    )

    # 1. Title on Left
    title_surf = font.render(config.WINDOW_TITLE, True, config.C_TEXT_PRIMARY)
    surface.blit(title_surf, (12, (config.STATUS_BAR_HEIGHT - title_surf.get_height()) // 2))

    # 2. Phase Label on Right
    phase_surf = font.render(config.PHASE_LABEL, True, config.C_TEXT_SECONDARY)
    right_x = config.SCREEN_WIDTH - phase_surf.get_width() - 14
    surface.blit(phase_surf, (right_x, (config.STATUS_BAR_HEIGHT - phase_surf.get_height()) // 2))

    # 3. Safety color & state badge in Center
    safety_col_map = {
        "SAFE": config.C_SAFETY_SAFE,
        "CAUTION": config.C_SAFETY_CAUTION,
        "REPLANNING": config.C_SAFETY_REPLANNING,
        "EMERGENCY_STOP": config.C_SAFETY_EMERGENCY,
        "NO_SAFE_PATH": config.C_SAFETY_NO_PATH,
    }
    safety_col = safety_col_map.get(safety_state, config.C_TEXT_PRIMARY)

    scen_tag = f"[{scenario_name[:16]}] " if scenario_name else ""
    status_tag = f"({scenario_status}) " if scenario_status else ""
    info_str = f"{scen_tag}{status_tag}Obs:{obstacle_count} | Det:{detected_count} | Pred:{horizon:.1f}s | {veh_state} ({sim_speed:.1f}x)"
    info_surf = font.render(info_str, True, config.C_TEXT_PRIMARY)

    # Safety badge pill
    badge_txt = font.render(safety_state, True, (255, 255, 255))
    badge_w = badge_txt.get_width() + 16
    badge_h = 22

    total_center_w = badge_w + 10 + info_surf.get_width()
    center_start_x = max(title_surf.get_width() + 24, (config.SCREEN_WIDTH - total_center_w) // 2)

    badge_rect = pygame.Rect(center_start_x, (config.STATUS_BAR_HEIGHT - badge_h) // 2, badge_w, badge_h)
    pygame.draw.rect(surface, safety_col, badge_rect, border_radius=4)
    surface.blit(badge_txt, (badge_rect.centerx - badge_txt.get_width() // 2, badge_rect.centery - badge_txt.get_height() // 2))

    surface.blit(info_surf, (center_start_x + badge_w + 10, (config.STATUS_BAR_HEIGHT - info_surf.get_height()) // 2))


def draw_grid(surface: pygame.Surface) -> None:
    """Draw faint grid lines over simulation area."""
    sim_x, sim_y, sim_w, sim_h = config.SIM_RECT

    for col in range(config.GRID_COLS + 1):
        x = sim_x + col * config.GRID_CELL_SIZE
        pygame.draw.line(surface, config.C_GRID_LINE, (x, sim_y), (x, sim_y + sim_h))

    for row in range(config.GRID_ROWS + 1):
        y = sim_y + row * config.GRID_CELL_SIZE
        pygame.draw.line(surface, config.C_GRID_LINE, (sim_x, y), (sim_x + sim_w, y))


def draw_markers(
    surface: pygame.Surface,
    start_pos: Optional[Tuple[int, int]] = None,
    dest_pos: Optional[Tuple[int, int]] = None,
) -> None:
    """Draw green start and red destination markers."""
    cell = config.GRID_CELL_SIZE
    pad = 4

    start_c, start_r = start_pos if start_pos is not None else (config.START_GRID_COL, config.START_GRID_ROW)
    dest_c, dest_r = dest_pos if dest_pos is not None else (config.DEST_GRID_COL, config.DEST_GRID_ROW)

    sx = start_c * cell + pad
    sy = start_r * cell + config.SIM_AREA_TOP + pad
    start_rect = pygame.Rect(sx, sy, cell - 2 * pad, cell - 2 * pad)
    pygame.draw.rect(surface, config.C_START_MARKER, start_rect, border_radius=6)

    s_font = pygame.font.SysFont("consolas", 16, bold=True)
    s_surf = s_font.render("S", True, (255, 255, 255))
    surface.blit(s_surf, (start_rect.centerx - s_surf.get_width() // 2, start_rect.centery - s_surf.get_height() // 2))

    dx = dest_c * cell + pad
    dy = dest_r * cell + config.SIM_AREA_TOP + pad
    dest_rect = pygame.Rect(dx, dy, cell - 2 * pad, cell - 2 * pad)
    pygame.draw.rect(surface, config.C_DEST_MARKER, dest_rect, border_radius=6)

    d_font = pygame.font.SysFont("consolas", 16, bold=True)
    d_surf = d_font.render("D", True, (255, 255, 255))
    surface.blit(d_surf, (dest_rect.centerx - d_surf.get_width() // 2, dest_rect.centery - d_surf.get_height() // 2))


def draw_cell_highlight(surface: pygame.Surface, col: int, row: int, is_valid: bool) -> None:
    """Highlight hovered cell on grid."""
    cell = config.GRID_CELL_SIZE
    x = col * cell
    y = row * cell + config.SIM_AREA_TOP
    rect = pygame.Rect(x, y, cell, cell)

    highlight_surface = pygame.Surface((cell, cell), pygame.SRCALPHA)
    color = config.C_HOVER_VALID if is_valid else config.C_HOVER_INVALID
    highlight_surface.fill(color)
    surface.blit(highlight_surface, (x, y))
    pygame.draw.rect(surface, (255, 255, 255, 120), rect, width=1)


def draw_planned_path(surface: pygame.Surface, path: Path, is_outdated: bool) -> None:
    """Render planned path line and waypoints."""
    if not path or len(path) < 2:
        return

    cell = config.GRID_CELL_SIZE
    half = cell // 2
    points = [(c * cell + half, r * cell + config.SIM_AREA_TOP + half) for c, r in path]

    path_color = config.C_PATH_OUTDATED if is_outdated else config.C_PATH
    node_color = (200, 200, 120) if is_outdated else config.C_PATH_NODE

    pygame.draw.lines(surface, path_color, False, points, 3)
    for pt in points:
        pygame.draw.circle(surface, node_color, pt, 3)


def _draw_swatch(surface: pygame.Surface, x: int, y: int, color, size: int = 12) -> None:
    rect = pygame.Rect(x, y, size, size)
    pygame.draw.rect(surface, color, rect, border_radius=3)
    pygame.draw.rect(surface, config.C_SWATCH_BORDER, rect, width=1, border_radius=3)


_BASE_LEGEND = [
    (config.C_ROAD_SURFACE,       "Road Surface"),
    (config.C_PATH,               "A* Planned Path"),
    (config.C_SAFETY_REPLANNING,  "Replanned Detour"),
    (config.C_PERCEIVED_BBOX,     "Detected Obstacle"),
    (config.C_PREDICTED_LINE,     "Predicted Traj"),
    (config.C_SAFETY_EMERGENCY,   "Conflict Risk Zone"),
]


def draw_side_panel(
    surface: pygame.Surface,
    buttons: List[Button],
    selected_obs_type: ObstacleType,
    vehicle: Vehicle,
    path_result: Optional[PathResult],
    is_path_outdated: bool,
    perceived_objects: List[PerceivedObject],
    predictions: List[PredictedTrajectory],
    horizon: float,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
    supervisor=None,
    scenario_name: Optional[str] = None,
    scenario_status: Optional[str] = None,
    scenario_time: Optional[float] = None,
    scenario_timeout: Optional[float] = None,
    planner_algorithm: str = "astar",
    demo_active: bool = False,
) -> None:
    """Draw right-side panel: Controls, Scenario Status, Telemetry, Safety, Prediction, Legend."""
    panel_x = config.SIM_AREA_WIDTH
    panel_y = config.SIM_AREA_TOP
    panel_w = config.PANEL_WIDTH
    panel_h = config.SCREEN_HEIGHT - config.SIM_AREA_TOP

    panel_rect = pygame.Rect(panel_x, panel_y, panel_w, panel_h)
    pygame.draw.rect(surface, config.C_PANEL_BG, panel_rect)

    pygame.draw.line(surface, config.C_DIVIDER, (panel_x, panel_y), (panel_x, panel_y + panel_h))

    # --- 1. Controls Section ---
    heading_font = pygame.font.SysFont("segoeui", config.FONT_SIZE_TITLE, bold=True)
    ctrl_heading = heading_font.render("Controls", True, config.C_ACCENT)
    surface.blit(ctrl_heading, (panel_x + 14, panel_y + 2))

    for btn in buttons:
        btn.draw(surface, small_font)

    # Divider after buttons
    btn_bottom = max((btn.rect.bottom for btn in buttons), default=panel_y + 252)
    div_y = btn_bottom + 5
    pygame.draw.line(surface, config.C_DIVIDER, (panel_x + 14, div_y), (panel_x + panel_w - 14, div_y))

    # --- 2. Scenario & Telemetry ---
    tele_heading = heading_font.render("Telemetry & Safety", True, config.C_ACCENT)
    surface.blit(tele_heading, (panel_x + 14, div_y + 2))

    ty = div_y + 20
    if demo_active:
        demo_tag = small_font.render("[DEMO MODE ACTIVE]", True, config.C_DEMO_ACCENT)
        surface.blit(demo_tag, (panel_x + 16, ty))
        ty += 14

    algo_label = config.PLANNER_LABELS.get(planner_algorithm, planner_algorithm.upper())
    algo_surf = small_font.render(f"Planner: {algo_label}", True, (130, 210, 255))
    surface.blit(algo_surf, (panel_x + 16, ty))
    ty += 14

    if scenario_name:
        scen_surf = small_font.render(f"Scen: {scenario_name[:24]}", True, config.C_ACCENT)
        surface.blit(scen_surf, (panel_x + 16, ty))
        ty += 14

        out_col = config.C_STATUS_SUCCESS if scenario_status == "COMPLETED" else (
            config.C_STATUS_FAILED if scenario_status in ("TIMEOUT", "NO_SAFE_PATH") else (
                config.C_STATUS_OUTDATED if scenario_status == "EMERGENCY_STOP" else config.C_TEXT_PRIMARY
            )
        )
        time_part = f" ({scenario_time:.1f}s / {scenario_timeout:.0f}s)" if scenario_time is not None and scenario_timeout else ""
        out_surf = small_font.render(f"Status: [{scenario_status or 'RUNNING'}]{time_part}", True, out_col)
        surface.blit(out_surf, (panel_x + 16, ty))
        ty += 14

    state_col = config.C_TEXT_PRIMARY
    if vehicle.state == VehicleState.MOVING:
        state_col = config.C_STATUS_SUCCESS
    elif vehicle.state == VehicleState.BLOCKED:
        state_col = config.C_STATUS_FAILED
    elif vehicle.state == VehicleState.ARRIVED:
        state_col = (100, 220, 255)
    elif vehicle.state == VehicleState.PAUSED:
        state_col = config.C_STATUS_OUTDATED

    wp_text = f"{vehicle.target_waypoint_index}/{vehicle.total_waypoints}"
    telemetry_lines = [
        (f"Veh: {vehicle.state.value} | Spd: {vehicle.speed:.1f}px/s", state_col),
        (f"Heading: {vehicle.heading_degrees:.1f}° | WP: {wp_text}", config.C_TEXT_PRIMARY),
    ]

    for line, col in telemetry_lines:
        surf = small_font.render(line, True, col)
        surface.blit(surf, (panel_x + 16, ty))
        ty += 13

    # Safety Supervisor Status block
    if supervisor is not None:
        sup_state_str = supervisor.state.value
        sup_col = {
            "SAFE": config.C_SAFETY_SAFE,
            "CAUTION": config.C_SAFETY_CAUTION,
            "REPLANNING": config.C_SAFETY_REPLANNING,
            "EMERGENCY_STOP": config.C_SAFETY_EMERGENCY,
            "NO_SAFE_PATH": config.C_SAFETY_NO_PATH,
        }.get(sup_state_str, config.C_TEXT_PRIMARY)

        sup_line = f"Safety: [{sup_state_str}] | Spd: {int(supervisor.speed_multiplier * 100)}%"
        surf_sup = small_font.render(sup_line, True, sup_col)
        surface.blit(surf_sup, (panel_x + 16, ty))
        ty += 13

        conf_cnt = len(supervisor.active_conflicts)
        replan_cnt = supervisor.replan_count
        conf_line = f"Conflicts: {conf_cnt} | Replans: {replan_cnt}"
        surf_conf = small_font.render(conf_line, True, config.C_TEXT_SECONDARY)
        surface.blit(surf_conf, (panel_x + 16, ty))
        ty += 13

    # Divider after telemetry/safety
    div2_y = ty + 2
    pygame.draw.line(surface, config.C_DIVIDER, (panel_x + 14, div2_y), (panel_x + panel_w - 14, div2_y))

    # --- 3. Motion Prediction Dashboard ---
    pred_heading = heading_font.render(f"Motion Prediction ({horizon:.1f}s)", True, config.C_ACCENT)
    surface.blit(pred_heading, (panel_x + 14, div2_y + 2))

    py_val = div2_y + 18
    if not predictions:
        none_surf = small_font.render("No obstacles in view", True, config.C_TEXT_SECONDARY)
        surface.blit(none_surf, (panel_x + 16, py_val))
        py_val += 13
    else:
        for traj in predictions[:2]:
            conf_col = config.C_CONFIDENCE_HIGH if traj.confidence_level == "HIGH" else (
                config.C_CONFIDENCE_MEDIUM if traj.confidence_level == "MEDIUM" else config.C_CONFIDENCE_LOW
            )
            final_st = traj.final_state
            unc_r = final_st.uncertainty_radius if final_st else 0.0

            line1 = f"#{traj.track_id} {traj.category.value} [{traj.confidence_level}]"
            surf1 = small_font.render(line1, True, conf_col)
            surface.blit(surf1, (panel_x + 16, py_val))

            line2 = f" Pos:({int(traj.current_x)},{int(traj.current_y)}) V:({traj.current_vx:.1f},{traj.current_vy:.1f}) +/-{unc_r:.1f}"
            surf2 = small_font.render(line2, True, config.C_TEXT_SECONDARY)
            surface.blit(surf2, (panel_x + 16, py_val + 11))

            py_val += 23

    # Divider after prediction
    div3_y = max(py_val + 2, div2_y + 68)
    pygame.draw.line(surface, config.C_DIVIDER, (panel_x + 14, div3_y), (panel_x + panel_w - 14, div3_y))

    # --- 4. Legend & Hotkeys ---
    leg_heading = heading_font.render("Legend & Hotkeys", True, config.C_ACCENT)
    surface.blit(leg_heading, (panel_x + 14, div3_y + 2))

    ly = div3_y + 18
    # 2-column legend to fit nicely
    col1_items = _BASE_LEGEND[:3]
    col2_items = _BASE_LEGEND[3:]

    for (c1, l1), (c2, l2) in zip(col1_items, col2_items):
        # Col 1
        _draw_swatch(surface, panel_x + 16, ly + 2, c1)
        surface.blit(small_font.render(l1, True, config.C_TEXT_SECONDARY), (panel_x + 34, ly))
        # Col 2
        _draw_swatch(surface, panel_x + 160, ly + 2, c2)
        surface.blit(small_font.render(l2, True, config.C_TEXT_SECONDARY), (panel_x + 178, ly))
        ly += 13

    sel_fill, _ = _TYPE_COLOURS.get(selected_obs_type, ((180, 180, 180), (220, 220, 220)))
    _draw_swatch(surface, panel_x + 16, ly + 2, sel_fill)
    active_str = f"Spawn Type: {selected_obs_type.value}"
    surface.blit(small_font.render(active_str, True, config.C_TEXT_PRIMARY), (panel_x + 34, ly))
    ly += 15

    hint1 = small_font.render("D: Demo Mode | B: Benchmarks | M: Menu", True, config.C_ACCENT)
    surface.blit(hint1, (panel_x + 14, ly))
    ly += 12

    hint2 = small_font.render("SPACE: Run/Pause | P: Plan | A: Toggle Algo", True, (160, 220, 255))
    surface.blit(hint2, (panel_x + 14, ly))
    ly += 12

    hint3 = small_font.render("F1-F5: Load Scen | 1-6: Obstacles | R: Reset", True, config.C_TEXT_SECONDARY)
    surface.blit(hint3, (panel_x + 14, ly))


# ---------------------------------------------------------------------------
# Scenario Selection Menu Overlay (Phase 8)
# ---------------------------------------------------------------------------
def get_scenario_menu_card_rect() -> pygame.Rect:
    return pygame.Rect(140, 60, 1000, 600)


def get_scenario_menu_item_rects(total_items: int = 5) -> List[pygame.Rect]:
    rects = []
    card = get_scenario_menu_card_rect()
    start_y = card.y + 70
    item_h = 75
    gap = 12
    for i in range(total_items):
        r = pygame.Rect(card.x + 24, start_y + i * (item_h + gap), 380, item_h)
        rects.append(r)
    return rects


def draw_scenario_menu(
    surface: pygame.Surface,
    scenarios: list,
    selected_idx: int,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
) -> None:
    """Render full-featured modal dialog for selecting Indian road scenarios."""
    # Darkened backdrop
    overlay = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill(config.C_MENU_BG)
    surface.blit(overlay, (0, 0))

    card_rect = get_scenario_menu_card_rect()
    pygame.draw.rect(surface, config.C_MENU_CARD_BG, card_rect, border_radius=8)
    pygame.draw.rect(surface, config.C_ACCENT, card_rect, width=2, border_radius=8)

    # Top Header
    header_surf = title_font.render("INDIAN ROAD SCENARIOS - PHASE 8", True, config.C_TEXT_PRIMARY)
    surface.blit(header_surf, (card_rect.x + 24, card_rect.y + 16))

    sub_surf = small_font.render(
        "Select a realistic unstructured road condition to test adaptive planning & collision avoidance.",
        True,
        config.C_TEXT_SECONDARY,
    )
    surface.blit(sub_surf, (card_rect.x + 24, card_rect.y + 42))

    # Divider
    pygame.draw.line(
        surface,
        config.C_DIVIDER,
        (card_rect.x + 16, card_rect.y + 60),
        (card_rect.right - 16, card_rect.y + 60),
    )

    item_rects = get_scenario_menu_item_rects(len(scenarios))
    mouse_pos = pygame.mouse.get_pos()

    # Left Column: Scenario List Items
    for i, sc in enumerate(scenarios):
        rect = item_rects[i]
        is_sel = (i == selected_idx)
        is_hov = rect.collidepoint(mouse_pos)

        bg_col = config.C_MENU_CARD_ACTIVE if is_sel else (config.C_BUTTON_HOVER if is_hov else (16, 24, 48))
        border_col = config.C_ACCENT if is_sel else config.C_MENU_CARD_BORDER

        pygame.draw.rect(surface, bg_col, rect, border_radius=6)
        pygame.draw.rect(surface, border_col, rect, width=2 if is_sel else 1, border_radius=6)

        # Index badge [1] to [5]
        badge_rect = pygame.Rect(rect.x + 10, rect.y + 10, 26, 22)
        pygame.draw.rect(surface, config.C_MENU_BADGE if is_sel else (35, 50, 85), badge_rect, border_radius=4)
        b_num = small_font.render(f"F{i+1}", True, (255, 255, 255))
        surface.blit(b_num, (badge_rect.centerx - b_num.get_width() // 2, badge_rect.centery - b_num.get_height() // 2))

        # Scenario Name
        name_col = config.C_ACCENT if is_sel else config.C_TEXT_PRIMARY
        name_surf = body_font.render(sc.name, True, name_col)
        surface.blit(name_surf, (rect.x + 44, rect.y + 10))

        # Road type tag
        tag_str = f"Geometry: {sc.road_type.upper()} | Timeout: {sc.timeout:.0f}s"
        tag_surf = small_font.render(tag_str, True, config.C_TEXT_SECONDARY)
        surface.blit(tag_surf, (rect.x + 44, rect.y + 34))

        # Selection indicator
        if is_sel:
            arrow_surf = body_font.render(">", True, config.C_ACCENT)
            surface.blit(arrow_surf, (rect.right - 20, rect.centery - arrow_surf.get_height() // 2))

    # Right Column: Detailed Scenario Information Card
    if 0 <= selected_idx < len(scenarios):
        sc_sel = scenarios[selected_idx]
        detail_x = card_rect.x + 424
        detail_y = card_rect.y + 70
        detail_w = card_rect.width - 448
        detail_h = 440

        d_rect = pygame.Rect(detail_x, detail_y, detail_w, detail_h)
        pygame.draw.rect(surface, (14, 22, 44), d_rect, border_radius=6)
        pygame.draw.rect(surface, config.C_MENU_CARD_BORDER, d_rect, width=1, border_radius=6)

        # Scenario Title
        st_surf = title_font.render(sc_sel.name, True, config.C_ACCENT)
        surface.blit(st_surf, (detail_x + 16, detail_y + 14))

        # Road & Environment badges
        badges_y = detail_y + 44
        badge_text = f"Road: {sc_sel.road_type.upper()}  |  Seed: {sc_sel.seed}  |  Timeout: {sc_sel.timeout:.0f}s"
        b_surf = small_font.render(badge_text, True, config.C_TEXT_SECONDARY)
        surface.blit(b_surf, (detail_x + 16, badges_y))

        # Description header
        dh_y = badges_y + 28
        dh_surf = body_font.render("Scenario Description & Conditions:", True, config.C_TEXT_PRIMARY)
        surface.blit(dh_surf, (detail_x + 16, dh_y))

        # Description wrapped text
        words = sc_sel.description.split(" ")
        lines = []
        cur_line = ""
        for w in words:
            test_l = cur_line + (" " if cur_line else "") + w
            if body_font.size(test_l)[0] < detail_w - 36:
                cur_line = test_l
            else:
                lines.append(cur_line)
                cur_line = w
        if cur_line:
            lines.append(cur_line)

        desc_y = dh_y + 24
        for l in lines:
            ls = body_font.render(l, True, config.C_TEXT_SECONDARY)
            surface.blit(ls, (detail_x + 16, desc_y))
            desc_y += 20

        # Technical Obstacle Breakdown
        obs_y = desc_y + 16
        obs_h = body_font.render("Obstacle Inventory:", True, config.C_TEXT_PRIMARY)
        surface.blit(obs_h, (detail_x + 16, obs_y))
        obs_y += 22

        static_cnt = len(sc_sel.obstacle_positions)
        dyn_cnt = len(sc_sel.dynamic_obstacles)
        ev_cnt = len(sc_sel.scheduled_events)

        items_info = [
            f"• Static Obstacles: {static_cnt} roadside obstacles / road barriers",
            f"• Dynamic Road Users: {dyn_cnt} active entities (Cars, Moto, Pedestrians, Cattle)",
            f"• Scheduled Timeline Events: {ev_cnt} triggered dynamic events",
        ]
        for it in items_info:
            it_s = small_font.render(it, True, (200, 220, 245))
            surface.blit(it_s, (detail_x + 20, obs_y))
            obs_y += 18

        # Expected Safety Behavior
        beh_y = obs_y + 12
        beh_h = body_font.render("Autonomous Safety Goal:", True, config.C_TEXT_PRIMARY)
        surface.blit(beh_h, (detail_x + 16, beh_y))
        beh_y += 20

        goals = {
            "village_road": "Navigate narrow corridor within drivable bounds, steering around stones.",
            "urban_intersection": "Anticipate cross-traffic trajectories, stop/slow at mouth, clear safely.",
            "highway_merge": "Evaluate highway gaps, adjust speed behind slow vehicles, merge safely.",
            "dense_market": "Moderate speed under CAUTION, replan around vendor stalls and pedestrians.",
            "cattle_crossing": "Detect unexpected cow entering roadway, execute emergency stop, recover to SAFE.",
        }
        g_text = goals.get(sc_sel.id, "Follow safe route and arrive at destination safely.")
        gs = small_font.render(g_text, True, config.C_SAFETY_SAFE)
        surface.blit(gs, (detail_x + 20, beh_y))

    # Bottom Actions Bar
    btn_y = card_rect.bottom - 46
    pygame.draw.line(surface, config.C_DIVIDER, (card_rect.x + 16, btn_y - 8), (card_rect.right - 16, btn_y - 8))

    # Action hint labels
    act1 = body_font.render("[ENTER] Load Scenario", True, config.C_SAFETY_SAFE)
    surface.blit(act1, (card_rect.x + 28, btn_y))

    act2 = body_font.render("[F5] Restart Current", True, config.C_ACCENT)
    surface.blit(act2, (card_rect.x + 240, btn_y))

    act3 = body_font.render("[ESC] Back to Simulation", True, config.C_TEXT_SECONDARY)
    surface.blit(act3, (card_rect.x + 440, btn_y))

    nav_hint = small_font.render("Use UP/DOWN or F1-F5 keys to select | Click to choose", True, config.C_TEXT_SECONDARY)
    surface.blit(nav_hint, (card_rect.right - nav_hint.get_width() - 24, btn_y + 4))


# ---------------------------------------------------------------------------
# Phase 9: Benchmark Performance Screen
# ---------------------------------------------------------------------------
def get_benchmark_screen_card_rect() -> pygame.Rect:
    """Return bounding box for the Phase 9 benchmark results modal card."""
    card_w = min(1180, config.SCREEN_WIDTH - 40)
    card_h = min(640, config.SCREEN_HEIGHT - 40)
    card_x = (config.SCREEN_WIDTH - card_w) // 2
    card_y = (config.SCREEN_HEIGHT - card_h) // 2
    return pygame.Rect(card_x, card_y, card_w, card_h)


def draw_benchmark_screen(
    surface: pygame.Surface,
    benchmark_suite,
    title_font: pygame.font.Font,
    body_font: pygame.font.Font,
    small_font: pygame.font.Font,
    status_msg: str = "",
) -> None:
    """Render full-featured modal dialog for Phase 9 performance evaluation & comparisons."""
    # Dark semi-transparent backdrop
    overlay = pygame.Surface((config.SCREEN_WIDTH, config.SCREEN_HEIGHT), pygame.SRCALPHA)
    overlay.fill(config.C_BENCHMARK_BG)
    surface.blit(overlay, (0, 0))

    card_rect = get_benchmark_screen_card_rect()
    pygame.draw.rect(surface, (16, 24, 46), card_rect, border_radius=8)
    pygame.draw.rect(surface, config.C_ACCENT, card_rect, width=2, border_radius=8)

    # 1. Header
    h_surf = title_font.render("PHASE 9: PERFORMANCE EVALUATION & BENCHMARKING", True, config.C_TEXT_PRIMARY)
    surface.blit(h_surf, (card_rect.x + 24, card_rect.y + 16))

    sub_text = "Comparative analysis of A*, Dijkstra, and Greedy Best-First Search on Indian road scenarios."
    sub_surf = small_font.render(sub_text, True, config.C_TEXT_SECONDARY)
    surface.blit(sub_surf, (card_rect.x + 24, card_rect.y + 42))

    # Divider
    pygame.draw.line(surface, config.C_DIVIDER, (card_rect.x + 16, card_rect.y + 64), (card_rect.right - 16, card_rect.y + 64))

    # 2. Control Actions Bar
    ctrl_y = card_rect.y + 74
    hints = [
        ("[ENTER / R] Run Benchmarks", config.C_SAFETY_SAFE),
        ("[E] Export CSV & Charts", (245, 170, 40)),
        ("[ESC / B] Back to Simulation", config.C_TEXT_PRIMARY),
    ]
    hx = card_rect.x + 24
    for label, col in hints:
        s = body_font.render(label, True, col)
        surface.blit(s, (hx, ctrl_y))
        hx += s.get_width() + 32

    if status_msg:
        st_surf = small_font.render(status_msg, True, (100, 230, 255))
        surface.blit(st_surf, (card_rect.right - st_surf.get_width() - 28, ctrl_y + 2))

    # Divider before table
    table_top = ctrl_y + 28
    pygame.draw.line(surface, config.C_DIVIDER, (card_rect.x + 16, table_top), (card_rect.right - 16, table_top))

    # 3. Results Table
    summary = benchmark_suite.get_summary() if benchmark_suite else []
    col_x = [
        card_rect.x + 24,    # Scenario (210px)
        card_rect.x + 234,   # Algorithm (110px)
        card_rect.x + 354,   # Plan Time (ms) (160px)
        card_rect.x + 524,   # Nodes Exp (120px)
        card_rect.x + 654,   # Path Cost (100px)
        card_rect.x + 764,   # Success Rate (110px)
        card_rect.x + 884,   # Conflict Risks (110px)
        card_rect.x + 1004,  # Collisions (90px)
    ]
    headers = [
        "Scenario", "Algorithm", "Plan Time (ms)", "Expanded Nodes", "Path Cost", "Success Rate", "Risk Warnings", "Collisions"
    ]

    header_rect = pygame.Rect(card_rect.x + 16, table_top + 6, card_rect.width - 32, 26)
    pygame.draw.rect(surface, config.C_BENCHMARK_TABLE_HEADER, header_rect, border_radius=4)

    for cx, h_name in zip(col_x, headers):
        hs = small_font.render(h_name, True, (255, 255, 255))
        surface.blit(hs, (cx, table_top + 10))

    row_y = table_top + 36
    row_h = 24
    max_visible_rows = (card_rect.bottom - row_y - 36) // row_h

    if not summary:
        empty_msg = "No benchmark runs executed yet. Press [ENTER] to run A* vs Dijkstra vs GBFS across all scenarios."
        es = body_font.render(empty_msg, True, config.C_TEXT_SECONDARY)
        surface.blit(es, (card_rect.x + 36, row_y + 30))
    else:
        for idx, row in enumerate(summary[:max_visible_rows]):
            is_alt = (idx % 2 == 1)
            if is_alt:
                r_rect = pygame.Rect(card_rect.x + 16, row_y, card_rect.width - 32, row_h)
                pygame.draw.rect(surface, config.C_BENCHMARK_TABLE_ROW_ALT, r_rect, border_radius=3)

            sc_label = row["scenario_name"].split(". ")[-1] if ". " in row["scenario_name"] else row["scenario_name"]
            algo_str = row["algorithm"].upper()

            algo_col = (100, 180, 255) if algo_str == "ASTAR" else (
                (255, 120, 120) if algo_str == "DIJKSTRA" else (100, 225, 150)
            )

            pt_mean = f"{row['plan_time_mean_ms']:.2f} ± {row['plan_time_std_dev_ms'] or 0.0:.2f}" if row['plan_time_mean_ms'] is not None else "N/A"
            ne_mean = f"{int(row['nodes_expanded_mean'])}" if row['nodes_expanded_mean'] is not None else "N/A"
            pc_mean = f"{row['path_cost_mean']:.1f}" if row['path_cost_mean'] is not None else "N/A"
            succ_pct = f"{row['success_rate']*100:.0f}%" if row['success_rate'] is not None else "N/A"
            risks_cnt = str(row.get('total_collision_risk_events', 0))
            coll_cnt = str(row.get('total_actual_collisions', 0))

            vals = [
                (sc_label[:20], config.C_TEXT_PRIMARY),
                (algo_str, algo_col),
                (pt_mean, config.C_TEXT_PRIMARY),
                (ne_mean, config.C_TEXT_PRIMARY),
                (pc_mean, config.C_TEXT_PRIMARY),
                (succ_pct, config.C_SAFETY_SAFE if row['success_rate'] == 1.0 else (240, 180, 50)),
                (risks_cnt, (255, 200, 80)),
                (coll_cnt, config.C_SAFETY_SAFE if coll_cnt == "0" else config.C_SAFETY_EMERGENCY),
            ]

            for cx, (val_str, vcol) in zip(col_x, vals):
                vs = small_font.render(val_str, True, vcol)
                surface.blit(vs, (cx, row_y + 4))

            row_y += row_h

    # 4. Footer
    footer_y = card_rect.bottom - 28
    pygame.draw.line(surface, config.C_DIVIDER, (card_rect.x + 16, footer_y - 4), (card_rect.right - 16, footer_y - 4))
    runs_count = len(benchmark_suite.all_runs) if benchmark_suite else 0
    foot_text = f"Total Runs Logged: {runs_count} | Results stored in reports/ | Deterministic & reproducible"
    fs = small_font.render(foot_text, True, config.C_TEXT_SECONDARY)
    surface.blit(fs, (card_rect.x + 24, footer_y))
