"""
src/road.py
===========
Renders road geometries for Phase 8 realistic Indian road scenarios:
- straight: Standard horizontal 2-lane road with dashed centerline
- village: Narrow rural road without lane markings, uneven gravel shoulders
- intersection: Unsignalized 4-way urban intersection
- merge: Highway with merging on-ramp lane
- market: Dense market street flanked by shop stalls and curbs
- cattle: Rural road with open grassy verges
"""

from typing import Optional
import pygame
import config


def draw_road(surface: pygame.Surface, road_type: str = "straight") -> None:
    """Draw the road surface, shoulder, edges, and markings according to scenario road_type."""
    sim_x, sim_y, sim_w, sim_h = config.SIM_RECT

    if road_type == "village":
        _draw_village_road(surface, sim_x, sim_y, sim_w, sim_h)
    elif road_type == "intersection":
        _draw_intersection_road(surface, sim_x, sim_y, sim_w, sim_h)
    elif road_type == "merge":
        _draw_merge_road(surface, sim_x, sim_y, sim_w, sim_h)
    elif road_type == "market":
        _draw_market_road(surface, sim_x, sim_y, sim_w, sim_h)
    else:  # "straight" or "cattle"
        _draw_straight_road(surface, sim_x, sim_y, sim_w, sim_h, is_rural=(road_type == "cattle"))


def _draw_straight_road(surface: pygame.Surface, sim_x: int, sim_y: int, sim_w: int, sim_h: int, is_rural: bool = False) -> None:
    """Standard horizontal 2-lane road with dashed centre line."""
    road_rect = pygame.Rect(
        sim_x,
        config.ROAD_TOP_Y,
        sim_w,
        config.ROAD_BOTTOM_Y - config.ROAD_TOP_Y,
    )

    shoulder_pad = config.GRID_CELL_SIZE // 2
    shoulder_rect = road_rect.inflate(0, shoulder_pad * 2)
    shoulder_col = (50, 75, 45) if is_rural else config.C_ROAD_SHOULDER
    pygame.draw.rect(surface, shoulder_col, shoulder_rect)

    pygame.draw.rect(surface, config.C_ROAD_SURFACE, road_rect)

    pygame.draw.line(surface, config.C_ROAD_EDGE, (sim_x, config.ROAD_TOP_Y), (sim_x + sim_w, config.ROAD_TOP_Y), 2)
    pygame.draw.line(surface, config.C_ROAD_EDGE, (sim_x, config.ROAD_BOTTOM_Y), (sim_x + sim_w, config.ROAD_BOTTOM_Y), 2)

    # Dashed centre line
    dash_len = 30
    gap_len = 20
    y_centre = config.ROAD_CENTER_Y
    x_cursor = sim_x
    drawing = True
    while x_cursor < sim_x + sim_w:
        if drawing:
            end_x = min(x_cursor + dash_len, sim_x + sim_w)
            pygame.draw.line(surface, config.C_ROAD_MARKING, (x_cursor, y_centre), (end_x, y_centre), 2)
        x_cursor += dash_len if drawing else gap_len
        drawing = not drawing


def _draw_village_road(surface: pygame.Surface, sim_x: int, sim_y: int, sim_w: int, sim_h: int) -> None:
    """Narrow rural road: no lane markings, uneven dirt shoulders."""
    half_w = config.GRID_CELL_SIZE * 1.5
    top_y = config.ROAD_CENTER_Y - int(half_w)
    bot_y = config.ROAD_CENTER_Y + int(half_w)

    # Dirt/gravel shoulder
    shoulder_rect = pygame.Rect(sim_x, top_y - 20, sim_w, (bot_y - top_y) + 40)
    pygame.draw.rect(surface, (95, 80, 55), shoulder_rect)

    # Narrow asphalt road surface
    road_rect = pygame.Rect(sim_x, top_y, sim_w, bot_y - top_y)
    pygame.draw.rect(surface, (58, 56, 54), road_rect)

    # Uneven roadside edges (faint brown/grey)
    pygame.draw.line(surface, (140, 130, 100), (sim_x, top_y), (sim_x + sim_w, top_y), 2)
    pygame.draw.line(surface, (140, 130, 100), (sim_x, bot_y), (sim_x + sim_w, bot_y), 2)
    # No centre lane markings on unmarked village roads


def _draw_intersection_road(surface: pygame.Surface, sim_x: int, sim_y: int, sim_w: int, sim_h: int) -> None:
    """4-way unsignalized urban intersection."""
    # Main horizontal road
    _draw_straight_road(surface, sim_x, sim_y, sim_w, sim_h)

    # Crossing vertical road
    mid_col = config.GRID_COLS // 2
    cross_w = config.GRID_CELL_SIZE * 4
    cross_x = (mid_col - 2) * config.GRID_CELL_SIZE

    vert_rect = pygame.Rect(cross_x, sim_y, cross_w, sim_h)
    pygame.draw.rect(surface, config.C_ROAD_SHOULDER, vert_rect.inflate(20, 0))
    pygame.draw.rect(surface, config.C_ROAD_SURFACE, vert_rect)

    # Vertical road edge lines (outside intersection)
    pygame.draw.line(surface, config.C_ROAD_EDGE, (cross_x, sim_y), (cross_x, config.ROAD_TOP_Y), 2)
    pygame.draw.line(surface, config.C_ROAD_EDGE, (cross_x, config.ROAD_BOTTOM_Y), (cross_x, sim_y + sim_h), 2)
    pygame.draw.line(surface, config.C_ROAD_EDGE, (cross_x + cross_w, sim_y), (cross_x + cross_w, config.ROAD_TOP_Y), 2)
    pygame.draw.line(surface, config.C_ROAD_EDGE, (cross_x + cross_w, config.ROAD_BOTTOM_Y), (cross_x + cross_w, sim_y + sim_h), 2)

    # Intersection box outline
    box_rect = pygame.Rect(cross_x, config.ROAD_TOP_Y, cross_w, config.ROAD_BOTTOM_Y - config.ROAD_TOP_Y)
    pygame.draw.rect(surface, (70, 70, 80), box_rect)
    pygame.draw.rect(surface, (180, 180, 180), box_rect, width=1)


def _draw_merge_road(surface: pygame.Surface, sim_x: int, sim_y: int, sim_w: int, sim_h: int) -> None:
    """Highway with on-ramp merging lane."""
    # Main highway
    _draw_straight_road(surface, sim_x, sim_y, sim_w, sim_h)

    # On-ramp merging lane from bottom-left
    ramp_top = config.ROAD_BOTTOM_Y
    ramp_bot = sim_y + sim_h
    ramp_w = config.GRID_CELL_SIZE * 2
    ramp_rect = pygame.Rect(sim_x, ramp_top, config.GRID_CELL_SIZE * 9, config.GRID_CELL_SIZE * 2)

    pygame.draw.rect(surface, (75, 70, 55), ramp_rect.inflate(0, 10))
    pygame.draw.rect(surface, config.C_ROAD_SURFACE, ramp_rect)

    # Tapering edge towards merge point
    merge_pt = (sim_x + config.GRID_CELL_SIZE * 9, config.ROAD_BOTTOM_Y)
    pygame.draw.line(surface, config.C_ROAD_EDGE, (sim_x, ramp_top + config.GRID_CELL_SIZE * 2), merge_pt, 2)
    pygame.draw.line(surface, (240, 220, 80), (sim_x, ramp_top), merge_pt, 1)


def _draw_market_road(surface: pygame.Surface, sim_x: int, sim_y: int, sim_w: int, sim_h: int) -> None:
    """Crowded market street with curb stripes and stall walkways."""
    _draw_straight_road(surface, sim_x, sim_y, sim_w, sim_h)

    # Striped curbs representing stalls and vendor walkways
    stall_h = 16
    for sx in range(sim_x, sim_x + sim_w, 24):
        col_stripe = (180, 70, 60) if (sx // 24) % 2 == 0 else (220, 210, 180)
        # Top stall curb
        pygame.draw.rect(surface, col_stripe, pygame.Rect(sx, config.ROAD_TOP_Y - stall_h, 24, stall_h))
        # Bottom stall curb
        pygame.draw.rect(surface, col_stripe, pygame.Rect(sx, config.ROAD_BOTTOM_Y, 24, stall_h))
