"""
src/grid.py
===========
Occupancy grid for the simulation environment.
Stores which cells are free (0), which are occupied by obstacles (1),
and which are road vs non-road.

This module contains ONLY data logic - no rendering.
The grid is designed to be consumed directly by a future A* planner.
"""

from typing import Optional, Tuple, List
import numpy as np
import config

# Cell values
CELL_FREE     = 0
CELL_OCCUPIED = 1


class OccupancyGrid:
    """
    2-D occupancy grid aligned with the simulation canvas.

    Grid coordinates: (col, row) where
        col in [0, GRID_COLS)
        row in [0, GRID_ROWS)

    The grid stores:
        0  = free / traversable
        1  = occupied by an obstacle
    """

    def __init__(
        self,
        start_pos: Optional[Tuple[int, int]] = None,
        dest_pos: Optional[Tuple[int, int]] = None,
    ) -> None:
        self._grid = np.zeros(
            (config.GRID_ROWS, config.GRID_COLS), dtype=np.int8
        )
        self._road_mask = self._build_road_mask()
        self.start_pos = start_pos if start_pos is not None else (config.START_GRID_COL, config.START_GRID_ROW)
        self.dest_pos = dest_pos if dest_pos is not None else (config.DEST_GRID_COL, config.DEST_GRID_ROW)

    # ------------------------------------------------------------------ #
    # Road mask
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_road_mask() -> np.ndarray:
        """Return a boolean array: True for cells that lie on the road."""
        mask = np.zeros((config.GRID_ROWS, config.GRID_COLS), dtype=bool)
        top_row = config.ROAD_ROW_CENTER - config.ROAD_HALF_WIDTH
        bot_row = config.ROAD_ROW_CENTER + config.ROAD_HALF_WIDTH
        mask[top_row:bot_row, :] = True
        return mask

    def set_road_mask(self, mask: np.ndarray) -> None:
        """Set a custom boolean mask for drivable road cells in a scenario."""
        if mask is not None and mask.shape == (config.GRID_ROWS, config.GRID_COLS):
            self._road_mask = mask.astype(bool)

    def reset_road_mask(self) -> None:
        """Restore default straight road mask."""
        self._road_mask = self._build_road_mask()

    def is_road(self, col: int, row: int) -> bool:
        """Return True if (col, row) lies on the road surface."""
        if not self.in_bounds(col, row):
            return False
        return bool(self._road_mask[row, col])

    # ------------------------------------------------------------------ #
    # Bounds checking
    # ------------------------------------------------------------------ #
    def in_bounds(self, col: int, row: int) -> bool:
        """Return True if (col, row) is inside the grid."""
        return 0 <= col < config.GRID_COLS and 0 <= row < config.GRID_ROWS

    # ------------------------------------------------------------------ #
    # Protected cells (start & destination)
    # ------------------------------------------------------------------ #
    def is_protected(self, col: int, row: int) -> bool:
        """Return True if the cell is the start or destination marker."""
        if isinstance(self, int):
            c, r = self, col
            return (c == config.START_GRID_COL and r == config.START_GRID_ROW) or (c == config.DEST_GRID_COL and r == config.DEST_GRID_ROW)
        if (col, row) == self.start_pos or (col, row) == self.dest_pos:
            return True
        if col == config.START_GRID_COL and row == config.START_GRID_ROW:
            return True
        if col == config.DEST_GRID_COL and row == config.DEST_GRID_ROW:
            return True
        return False

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def is_free(self, col: int, row: int) -> bool:
        if not self.in_bounds(col, row):
            return False
        return bool(self._grid[row, col] == CELL_FREE)

    def is_occupied(self, col: int, row: int) -> bool:
        if not self.in_bounds(col, row):
            return False
        return bool(self._grid[row, col] == CELL_OCCUPIED)

    @property
    def obstacle_count(self) -> int:
        """Total number of occupied cells."""
        return int(np.sum(self._grid == CELL_OCCUPIED))

    @property
    def grid_array(self) -> np.ndarray:
        """Raw numpy array (read-only view) for A* or other algorithms."""
        view = self._grid.view()
        view.flags.writeable = False
        return view

    # ------------------------------------------------------------------ #
    # Mutators
    # ------------------------------------------------------------------ #
    def add_obstacle(self, col: int, row: int) -> bool:
        """
        Mark a cell as occupied.

        Returns True on success, False if the placement is invalid:
          - out of bounds
          - not on the road
          - already occupied
          - overlaps start or destination
        """
        if not self.in_bounds(col, row):
            return False
        if not self.is_road(col, row):
            return False
        if self.is_protected(col, row):
            return False
        if self.is_occupied(col, row):
            return False
        self._grid[row, col] = CELL_OCCUPIED
        return True

    def remove_obstacle(self, col: int, row: int) -> bool:
        """
        Clear an occupied cell back to free.
        Returns True if something was removed, False otherwise.
        """
        if not self.in_bounds(col, row):
            return False
        if not self.is_occupied(col, row):
            return False
        self._grid[row, col] = CELL_FREE
        return True

    def clear_obstacles(self) -> None:
        """Remove every obstacle, restoring the grid to all-free."""
        self._grid[:] = CELL_FREE

    def reset(self) -> None:
        """Full environment reset (same as clear for now)."""
        self.clear_obstacles()

    # ------------------------------------------------------------------ #
    # Convenience: list of occupied positions for iteration
    # ------------------------------------------------------------------ #
    def occupied_positions(self):
        """Yield (col, row) for every occupied cell."""
        rows, cols = np.where(self._grid == CELL_OCCUPIED)
        for r, c in zip(rows, cols):
            yield int(c), int(r)
