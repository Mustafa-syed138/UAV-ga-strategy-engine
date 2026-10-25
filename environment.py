"""
environment.py
--------------
Step 1: The discretized world (grid) in which the UAV fleet operates.

The grid is a GRID_SIZE x GRID_SIZE map. Each cell (x, y) stores three layers:

    no_fly[y, x]    -> True if the cell is a restricted / blocked area
    hazard[y, x]    -> risk value in [0, 1]  (0 = safe, 1 = very dangerous)
    priority[y, x]  -> importance value in [0, 1] (0 = ignore, 1 = critical
                       disaster zone that must be surveyed)

Conventions
-----------
* Coordinates are (x, y) = (column, row).
* Arrays are indexed [row, column] = [y, x], as usual in NumPy.
"""

import numpy as np


class Environment:
    def __init__(self, size=50, seed=42, n_no_fly=4, n_hazards=5,
                 n_priority=4, base=(2, 2)):
        self.size = size
        self.seed = seed
        self.rng = np.random.default_rng(seed)   # reproducible results
        self.base = base                          # UAV launch / landing cell

        # The three layers of the map
        self.no_fly = np.zeros((size, size), dtype=bool)
        self.hazard = np.zeros((size, size), dtype=float)
        self.priority = np.zeros((size, size), dtype=float)

        self._add_no_fly_zones(n_no_fly)
        self._add_hazard_zones(n_hazards)
        self._add_priority_zones(n_priority)

        # Keep the base area always safe and open
        self._clear_base_area(radius=3)

    # ------------------------------------------------------------------
    # Map generation helpers
    # ------------------------------------------------------------------
    def _circle_mask(self, cx, cy, r):
        """Boolean mask of all cells within distance r of (cx, cy)."""
        yy, xx = np.ogrid[:self.size, :self.size]
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2

    def _add_no_fly_zones(self, n):
        """Rectangular blocked areas (buildings, restricted airspace)."""
        for _ in range(n):
            w, h = self.rng.integers(3, 8, size=2)
            x0 = self.rng.integers(8, self.size - w)
            y0 = self.rng.integers(8, self.size - h)
            self.no_fly[y0:y0 + h, x0:x0 + w] = True

    def _add_hazard_zones(self, n):
        """Circular pockets of risk (wind, radiation). Risk fades from the
        centre outwards, so the middle of a pocket is the most dangerous."""
        yy, xx = np.mgrid[:self.size, :self.size]
        for _ in range(n):
            cx, cy = self.rng.integers(5, self.size - 5, size=2)
            r = self.rng.integers(4, 9)
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            risk = np.clip(1 - dist / r, 0, 1)          # 1 at centre, 0 at edge
            self.hazard = np.maximum(self.hazard, risk)

    def _add_priority_zones(self, n):
        """High-priority disaster areas that the UAVs must survey."""
        yy, xx = np.mgrid[:self.size, :self.size]
        for _ in range(n):
            cx, cy = self.rng.integers(8, self.size - 4, size=2)
            r = self.rng.integers(4, 8)
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            value = np.clip(1 - dist / r, 0, 1)
            self.priority = np.maximum(self.priority, value)
        # Nothing worth surveying inside a no-fly zone
        self.priority[self.no_fly] = 0.0

    def _clear_base_area(self, radius=3):
        mask = self._circle_mask(self.base[0], self.base[1], radius)
        self.no_fly[mask] = False
        self.hazard[mask] = 0.0
        self.priority[mask] = 0.0

    # ------------------------------------------------------------------
    # Queries used later by the fitness function / repair mechanism
    # ------------------------------------------------------------------
    def in_bounds(self, x, y):
        return 0 <= x < self.size and 0 <= y < self.size

    def is_no_fly(self, x, y):
        """True if (x, y) is outside the map or inside a no-fly zone."""
        x, y = int(round(x)), int(round(y))
        if not self.in_bounds(x, y):
            return True
        return bool(self.no_fly[y, x])

    def hazard_at(self, x, y):
        x, y = int(round(x)), int(round(y))
        return float(self.hazard[y, x]) if self.in_bounds(x, y) else 1.0

    def priority_at(self, x, y):
        x, y = int(round(x)), int(round(y))
        return float(self.priority[y, x]) if self.in_bounds(x, y) else 0.0

    def total_priority(self):
        """Maximum achievable coverage score (used to normalise fitness)."""
        return float(self.priority.sum())

    def nearest_free_cell(self, x, y):
        """Repair helper: closest cell that is NOT a no-fly cell."""
        x = int(np.clip(round(x), 0, self.size - 1))
        y = int(np.clip(round(y), 0, self.size - 1))
        if not self.no_fly[y, x]:
            return x, y
        free = np.argwhere(~self.no_fly)                 # rows of [y, x]
        d = (free[:, 1] - x) ** 2 + (free[:, 0] - y) ** 2
        fy, fx = free[np.argmin(d)]
        return int(fx), int(fy)

    def random_free_cell(self):
        """Random valid cell (used in population initialisation)."""
        free = np.argwhere(~self.no_fly)
        fy, fx = free[self.rng.integers(len(free))]
        return int(fx), int(fy)

    def summary(self):
        n = self.size * self.size
        return (f"Grid {self.size}x{self.size} | base={self.base}\n"
                f"  no-fly cells : {int(self.no_fly.sum())} "
                f"({100 * self.no_fly.sum() / n:.1f}%)\n"
                f"  hazard cells : {int((self.hazard > 0).sum())} "
                f"({100 * (self.hazard > 0).sum() / n:.1f}%)\n"
                f"  priority cells: {int((self.priority > 0).sum())} "
                f"({100 * (self.priority > 0).sum() / n:.1f}%)\n"
                f"  total priority value: {self.total_priority():.1f}")
