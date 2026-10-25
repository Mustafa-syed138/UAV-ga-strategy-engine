"""
chromosome.py
-------------
Step 2: Chromosome representation (hybrid real-valued + integer encoding).

One chromosome = a complete mission plan for the WHOLE fleet.
It is split into one segment per drone, and each drone has W waypoints.
For every waypoint we store three genes:

    waypoints[d, w] = (x, y)     real-valued  -> where to fly
    speeds[d, w]                 real-valued  -> speed on the leg to this waypoint
    sensors[d, w]                integer      -> sensor power level (0=off, 1=low, 2=high)

Layout (D drones, W waypoints each):

    Drone 0: [(x,y,speed,sensor) x W] | Drone 1: [...] | Drone 2: [...]

Every route implicitly starts and ends at the base station.
"""

import numpy as np

# ---- Fleet / gene settings (easy to change) --------------------------------
N_DRONES = 3
N_WAYPOINTS = 10
SPEED_MIN, SPEED_MAX = 1.0, 5.0          # cells per time step
SENSOR_LEVELS = (0, 1, 2)                # off, low power, high power


class Chromosome:
    def __init__(self, waypoints, speeds, sensors):
        self.waypoints = waypoints   # shape (D, W, 2)  float
        self.speeds = speeds         # shape (D, W)     float
        self.sensors = sensors       # shape (D, W)     int
        self.fitness = None          # filled in by the fitness function (Step 3)

    # ------------------------------------------------------------------
    @classmethod
    def random(cls, env, rng, bias=0.5):
        """Create a random chromosome.

        bias = probability that a waypoint is placed on a high-priority cell
        (smart initialisation). The rest are placed on random free cells so
        the population starts diverse.
        """
        D, W = N_DRONES, N_WAYPOINTS
        priority_cells = np.argwhere(env.priority > 0.2)      # rows of [y, x]
        waypoints = np.zeros((D, W, 2))

        for d in range(D):
            for w in range(W):
                if len(priority_cells) > 0 and rng.random() < bias:
                    y, x = priority_cells[rng.integers(len(priority_cells))]
                    x += rng.normal(0, 1.5)                    # small jitter
                    y += rng.normal(0, 1.5)
                else:
                    x, y = rng.uniform(0, env.size - 1, size=2)
                waypoints[d, w] = env.nearest_free_cell(x, y)  # repair

        speeds = rng.uniform(SPEED_MIN, SPEED_MAX, size=(D, W))
        sensors = rng.choice(SENSOR_LEVELS, size=(D, W))
        return cls(waypoints, speeds, sensors)

    # ------------------------------------------------------------------
    def copy(self):
        c = Chromosome(self.waypoints.copy(), self.speeds.copy(), self.sensors.copy())
        c.fitness = self.fitness
        return c

    def full_path(self, env, d):
        """Route of drone d as a list of (x, y): base -> waypoints -> base."""
        pts = [tuple(env.base)] + [tuple(p) for p in self.waypoints[d]] + [tuple(env.base)]
        return np.array(pts, dtype=float)

    def describe(self, max_drones=1):
        """Print a readable version of the genes (useful for the report)."""
        for d in range(min(max_drones, N_DRONES)):
            print(f"Drone {d}:")
            for w in range(N_WAYPOINTS):
                x, y = self.waypoints[d, w]
                print(f"  WP{w:02d}  pos=({x:5.1f},{y:5.1f})  "
                      f"speed={self.speeds[d, w]:.2f}  sensor={self.sensors[d, w]}")
