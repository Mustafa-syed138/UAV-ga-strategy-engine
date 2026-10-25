"""
fitness.py
----------
Step 3: Multi-objective fitness function + constraint handling.

    Fitness = + W_COV * Coverage          (maximise)
              + W_DET * DetectionRate     (maximise)
              - W_ENE * Energy            (minimise)
              - W_TIM * MissionTime       (minimise)
              - W_HAZ * HazardExposure    (minimise)
              - Penalties                 (hard constraints)

Every term is normalised to roughly 0..1 so the weights are meaningful.
Higher fitness = better plan.
"""

import numpy as np
from src.chromosome import SPEED_MIN, SPEED_MAX, N_DRONES

# Detection radius (in cells) for each sensor power level: off / low / high
DETECT_RADIUS = {0: 0.0, 1: 1.5, 2: 3.0}

# ---- Objective weights (justify these in the report) -----------------------
WEIGHTS = {"coverage": 0.35, "detection": 0.10,
           "energy": 0.20, "time": 0.10, "hazard": 0.25}

# ---- Energy model -----------------------------------------------------------
E_BASE = 1.0                 # energy per cell flown at zero speed (drag-free)
E_SPEED = 0.1                # extra energy per cell, grows with speed^2
SENSOR_COST = {0: 0.0, 1: 0.5, 2: 1.5}   # energy per time-unit, by sensor level

# ---- Hard constraints ---------------------------------------------------------
MAX_RANGE = 150.0            # max flight distance per drone (cells)
PEN_RANGE = 1.0              # penalty per 100% range overshoot
PEN_NOFLY = 0.05             # penalty per sampled path point inside a no-fly cell

STEP = 1.0                   # sampling distance along each leg (cells)

# ---- Normalisation references -------------------------------------------------
TIME_REF = MAX_RANGE / ((SPEED_MIN + SPEED_MAX) / 2)
ENERGY_REF = N_DRONES * MAX_RANGE * (E_BASE + E_SPEED * ((SPEED_MIN + SPEED_MAX) / 2) ** 2)
HAZARD_REF = N_DRONES * TIME_REF * 0.15


def make_targets(env, n=40, seed=7):
    """Hidden targets (victims / hotspots) placed mostly in high-priority areas.
    Returns array of shape (n, 2) with (x, y) positions."""
    rng = np.random.default_rng(seed)
    cells = np.argwhere(env.priority > 0)                  # [y, x]
    p = env.priority[cells[:, 0], cells[:, 1]]
    idx = rng.choice(len(cells), size=min(n, len(cells)), replace=False, p=p / p.sum())
    return cells[idx][:, ::-1].astype(float)               # -> (x, y)


def repair_chromosome(chrom, env):
    """REPAIR mechanism: force every gene back into its legal range and move
    any waypoint that sits in a no-fly cell to the nearest free cell."""
    for d in range(chrom.waypoints.shape[0]):
        for w in range(chrom.waypoints.shape[1]):
            x, y = chrom.waypoints[d, w]
            chrom.waypoints[d, w] = env.nearest_free_cell(x, y)
    chrom.speeds = np.clip(chrom.speeds, SPEED_MIN, SPEED_MAX)
    chrom.sensors = np.clip(np.rint(chrom.sensors), 0, 2).astype(int)
    return chrom


class FitnessEvaluator:
    def __init__(self, env, n_targets=40, weights=None):
        self.env = env
        self.weights = weights or WEIGHTS
        self.targets = make_targets(env, n_targets)
        # Pre-compute priority cells once (makes evaluation fast)
        cells = np.argwhere(env.priority > 0)                       # [y, x]
        self.pc_xy = cells[:, ::-1].astype(float)                   # (x, y)
        self.pc_val = env.priority[cells[:, 0], cells[:, 1]]
        self.total_priority = env.total_priority()

    # ------------------------------------------------------------------
    def _sample_drone(self, chrom, d):
        """Walk along drone d's route. Returns sampled points, and for each
        sample: sensor radius, speed, plus total distance & energy & time."""
        env = self.env
        path = chrom.full_path(env, d)                 # base -> wps -> base
        n_legs = len(path) - 1
        mean_speed = float(chrom.speeds[d].mean())

        pts, radii, hazard_exp = [], [], 0.0
        dist_total = energy = time_total = 0.0
        for i in range(n_legs):
            a, b = path[i], path[i + 1]
            L = float(np.linalg.norm(b - a))
            if i < n_legs - 1:                          # leg towards waypoint i
                v, s = chrom.speeds[d, i], int(chrom.sensors[d, i])
            else:                                       # final leg home: sensor off
                v, s = mean_speed, 0
            t = L / v
            dist_total += L
            time_total += t
            energy += L * (E_BASE + E_SPEED * v ** 2) + SENSOR_COST[s] * t

            n = max(2, int(np.ceil(L / STEP)) + 1)
            xs = np.linspace(a[0], b[0], n)
            ys = np.linspace(a[1], b[1], n)
            leg_pts = np.stack([xs, ys], axis=1)
            pts.append(leg_pts)
            radii.append(np.full(n, DETECT_RADIUS[s]))
            # hazard exposure = hazard * time spent (slow flight = more exposure)
            xi = np.clip(np.rint(xs).astype(int), 0, env.size - 1)
            yi = np.clip(np.rint(ys).astype(int), 0, env.size - 1)
            hazard_exp += float(env.hazard[yi, xi].sum()) * (t / n)

        pts = np.vstack(pts)
        radii = np.concatenate(radii)
        xi = np.clip(np.rint(pts[:, 0]).astype(int), 0, env.size - 1)
        yi = np.clip(np.rint(pts[:, 1]).astype(int), 0, env.size - 1)
        n_nofly = int(env.no_fly[yi, xi].sum())
        return dict(pts=pts, radii=radii, dist=dist_total, energy=energy,
                    time=time_total, hazard=hazard_exp, n_nofly=n_nofly)

    # ------------------------------------------------------------------
    def evaluate(self, chrom, repair=True):
        """Compute fitness, store it in chrom.fitness and return a breakdown."""
        if repair:
            repair_chromosome(chrom, self.env)

        runs = [self._sample_drone(chrom, d) for d in range(chrom.waypoints.shape[0])]

        # ---- Coverage & detection (union over all drones, no double counting)
        pts = np.vstack([r["pts"] for r in runs])
        radii = np.concatenate([r["radii"] for r in runs])
        on = radii > 0
        pts, radii = pts[on], radii[on]
        if len(pts):
            d_cells = np.linalg.norm(self.pc_xy[:, None, :] - pts[None, :, :], axis=2)
            covered = (d_cells <= radii[None, :]).any(axis=1)
            d_targ = np.linalg.norm(self.targets[:, None, :] - pts[None, :, :], axis=2)
            detected = (d_targ <= radii[None, :]).any(axis=1)
            coverage = float(self.pc_val[covered].sum() / self.total_priority)
            detection = float(detected.mean())
        else:
            coverage = detection = 0.0

        # ---- Costs
        energy = sum(r["energy"] for r in runs) / ENERGY_REF
        mission_time = max(r["time"] for r in runs) / TIME_REF   # fleet finishes with last drone
        hazard = sum(r["hazard"] for r in runs) / HAZARD_REF

        # ---- Penalties (hard constraints)
        pen_range = PEN_RANGE * sum(max(0.0, r["dist"] - MAX_RANGE) / MAX_RANGE for r in runs)
        pen_nofly = PEN_NOFLY * sum(r["n_nofly"] for r in runs)
        penalty = pen_range + pen_nofly

        w = self.weights
        fitness = (w["coverage"] * coverage + w["detection"] * detection
                   - w["energy"] * energy - w["time"] * mission_time
                   - w["hazard"] * hazard - penalty)

        chrom.fitness = float(fitness)
        return dict(fitness=float(fitness), coverage=coverage, detection=detection,
                    energy=energy, time=mission_time, hazard=hazard,
                    pen_range=pen_range, pen_nofly=pen_nofly,
                    distances=[r["dist"] for r in runs])
