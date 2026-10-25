"""
operators.py
------------
Step 4: Genetic operators tailored to spatial (x, y) route data.

    tournament_selection  -> picks parents (tournament size = selection pressure)
    crossover             -> segment-swap OR geometric (blend) crossover
    mutation              -> Gaussian spatial shift, waypoint reset, speed tweak,
                             sensor change, waypoint-order swap
    population_diversity  -> measures how spread-out the population is
"""

import numpy as np
from src.chromosome import Chromosome, SPEED_MIN, SPEED_MAX, N_DRONES, N_WAYPOINTS


# ----------------------------------------------------------------------------
# SELECTION
# ----------------------------------------------------------------------------
def tournament_selection(population, k, rng):
    """Pick k random individuals and return a copy of the best one.
    Larger k  -> higher selection pressure (faster but riskier convergence).
    Smaller k -> more exploration and diversity."""
    idx = rng.choice(len(population), size=k, replace=False)
    best = max((population[i] for i in idx), key=lambda c: c.fitness)
    return best.copy()


# ----------------------------------------------------------------------------
# CROSSOVER
# ----------------------------------------------------------------------------
def segment_swap_crossover(p1, p2, rng):
    """Swap whole parts of the plan between parents.
    1) Drone-level swap: each drone's complete route comes from either parent.
    2) Waypoint-segment swap: inside a drone, a block of consecutive waypoints
       (with their speeds & sensors) is exchanged between the two parents."""
    c1, c2 = p1.copy(), p2.copy()
    for d in range(N_DRONES):
        if rng.random() < 0.5:                                   # swap entire drone
            for arr in ("waypoints", "speeds", "sensors"):
                a, b = getattr(c1, arr), getattr(c2, arr)
                a[d], b[d] = b[d].copy(), a[d].copy()
        else:                                                    # swap a segment
            i, j = sorted(rng.choice(N_WAYPOINTS + 1, size=2, replace=False))
            for arr in ("waypoints", "speeds", "sensors"):
                a, b = getattr(c1, arr), getattr(c2, arr)
                a[d, i:j], b[d, i:j] = b[d, i:j].copy(), a[d, i:j].copy()
    return c1, c2


def geometric_crossover(p1, p2, rng):
    """Blend (arithmetic) crossover: a child waypoint lies on the straight line
    between the two parents' waypoints, child = a*p1 + (1-a)*p2.
    Sensors (integers) are inherited from one parent at random."""
    c1, c2 = p1.copy(), p2.copy()
    for d in range(N_DRONES):
        a = rng.uniform(0.2, 0.8)
        c1.waypoints[d] = a * p1.waypoints[d] + (1 - a) * p2.waypoints[d]
        c2.waypoints[d] = (1 - a) * p1.waypoints[d] + a * p2.waypoints[d]
        c1.speeds[d] = a * p1.speeds[d] + (1 - a) * p2.speeds[d]
        c2.speeds[d] = (1 - a) * p1.speeds[d] + a * p2.speeds[d]
        mask = rng.random(N_WAYPOINTS) < 0.5
        c1.sensors[d] = np.where(mask, p1.sensors[d], p2.sensors[d])
        c2.sensors[d] = np.where(mask, p2.sensors[d], p1.sensors[d])
    return c1, c2


def crossover(p1, p2, rng, rate=0.9, p_geometric=0.5):
    """Apply crossover with probability `rate`, choosing between the two types."""
    if rng.random() > rate:
        return p1.copy(), p2.copy()
    if rng.random() < p_geometric:
        c1, c2 = geometric_crossover(p1, p2, rng)
    else:
        c1, c2 = segment_swap_crossover(p1, p2, rng)
    c1.fitness = c2.fitness = None            # must be re-evaluated
    return c1, c2


# ----------------------------------------------------------------------------
# MUTATION
# ----------------------------------------------------------------------------
def mutate(chrom, env, rng, rate=0.1, sigma=3.0):
    """Mutate a chromosome in place (returns it).

    rate  : probability that each waypoint is mutated
    sigma : std-dev (in cells) of the Gaussian spatial shift
    """
    for d in range(N_DRONES):
        for w in range(N_WAYPOINTS):
            if rng.random() < rate:
                r = rng.random()
                if r < 0.70:                       # local move: Gaussian shift
                    x, y = chrom.waypoints[d, w] + rng.normal(0, sigma, size=2)
                elif r < 0.85:                     # global jump: restart waypoint
                    if rng.random() < 0.5:         # ... somewhere on the map
                        x, y = rng.uniform(0, env.size - 1, size=2)
                    else:                          # ... or inside a disaster zone
                        cells = np.argwhere(env.priority > 0.2)
                        y, x = cells[rng.integers(len(cells))]
                else:                              # leave position unchanged
                    x, y = chrom.waypoints[d, w]
                chrom.waypoints[d, w] = env.nearest_free_cell(x, y)   # repair

            if rng.random() < rate:                # speed gene
                chrom.speeds[d, w] = np.clip(
                    chrom.speeds[d, w] + rng.normal(0, 0.5), SPEED_MIN, SPEED_MAX)

            if rng.random() < rate:                # sensor gene (integer)
                chrom.sensors[d, w] = rng.integers(0, 3)

        if rng.random() < rate:                    # order mutation: swap 2 waypoints
            i, j = rng.choice(N_WAYPOINTS, size=2, replace=False)
            for arr in (chrom.waypoints, chrom.speeds, chrom.sensors):
                arr[d, [i, j]] = arr[d, [j, i]]
    chrom.fitness = None
    return chrom


# ----------------------------------------------------------------------------
# DIVERSITY (for the report: shows whether premature convergence happens)
# ----------------------------------------------------------------------------
def population_diversity(population):
    """Mean distance of every individual's waypoints from the population's
    average waypoints. Near 0 means the population has collapsed."""
    W = np.array([c.waypoints for c in population])    # (pop, D, W, 2)
    return float(np.linalg.norm(W - W.mean(axis=0), axis=-1).mean())
