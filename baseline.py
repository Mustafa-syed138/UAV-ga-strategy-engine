"""
baseline.py
-----------
Step 6: Baseline for comparison = pure RANDOM SEARCH.

It uses exactly the same chromosome generator, repair and fitness function as
the GA, and the SAME number of fitness evaluations, so the comparison is fair.
The only difference: no selection, crossover or mutation (no learning).
"""

import numpy as np
from src.chromosome import Chromosome


def random_search(env, evaluator, n_evals, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    best, best_curve = None, []
    for i in range(n_evals):
        c = Chromosome.random(env, rng)
        evaluator.evaluate(c)
        if best is None or c.fitness > best.fitness:
            best = c
        best_curve.append(best.fitness)
        if verbose and (i + 1) % 2000 == 0:
            print(f"Random search: {i + 1:6d} evaluations | best so far {best.fitness:.3f}")
    return best, best_curve
