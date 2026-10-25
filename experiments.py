"""
experiments.py
--------------
Step 7: Hyperparameter sensitivity analysis.

We vary ONE hyperparameter at a time (others stay at their defaults) and give
every run the SAME fitness-evaluation budget, so the comparison is fair.
Each setting is repeated over several random seeds (mean +/- std).
"""

import numpy as np
from src.ga import GeneticAlgorithm

BASE = dict(pop_size=60, mutation_rate=0.07, tournament_k=5)
GRID = {
    "Population size": ("pop_size",      [20, 60, 100]),
    "Mutation rate":   ("mutation_rate", [0.02, 0.07, 0.25]),
    "Tournament size": ("tournament_k",  [2, 5, 10]),
}


def run_sensitivity(env, evaluator, budget=3000, n_seeds=3, elite=2, verbose=True):
    cache, results = {}, {}
    total = sum(len(v) for _, v in GRID.values())
    done = 0
    for label, (param, values) in GRID.items():
        for val in values:
            cfg = dict(BASE); cfg[param] = val
            key = tuple(sorted(cfg.items()))
            if key not in cache:                       # base setting is re-used, not re-run
                pop = cfg["pop_size"]
                gens = max(1, (budget - pop) // (pop - elite))
                runs = []
                for s in range(1, n_seeds + 1):
                    ga = GeneticAlgorithm(env, evaluator, generations=gens, patience=10**9,
                                          elite=elite, seed=s, verbose=False, **cfg)
                    ga.run()
                    runs.append(dict(best=ga.best.fitness,
                                     curve=ga.history["best"],
                                     diversity=ga.history["diversity"][-1],
                                     evals=ga.n_evaluations, pop=pop, elite=elite))
                cache[key] = runs
            results[(label, val)] = cache[key]
            done += 1
            if verbose:
                b = [r["best"] for r in cache[key]]
                print(f"[{done:2d}/{total}] {label:16s} = {val:<6} -> best fitness "
                      f"{np.mean(b):7.3f} +/- {np.std(b):.3f}", flush=True)
    return results


def summary_table(results):
    """Print a clean table (copy it into the report)."""
    print(f"\n{'Parameter':18s}{'Value':>8s}{'Mean best':>12s}{'Std':>9s}{'Final diversity':>17s}")
    print("-" * 64)
    last = None
    for (label, val), runs in results.items():
        b = np.array([r["best"] for r in runs]); dv = np.mean([r["diversity"] for r in runs])
        print(f"{label if label != last else '':18s}{val:>8}{b.mean():12.3f}{b.std():9.3f}{dv:17.2f}")
        last = label
