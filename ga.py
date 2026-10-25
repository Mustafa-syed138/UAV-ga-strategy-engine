"""
ga.py
-----
Step 5: The complete Genetic Algorithm (object-oriented).

Loop:
    initialise population -> evaluate -> [ select parents -> crossover ->
    mutate -> evaluate -> elitism ] x generations -> stop on termination rule.

Premature-convergence protection:
    * tournament selection with a small tournament size
    * several mutation types, including global waypoint resets
    * ADAPTIVE mutation: when the best fitness stagnates, the mutation rate and
      step size are temporarily boosted to push the population out of a local optimum.
"""

import numpy as np
from src.chromosome import Chromosome
from src.operators import (tournament_selection, crossover, mutate,
                           population_diversity)


class GeneticAlgorithm:
    def __init__(self, env, evaluator, pop_size=60, generations=200,
                 tournament_k=5, crossover_rate=0.9, mutation_rate=0.07,
                 sigma=3.0, elite=2, patience=80, stagnation_boost=15,
                 seed=0, verbose=True):
        self.env, self.evaluator = env, evaluator
        self.pop_size, self.generations = pop_size, generations
        self.k, self.cx_rate = tournament_k, crossover_rate
        self.mut_rate, self.sigma = mutation_rate, sigma
        self.elite, self.patience = elite, patience
        self.stagnation_boost = stagnation_boost
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

        self.population = []
        self.history = {"best": [], "avg": [], "diversity": [], "mut_rate": []}
        self.snapshots = {}            # generation -> copy of best chromosome
        self.n_evaluations = 0
        self.best = None

    # ------------------------------------------------------------------
    def _evaluate(self, chrom):
        self.evaluator.evaluate(chrom)
        self.n_evaluations += 1

    def initialise(self):
        self.population = [Chromosome.random(self.env, self.rng)
                           for _ in range(self.pop_size)]
        for c in self.population:
            self._evaluate(c)

    def _record(self, gen, mut_rate):
        fits = np.array([c.fitness for c in self.population])
        best = max(self.population, key=lambda c: c.fitness)
        self.history["best"].append(float(fits.max()))
        self.history["avg"].append(float(fits.mean()))
        self.history["diversity"].append(population_diversity(self.population))
        self.history["mut_rate"].append(mut_rate)
        if self.best is None or best.fitness > self.best.fitness:
            self.best = best.copy()
        return best

    # ------------------------------------------------------------------
    def run(self):
        self.initialise()
        best = self._record(0, self.mut_rate)
        self.snapshots[0] = best.copy()
        last_improve, best_so_far = 0, self.best.fitness

        for gen in range(1, self.generations + 1):
            # ---- adaptive mutation: boost when stagnating
            stagnant = gen - last_improve
            boost = stagnant >= self.stagnation_boost
            rate = min(0.4, self.mut_rate * 2.5) if boost else self.mut_rate
            sigma = self.sigma * 1.5 if boost else self.sigma

            # ---- elitism: carry the best individuals over unchanged
            self.population.sort(key=lambda c: -c.fitness)
            new_pop = [c.copy() for c in self.population[:self.elite]]

            # ---- breed the rest
            while len(new_pop) < self.pop_size:
                p1 = tournament_selection(self.population, self.k, self.rng)
                p2 = tournament_selection(self.population, self.k, self.rng)
                for child in crossover(p1, p2, self.rng, rate=self.cx_rate):
                    mutate(child, self.env, self.rng, rate=rate, sigma=sigma)
                    if child.fitness is None:
                        self._evaluate(child)
                    new_pop.append(child)
            self.population = new_pop[:self.pop_size]

            best = self._record(gen, rate)
            if self.best.fitness > best_so_far + 1e-6:
                best_so_far, last_improve = self.best.fitness, gen

            # snapshots of the best plan for the "evolution" plots
            if gen in (self.generations // 4, self.generations // 2,
                       3 * self.generations // 4, self.generations):
                self.snapshots[gen] = self.best.copy()

            if self.verbose and (gen % 25 == 0 or gen == 1):
                print(f"Gen {gen:4d} | best {self.history['best'][-1]:8.3f} | "
                      f"avg {self.history['avg'][-1]:8.3f} | "
                      f"diversity {self.history['diversity'][-1]:5.2f}"
                      f"{'  [mutation boost]' if boost else ''}")

            # ---- termination: no improvement for `patience` generations
            if gen - last_improve >= self.patience:
                if self.verbose:
                    print(f"Stopped early at generation {gen} (no improvement for {self.patience}).")
                self.snapshots[gen] = self.best.copy()
                break
        self.snapshots["final"] = self.best.copy()
        return self.best
