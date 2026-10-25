# UAV Fleet Strategy Engine using a Genetic Algorithm

**Course:** Artificial Intelligence (BSCS 5) - Assignment 2 (CLO 4)
**University:** Bahria University, Lahore Campus
**Team:** _Name 1 (Roll No), Name 2 (Roll No), Name 3 (Roll No)_

A Genetic Algorithm (GA) that plans deployment positions, sensor power profiles and
waypoint trajectories for a fleet of 3 UAVs surveying a hazardous disaster zone.
The problem is **multi-objective**: the fleet must cover high-priority areas and detect
targets while minimising energy use, flight time and exposure to hazards, and it must
respect hard constraints (flight-range limit, no-fly zones).

## Project structure

```
uav-ga/
├── main.py               # Runs GA + random-search baseline + all plots
├── run_experiments.py    # Hyperparameter sensitivity study
├── requirements.txt
├── src/
│   ├── environment.py    # 50x50 grid: no-fly zones, hazard risk, disaster priority
│   ├── chromosome.py     # Hybrid chromosome (real-valued + integer genes)
│   ├── fitness.py        # Multi-objective fitness, penalties, repair
│   ├── operators.py      # Tournament selection, crossover, mutation, diversity
│   ├── ga.py             # GeneticAlgorithm class (elitism, adaptive mutation, stopping)
│   ├── baseline.py       # Random-search baseline (same evaluation budget)
│   ├── experiments.py    # Sensitivity analysis helpers
│   └── visualize.py      # All plots
└── results/              # Generated figures and the sensitivity table
```

## Installation

Python 3.9+ is required.

```bash
pip install -r requirements.txt
```

## How to run

```bash
python main.py              # ~1.5 minutes: GA, random search, all figures in results/
python run_experiments.py   # ~5-10 minutes: hyperparameter sensitivity study
```

`main.py` prints the GA progress, a comparison table (GA vs. random search) and saves:

| File | Content |
|---|---|
| `results/environment.png` | The grid: no-fly zones, hazard risk, priority zones |
| `results/convergence.png` | Best vs. average fitness and population diversity |
| `results/best_plan.png` | Final evolved flight paths |
| `results/evolution.png` | Best flight paths at generations 0, 50, 100, 150, 200 |
| `results/heatmaps.png` | Sensor-coverage heatmaps: GA gen 0, GA final, random search |
| `results/ga_vs_random.png` | GA vs. random search, equal evaluation budget |
| `results/sensitivity.png` | Hyperparameter sensitivity (from `run_experiments.py`) |

All runs are reproducible: the map uses `seed=42` and the GA uses `seed=1`.
It also runs in Google Colab (paste the modules into cells, or upload the folder).

## Design summary

**Environment.** A 50x50 grid with three layers: no-fly cells (rectangles), hazard risk
in [0, 1] (circular pockets that fade outward) and disaster priority in [0, 1]. Hazard
and priority zones overlap, which creates the trade-off between coverage and safety.

**Chromosome.** One chromosome is a complete plan for 3 drones. Each drone has 10
waypoints, and each waypoint carries 3 genes: position `(x, y)` (real), speed in
[1, 5] (real) and sensor level 0/1/2 (integer: off/low/high). Every route starts and
ends at the base station.

**Fitness (maximise).**

```
F = 0.35*Coverage + 0.10*Detection - 0.20*Energy - 0.10*Time - 0.25*Hazard - Penalties
```

* *Coverage*: share of total priority value inside sensor range (union over drones).
* *Detection*: share of 40 hidden targets reached by a sensor.
* *Energy*: distance x (base + speed^2 term) + sensor power drain.
* *Time*: finishing time of the slowest drone.
* *Hazard*: risk x time spent in it (slow flight through danger costs more).
* All terms are normalised so the weights are comparable.

**Constraint handling.** Repair (waypoints in no-fly cells are moved to the nearest free
cell; gene ranges are clipped) plus penalties (flight range > 150 cells, and path
segments crossing no-fly cells).

**Genetic operators.**

* Selection: tournament (size `k` = selection pressure), with elitism (top 2 kept).
* Crossover (rate 0.9): segment-swap (whole drone routes or blocks of waypoints) or
  geometric/blend crossover (child waypoints lie between parent waypoints).
* Mutation: Gaussian spatial shift, global waypoint reset (random or in a disaster
  zone), speed tweak, sensor change, waypoint-order swap.
* Premature-convergence protection: adaptive mutation (rate and step size are boosted
  when the best fitness stagnates) and population-diversity monitoring.
* Termination: maximum generations (200) or no improvement for 80 generations.

## Default hyperparameters

| Parameter | Value |
|---|---|
| Population size | 60 |
| Generations | 200 |
| Tournament size | 5 |
| Crossover rate | 0.9 |
| Mutation rate | 0.07 (Gaussian step sigma = 3 cells) |
| Elitism | 2 |

## Key results

| Measure | GA (best plan) | Random search (same budget) |
|---|---|---|
| Fitness | **0.124** | -1.350 |
| Coverage | 0.843 | 0.842 |
| Energy (normalised) | 0.743 | 1.566 |
| Time (normalised) | 0.642 | 1.693 |
| Hazard exposure | 0.175 | 1.202 |
| Range / no-fly penalty | 0 / 0 | 0.79 / 0.15 |

The GA matches the coverage of random search while using less than half the energy and
time and about one-seventh of the hazard exposure. See `results/sensitivity_table.txt`
for the hyperparameter study.
