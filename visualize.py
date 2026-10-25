"""visualize.py - plotting helpers (more plots are added in later steps)."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_environment(env, save_path=None):
    """Show the three layers of the map side by side + the combined view."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))

    axes[0].imshow(env.no_fly, cmap=ListedColormap(["white", "black"]), origin="lower")
    axes[0].set_title("No-fly zones")

    im1 = axes[1].imshow(env.hazard, cmap="Reds", origin="lower", vmin=0, vmax=1)
    axes[1].set_title("Hazard risk")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(env.priority, cmap="Greens", origin="lower", vmin=0, vmax=1)
    axes[2].set_title("Disaster priority")
    fig.colorbar(im2, ax=axes[2], fraction=0.046)

    # Combined map
    axes[3].imshow(env.priority, cmap="Greens", origin="lower", vmin=0, vmax=1)
    axes[3].imshow(env.hazard, cmap="Reds", origin="lower", alpha=0.45, vmin=0, vmax=1)
    axes[3].imshow(env.no_fly, cmap=ListedColormap([(0, 0, 0, 0), (0, 0, 0, 1)]), origin="lower")
    axes[3].plot(*env.base, "b*", markersize=16, label="Base")
    axes[3].legend(loc="upper right")
    axes[3].set_title("Combined map")

    for ax in axes:
        ax.set_xlabel("x"); ax.set_ylabel("y")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_chromosome(env, chrom, title="UAV flight paths", save_path=None):
    """Draw the routes of all drones on top of the environment."""
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.imshow(env.priority, cmap="Greens", origin="lower", vmin=0, vmax=1)
    ax.imshow(env.hazard, cmap="Reds", origin="lower", alpha=0.45, vmin=0, vmax=1)
    ax.imshow(env.no_fly, cmap=ListedColormap([(0, 0, 0, 0), (0, 0, 0, 1)]), origin="lower")

    colors = ["tab:blue", "tab:orange", "tab:purple", "tab:brown", "tab:cyan"]
    for d in range(chrom.waypoints.shape[0]):
        p = chrom.full_path(env, d)
        ax.plot(p[:, 0], p[:, 1], "-o", color=colors[d % 5], markersize=4,
                linewidth=1.6, label=f"Drone {d}")
    ax.plot(*env.base, "b*", markersize=18)
    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(history, title="GA fitness convergence", save_path=None):
    """Best vs. average fitness per generation (+ population diversity)."""
    gens = range(len(history["best"]))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(gens, history["best"], label="Best fitness", linewidth=2)
    ax[0].plot(gens, history["avg"], label="Average fitness", linewidth=2)
    ax[0].set_xlabel("Generation"); ax[0].set_ylabel("Fitness")
    ax[0].set_title(title); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(gens, history["diversity"], color="tab:green", linewidth=2)
    ax[1].set_xlabel("Generation"); ax[1].set_ylabel("Diversity")
    ax[1].set_title("Population diversity"); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def coverage_map(env, evaluator, chrom):
    """Heatmap data: how many sensor samples reach each cell (0 = never covered)."""
    pts, radii = [], []
    for d in range(chrom.waypoints.shape[0]):
        run = evaluator._sample_drone(chrom, d)
        on = run["radii"] > 0
        pts.append(run["pts"][on]); radii.append(run["radii"][on])
    pts, radii = np.vstack(pts), np.concatenate(radii)
    yy, xx = np.mgrid[:env.size, :env.size]
    cells = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(float)
    if len(pts) == 0:
        return np.zeros((env.size, env.size))
    dist = np.linalg.norm(cells[:, None, :] - pts[None, :, :], axis=2)
    return (dist <= radii[None, :]).sum(axis=1).reshape(env.size, env.size)


def plot_evolution(env, snapshots, evaluator=None, save_path=None):
    """Best flight paths at several generations (shows the plan improving)."""
    from matplotlib.colors import ListedColormap
    gens = sorted(k for k in snapshots if isinstance(k, int))
    if len(gens) > 5:                                   # keep first, last, 3 in between
        gens = [gens[0]] + gens[1:-1][::max(1, (len(gens) - 2) // 3)][:3] + [gens[-1]]
    fig, axes = plt.subplots(1, len(gens), figsize=(4.2 * len(gens), 4.4))
    colors = ["tab:blue", "tab:orange", "tab:purple"]
    for ax, g in zip(np.atleast_1d(axes), gens):
        ch = snapshots[g]
        ax.imshow(env.priority, cmap="Greens", origin="lower", vmin=0, vmax=1)
        ax.imshow(env.hazard, cmap="Reds", origin="lower", alpha=0.45, vmin=0, vmax=1)
        ax.imshow(env.no_fly, cmap=ListedColormap([(0, 0, 0, 0), (0, 0, 0, 1)]), origin="lower")
        for d in range(ch.waypoints.shape[0]):
            p = ch.full_path(env, d)
            ax.plot(p[:, 0], p[:, 1], "-o", color=colors[d % 3], markersize=3, linewidth=1.4)
        ax.plot(*env.base, "b*", markersize=14)
        ax.set_title(f"Generation {g}\nfitness = {ch.fitness:.3f}")
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(env, evaluator, chroms, titles, save_path=None):
    """Coverage heatmaps (sensor passes per cell) for several plans side by side."""
    from matplotlib.colors import ListedColormap
    maps = [coverage_map(env, evaluator, c) for c in chroms]
    vmax = max(m.max() for m in maps) or 1
    fig, axes = plt.subplots(1, len(chroms), figsize=(5.2 * len(chroms), 4.8))
    for ax, m, t, c in zip(np.atleast_1d(axes), maps, titles, chroms):
        im = ax.imshow(m, cmap="YlOrRd", origin="lower", vmin=0, vmax=vmax)
        ax.contour(env.priority > 0.2, levels=[0.5], colors="green", linewidths=1.5)
        ax.imshow(env.no_fly, cmap=ListedColormap([(0, 0, 0, 0), (0, 0, 0, 1)]), origin="lower")
        ax.plot(*env.base, "b*", markersize=14)
        ax.set_title(f"{t}\n(green outline = disaster zones)")
        fig.colorbar(im, ax=ax, fraction=0.046, label="sensor passes")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ga_vs_random(ga, random_curve, save_path=None):
    """Best fitness vs number of fitness evaluations: GA against random search."""
    per_gen = ga.pop_size - ga.elite
    ga_x = [ga.pop_size + g * per_gen for g in range(len(ga.history["best"]))]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(ga_x, ga.history["best"], linewidth=2.2, label="Genetic Algorithm")
    ax.plot(range(1, len(random_curve) + 1), random_curve, linewidth=2.2,
            label="Random search (baseline)")
    ax.set_xlabel("Number of fitness evaluations"); ax.set_ylabel("Best fitness so far")
    ax.set_title("GA vs. random search (equal evaluation budget)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity(results, save_path=None):
    """Top row: final best fitness (mean +/- std). Bottom row: convergence curves."""
    labels = list(dict.fromkeys(k[0] for k in results))
    fig, axes = plt.subplots(2, len(labels), figsize=(5.2 * len(labels), 8))
    for j, label in enumerate(labels):
        items = [(v, r) for (l, v), r in results.items() if l == label]
        vals = [str(v) for v, _ in items]
        means = [np.mean([x["best"] for x in r]) for _, r in items]
        stds = [np.std([x["best"] for x in r]) for _, r in items]
        ax = axes[0, j]
        ax.bar(vals, means, yerr=stds, capsize=6, color=["#9ecae1", "#3182bd", "#9ecae1"])
        ax.set_title(label); ax.set_ylabel("Best fitness (mean +/- std)")
        ax.set_xlabel(label); ax.grid(axis="y", alpha=0.3)
        ax = axes[1, j]
        for v, r in items:
            curve = np.mean([x["curve"] for x in r], axis=0)
            pop, elite = r[0]["pop"], r[0]["elite"]
            xs = [pop + g * (pop - elite) for g in range(len(curve))]
            ax.plot(xs, curve, linewidth=2, label=f"{v}")
        ax.set_xlabel("Fitness evaluations"); ax.set_ylabel("Mean best fitness")
        ax.legend(title=label); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
