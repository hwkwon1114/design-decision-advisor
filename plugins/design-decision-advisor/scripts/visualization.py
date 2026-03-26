"""
Visualization templates for design decision experiments.

All functions return (fig, axes) and save to disk if save_path is provided.
Figures are saved as both .png and .pdf by default.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Consistent style
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.1)
COLORS = sns.color_palette("tab10")


def _save(fig: plt.Figure, path: Optional[str], tight: bool = True) -> None:
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")


# ── Convergence ───────────────────────────────────────────────────────────────

def plot_convergence(
    curves: Dict[str, dict],
    xlabel: str = "Evaluations",
    ylabel: str = "Best objective",
    title: str = "Convergence comparison",
    log_scale: bool = False,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Convergence curves with confidence intervals.

    Args:
        curves: dict mapping label -> stats dict from convergence_analysis.convergence_curve_stats()
                Each dict must have keys: mean, lower, upper (all 1D arrays).
        xlabel, ylabel, title: axis labels
        log_scale: use log y-axis
        save_path: base path to save (extensions .png/.pdf added automatically)

    Example:
        from convergence_analysis import convergence_curve_stats
        curves = {
            "EI": convergence_curve_stats(ei_runs),
            "UCB": convergence_curve_stats(ucb_runs),
        }
        plot_convergence(curves, save_path="results/convergence")
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (label, stats) in enumerate(curves.items()):
        x = np.arange(len(stats["mean"]))
        ax.plot(x, stats["mean"], label=f"{label} (n={stats.get('n_seeds', '?')})",
                color=COLORS[i], linewidth=2)
        ax.fill_between(x, stats["lower"], stats["upper"],
                        color=COLORS[i], alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    _save(fig, save_path)
    return fig, ax


def plot_final_values(
    groups: Dict[str, np.ndarray],
    ylabel: str = "Final objective value",
    title: str = "Final solution quality",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Violin + strip plot of final values across seeds.

    Args:
        groups: dict mapping label -> (K,) array of final values per seed
    """
    fig, ax = plt.subplots(figsize=(max(4, len(groups) * 1.5), 4))
    labels = list(groups.keys())
    data = [np.asarray(groups[k]) for k in labels]

    parts = ax.violinplot(data, positions=range(len(labels)), showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(COLORS[i])
        pc.set_alpha(0.6)

    for i, (d, pos) in enumerate(zip(data, range(len(labels)))):
        ax.scatter(np.full_like(d, pos) + np.random.uniform(-0.05, 0.05, len(d)),
                   d, color=COLORS[i], s=20, alpha=0.8, zorder=3)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _save(fig, save_path)
    return fig, ax


# ── Design space coverage ─────────────────────────────────────────────────────

def plot_design_space_pca(
    samples: Dict[str, np.ndarray],
    objectives: Optional[Dict[str, np.ndarray]] = None,
    title: str = "Design space coverage (PCA)",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    PCA projection of design samples colored by method (and optionally objective value).

    Args:
        samples: dict mapping label -> (N, D) array of design vectors
        objectives: optional dict mapping label -> (N,) objective values for coloring
        title: plot title
    """
    from sklearn.decomposition import PCA

    all_samples = np.vstack(list(samples.values()))
    pca = PCA(n_components=2)
    all_proj = pca.fit_transform(all_samples)
    ev = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, len(samples), figsize=(5 * len(samples), 4), squeeze=False)

    offset = 0
    for i, (label, X) in enumerate(samples.items()):
        proj = all_proj[offset: offset + len(X)]
        offset += len(X)
        ax = axes[0, i]

        if objectives is not None and label in objectives:
            sc = ax.scatter(proj[:, 0], proj[:, 1], c=objectives[label],
                            cmap="viridis", s=15, alpha=0.7)
            plt.colorbar(sc, ax=ax, label="Objective")
        else:
            ax.scatter(proj[:, 0], proj[:, 1], color=COLORS[i], s=15, alpha=0.7)

        ax.set_title(label)
        ax.set_xlabel(f"PC1 ({ev[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({ev[1]*100:.1f}%)")

    fig.suptitle(title, fontsize=13)
    _save(fig, save_path)
    return fig, axes


def plot_pairwise_distance_histogram(
    samples: Dict[str, np.ndarray],
    title: str = "Pairwise distance distribution",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Histogram of pairwise distances for each representation.
    Wider/further-right distribution = more diverse samples.
    """
    from scipy.spatial.distance import pdist

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (label, X) in enumerate(samples.items()):
        dists = pdist(X)
        ax.hist(dists, bins=50, alpha=0.5, color=COLORS[i], label=label, density=True)
        ax.axvline(np.mean(dists), color=COLORS[i], linestyle="--", linewidth=1.5,
                   label=f"{label} mean={np.mean(dists):.2f}")

    ax.set_xlabel("Pairwise distance")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)
    return fig, ax


# ── Temporal / trajectory ─────────────────────────────────────────────────────

def plot_trajectory_bundle(
    trajectories: Dict[str, List[np.ndarray]],
    dim_x: int = 0,
    dim_y: int = 1,
    title: str = "Trajectory bundle",
    max_shown: int = 50,
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot trajectory bundles for temporal design representations.

    Args:
        trajectories: dict mapping label -> list of (T, D) arrays
        dim_x, dim_y: which dimensions to plot (for D > 2)
        max_shown: cap number of trajectories per group (for readability)
    """
    fig, axes = plt.subplots(1, len(trajectories),
                              figsize=(5 * len(trajectories), 4), squeeze=False)
    for i, (label, traj_list) in enumerate(trajectories.items()):
        ax = axes[0, i]
        shown = traj_list[:max_shown]
        for t in shown:
            t = np.asarray(t)
            ax.plot(t[:, dim_x], t[:, dim_y], color=COLORS[i], alpha=0.3, linewidth=0.8)
            ax.scatter(t[0, dim_x], t[0, dim_y], color="green", s=20, zorder=3)
            ax.scatter(t[-1, dim_x], t[-1, dim_y], color="red", s=20, zorder=3)
        ax.set_title(f"{label} (n={len(traj_list)})")
        ax.set_xlabel(f"dim {dim_x}")
        ax.set_ylabel(f"dim {dim_y}")

    handles = [mpatches.Patch(color="green", label="Start"),
               mpatches.Patch(color="red", label="End")]
    fig.legend(handles=handles, loc="lower right")
    fig.suptitle(title, fontsize=13)
    _save(fig, save_path)
    return fig, axes


# ── Statistical / sensitivity ─────────────────────────────────────────────────

def plot_sobol_sensitivity(
    result: dict,
    title: str = "Sobol sensitivity indices",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Grouped bar chart of 1st-order and total-order Sobol indices.

    Args:
        result: output of statistical_tests.sobol_sensitivity()
    """
    names = result["feature_names"]
    S1 = np.array(result["S1"])
    ST = np.array(result["ST"])
    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), 4))
    ax.bar(x - width / 2, S1, width, label="S1 (1st-order)", color=COLORS[0], alpha=0.8)
    ax.bar(x + width / 2, ST, width, label="ST (total)", color=COLORS[1], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Sensitivity index")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)
    return fig, ax


def plot_gp_calibration(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    observed: np.ndarray,
    label: str = "",
    title: str = "GP calibration",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Predicted vs actual plot with ±2σ error bars.
    Well-calibrated GP: points lie along y=x within the error bars.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.errorbar(observed, pred_mean, yerr=2 * pred_std,
                fmt="o", alpha=0.5, color=COLORS[0], markersize=4, label=label)
    lo = min(observed.min(), pred_mean.min())
    hi = max(observed.max(), pred_mean.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted (±2σ)")
    ax.set_title(title)
    ax.legend()
    _save(fig, save_path)
    return fig, ax


def plot_coverage_summary(
    metrics_dict: dict,
    title: str = "Coverage metric comparison",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Bar chart comparing coverage metrics across representations.

    Args:
        metrics_dict: output of coverage_metrics.compute_spatial_metrics()
                      {label: {metric_name: value}}
    """
    labels = list(metrics_dict.keys())
    metric_names = list(next(iter(metrics_dict.values())).keys())
    n_metrics = len(metric_names)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4), squeeze=False)
    for j, metric in enumerate(metric_names):
        ax = axes[0, j]
        vals = [metrics_dict[lbl][metric] for lbl in labels]
        bars = ax.bar(labels, vals, color=COLORS[:len(labels)], alpha=0.8)
        ax.bar_label(bars, fmt="%.3f", padding=2)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylabel("Value")

    fig.suptitle(title, fontsize=13)
    _save(fig, save_path)
    return fig, axes
