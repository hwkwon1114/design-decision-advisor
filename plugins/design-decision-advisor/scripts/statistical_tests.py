"""
Statistical comparison utilities for empirical design experiments.

Covers: pairwise tests, multi-group tests, effect sizes,
Sobol sensitivity analysis, and multiple comparison correction.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple


# ── Pairwise comparison ───────────────────────────────────────────────────────

def compare_two(
    a: np.ndarray,
    b: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
    alternative: str = "two-sided",
) -> dict:
    """
    Full pairwise comparison between two samples.
    Uses Mann-Whitney U (non-parametric) + effect size.

    Args:
        a, b: 1D arrays of scalar results (e.g., final objective values across seeds)
        label_a, label_b: names for the report
        alternative: "two-sided", "less", or "greater"
    Returns:
        dict with test results, effect size, and summary string
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)

    # Mann-Whitney U
    stat, p_value = stats.mannwhitneyu(a, b, alternative=alternative)

    # Rank-biserial correlation as effect size: r = 1 - 2U/(n_a*n_b)
    r = 1.0 - (2 * stat) / (n_a * n_b)

    # Cohen's d
    pooled_std = np.sqrt(((n_a - 1) * a.std(ddof=1)**2 + (n_b - 1) * b.std(ddof=1)**2)
                         / (n_a + n_b - 2))
    cohens_d = (a.mean() - b.mean()) / (pooled_std + 1e-12)

    effect_label = _effect_label(abs(r))
    winner = label_a if a.mean() < b.mean() else label_b  # assumes minimization

    summary = (
        f"{label_a}: {a.mean():.4f} ± {a.std():.4f}  |  "
        f"{label_b}: {b.mean():.4f} ± {b.std():.4f}  |  "
        f"p={p_value:.4f}  |  r={r:.3f} ({effect_label} effect)"
    )

    return {
        "label_a": label_a,
        "label_b": label_b,
        "mean_a": float(a.mean()),
        "std_a": float(a.std()),
        "mean_b": float(b.mean()),
        "std_b": float(b.std()),
        "median_a": float(np.median(a)),
        "median_b": float(np.median(b)),
        "n_a": n_a,
        "n_b": n_b,
        "u_statistic": float(stat),
        "p_value": float(p_value),
        "rank_biserial_r": float(r),
        "cohens_d": float(cohens_d),
        "effect_size_label": effect_label,
        "significant": p_value < 0.05,
        "winner": winner,
        "summary": summary,
    }


def _effect_label(r: float) -> str:
    if r < 0.1:
        return "negligible"
    elif r < 0.3:
        return "small"
    elif r < 0.5:
        return "medium"
    else:
        return "large"


# ── Multi-group comparison ────────────────────────────────────────────────────

def compare_many(
    groups: Dict[str, np.ndarray],
    correction: str = "bonferroni",
) -> dict:
    """
    Kruskal-Wallis test + post-hoc Dunn test for 3+ groups.

    Args:
        groups: dict mapping label -> 1D array of values
        correction: p-value correction method ("bonferroni" or "holm")
    Returns:
        dict with overall test result and pairwise post-hoc results
    """
    labels = list(groups.keys())
    arrays = [np.asarray(groups[k], dtype=float) for k in labels]

    kw_stat, kw_p = stats.kruskal(*arrays)

    # Post-hoc pairwise with correction
    pairs = []
    raw_p = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            _, p = stats.mannwhitneyu(arrays[i], arrays[j], alternative="two-sided")
            pairs.append((labels[i], labels[j]))
            raw_p.append(p)

    corrected_p = _bonferroni(raw_p) if correction == "bonferroni" else _holm(raw_p)

    posthoc = []
    for (la, lb), p_raw, p_corr in zip(pairs, raw_p, corrected_p):
        posthoc.append({
            "pair": f"{la} vs {lb}",
            "p_raw": float(p_raw),
            "p_corrected": float(p_corr),
            "significant": p_corr < 0.05,
        })

    return {
        "kruskal_wallis_stat": float(kw_stat),
        "kruskal_wallis_p": float(kw_p),
        "overall_significant": kw_p < 0.05,
        "correction": correction,
        "posthoc": posthoc,
    }


def _bonferroni(p_values: list) -> list:
    n = len(p_values)
    return [min(p * n, 1.0) for p in p_values]


def _holm(p_values: list) -> list:
    n = len(p_values)
    order = np.argsort(p_values)
    corrected = np.array(p_values, dtype=float)
    for rank, idx in enumerate(order):
        corrected[idx] = min(p_values[idx] * (n - rank), 1.0)
    return corrected.tolist()


# ── Sobol sensitivity analysis ────────────────────────────────────────────────

def sobol_sensitivity(
    func,
    bounds: np.ndarray,
    n: int = 1024,
    feature_names: Optional[List[str]] = None,
    seed: int = 0,
) -> dict:
    """
    First-order and total-order Sobol sensitivity indices.
    Tells you which input dimensions drive the output most.

    Args:
        func: callable (N, D) -> (N,) — the function to analyze
        bounds: (D, 2) array of [lo, hi] for each dimension
        n: base sample count (total evaluations ≈ n * (D + 2))
        feature_names: optional list of D names
        seed: random seed
    Returns:
        dict with S1 (first-order), ST (total-order) per feature
    """
    from scipy.stats.qmc import Sobol

    D = bounds.shape[0]
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(D)]

    # Saltelli sampling: generate A, B, and AB matrices
    sampler = Sobol(d=2 * D, scramble=True, seed=seed)
    raw = sampler.random(n)  # (n, 2D)
    A = raw[:, :D]
    B = raw[:, D:]

    # Scale to bounds
    lo, hi = bounds[:, 0], bounds[:, 1]
    A_scaled = A * (hi - lo) + lo
    B_scaled = B * (hi - lo) + lo

    f_A = func(A_scaled)
    f_B = func(B_scaled)

    S1_list, ST_list = [], []
    for i in range(D):
        AB_i = A_scaled.copy()
        AB_i[:, i] = B_scaled[:, i]
        f_AB_i = func(AB_i)

        var_total = np.var(np.concatenate([f_A, f_B]))
        if var_total < 1e-12:
            S1_list.append(0.0)
            ST_list.append(0.0)
            continue

        # Jansen estimator
        S1 = 1.0 - np.mean((f_B - f_AB_i) ** 2) / (2 * var_total)
        ST = np.mean((f_A - f_AB_i) ** 2) / (2 * var_total)
        S1_list.append(float(np.clip(S1, 0, 1)))
        ST_list.append(float(np.clip(ST, 0, 1)))

    return {
        "feature_names": feature_names,
        "S1": S1_list,
        "ST": ST_list,
        "n_evaluations": n * (D + 2),
    }


# ── Cross-validation helper ───────────────────────────────────────────────────

def kfold_indices(N: int, k: int = 5, seed: int = 0):
    """
    Yield (train_indices, val_indices) for k-fold CV.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    fold_size = N // k
    for fold in range(k):
        val = idx[fold * fold_size: (fold + 1) * fold_size]
        train = np.concatenate([idx[:fold * fold_size], idx[(fold + 1) * fold_size:]])
        yield train, val


# ── Printing helpers ──────────────────────────────────────────────────────────

def print_comparison(result: dict) -> None:
    """Pretty-print the output of compare_two()."""
    print("\n" + "=" * 60)
    print(f"  {result['label_a']} vs {result['label_b']}")
    print("=" * 60)
    print(f"  {result['label_a']}: mean={result['mean_a']:.4f}, std={result['std_a']:.4f}, n={result['n_a']}")
    print(f"  {result['label_b']}: mean={result['mean_b']:.4f}, std={result['std_b']:.4f}, n={result['n_b']}")
    print(f"  Mann-Whitney p={result['p_value']:.4f} | r={result['rank_biserial_r']:.3f} ({result['effect_size_label']} effect)")
    sig = "✓ significant" if result["significant"] else "✗ not significant"
    print(f"  {sig} | winner: {result['winner']}")
    print("=" * 60 + "\n")


def print_sobol(result: dict) -> None:
    """Pretty-print Sobol sensitivity indices."""
    print("\n" + "=" * 60)
    print("  Sobol Sensitivity Analysis")
    print("=" * 60)
    print(f"  {'Feature':<15} {'S1 (1st-order)':>16} {'ST (total)':>12}")
    print("  " + "-" * 45)
    for name, s1, st in zip(result["feature_names"], result["S1"], result["ST"]):
        print(f"  {name:<15} {s1:>16.4f} {st:>12.4f}")
    print(f"\n  Total evaluations: {result['n_evaluations']}")
    print("=" * 60 + "\n")
