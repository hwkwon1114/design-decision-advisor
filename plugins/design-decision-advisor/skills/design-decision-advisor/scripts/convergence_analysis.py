"""
Convergence and optimization quality metrics.

Works with results from BoTorch BO loops or any optimizer that returns
a sequence of objective values per seed.
"""

import numpy as np
from typing import List, Optional


# ── Convergence curves ────────────────────────────────────────────────────────

def best_so_far(values: np.ndarray, minimize: bool = True) -> np.ndarray:
    """
    Cumulative best value found. Converts a raw sequence of observations
    into a non-increasing (or non-decreasing) convergence trace.

    Args:
        values: (T,) array of objective values in order of evaluation
        minimize: True for minimization, False for maximization
    Returns:
        (T,) array of best-so-far values
    """
    values = np.asarray(values, dtype=float)
    if minimize:
        return np.minimum.accumulate(values)
    else:
        return np.maximum.accumulate(values)


def convergence_curve_stats(
    curves: List[np.ndarray],
    minimize: bool = True,
    ci: float = 0.95,
) -> dict:
    """
    Aggregate convergence curves across seeds into mean ± CI.

    Args:
        curves: list of (T,) arrays — one per seed (must be same length)
        minimize: True for minimization
        ci: confidence interval level (default 0.95)
    Returns:
        dict with keys: mean, std, lower, upper, median (all (T,) arrays)
    """
    bsf = np.stack([best_so_far(c, minimize=minimize) for c in curves])  # (K, T)
    K = bsf.shape[0]
    mean = bsf.mean(axis=0)
    std = bsf.std(axis=0)
    # t-based CI for small K
    from scipy import stats
    t_crit = stats.t.ppf((1 + ci) / 2, df=K - 1) if K > 1 else 1.96
    se = std / np.sqrt(K)
    return {
        "mean": mean,
        "std": std,
        "median": np.median(bsf, axis=0),
        "lower": mean - t_crit * se,
        "upper": mean + t_crit * se,
        "n_seeds": K,
    }


def final_values(curves: List[np.ndarray], minimize: bool = True) -> np.ndarray:
    """
    Best value found at the end of each seed's run.

    Returns:
        (K,) array — one scalar per seed
    """
    return np.array([best_so_far(c, minimize=minimize)[-1] for c in curves])


def area_under_curve(curves: List[np.ndarray], minimize: bool = True) -> np.ndarray:
    """
    Area under convergence curve for each seed. Lower = faster convergence (minimization).

    Returns:
        (K,) array
    """
    return np.array([np.trapz(best_so_far(c, minimize=minimize)) for c in curves])


# ── Regret ────────────────────────────────────────────────────────────────────

def simple_regret(curves: List[np.ndarray], optimum: float, minimize: bool = True) -> dict:
    """
    Simple regret: |best_found - optimum| at each step, averaged across seeds.

    Args:
        curves: list of (T,) arrays
        optimum: known optimal value
        minimize: True for minimization
    Returns:
        dict with mean, std, lower, upper (all (T,) arrays)
    """
    regret_curves = []
    for c in curves:
        bsf = best_so_far(c, minimize=minimize)
        regret_curves.append(np.abs(bsf - optimum))
    return convergence_curve_stats(regret_curves, minimize=True)  # regret always minimized


# ── Hypervolume (multi-objective) ─────────────────────────────────────────────

def hypervolume(pareto_front: np.ndarray, reference_point: np.ndarray) -> float:
    """
    Hypervolume indicator for multi-objective optimization.
    Requires pygmo or pymoo; falls back to Monte Carlo approximation.

    Args:
        pareto_front: (N, M) array of non-dominated objective vectors
        reference_point: (M,) reference point (should be dominated by all solutions)
    Returns:
        hypervolume value
    """
    try:
        import pygmo as pg
        hv = pg.hypervolume(pareto_front)
        return float(hv.compute(reference_point))
    except ImportError:
        pass

    try:
        from pymoo.indicators.hv import HV
        ind = HV(ref_point=reference_point)
        return float(ind(pareto_front))
    except ImportError:
        pass

    # Monte Carlo fallback (2D only)
    if pareto_front.shape[1] == 2:
        lo = pareto_front.min(axis=0)
        hi = reference_point
        n_mc = 100_000
        rng = np.random.default_rng(0)
        pts = rng.uniform(lo, hi, size=(n_mc, 2))
        dominated = np.all(
            pts[:, None, :] >= pareto_front[None, :, :], axis=-1
        ).any(axis=-1)
        vol = np.prod(hi - lo)
        return float(np.mean(~dominated) * vol)

    raise RuntimeError(
        "Install pygmo or pymoo for hypervolume computation in >2 objectives."
    )


def hypervolume_curve(
    all_objectives: List[np.ndarray],
    reference_point: np.ndarray,
) -> np.ndarray:
    """
    Hypervolume at each step of optimization for a single run.

    Args:
        all_objectives: list of (M,) arrays in order of evaluation
        reference_point: (M,) reference point
    Returns:
        (T,) array of hypervolume values
    """
    hvs = []
    current_front = []
    for obj in all_objectives:
        current_front.append(obj)
        front = np.array(current_front)
        # Keep non-dominated points
        is_dominated = np.zeros(len(front), dtype=bool)
        for i in range(len(front)):
            for j in range(len(front)):
                if i != j and np.all(front[j] <= front[i]) and np.any(front[j] < front[i]):
                    is_dominated[i] = True
                    break
        pareto = front[~is_dominated]
        hvs.append(hypervolume(pareto, reference_point))
    return np.array(hvs)


# ── Surrogate quality (BoTorch GP) ────────────────────────────────────────────

def gp_cross_val_metrics(
    train_X: "torch.Tensor",
    train_Y: "torch.Tensor",
    n_folds: int = 5,
    model_cls=None,
    likelihood_cls=None,
) -> dict:
    """
    K-fold cross-validation for a BoTorch SingleTaskGP.

    Args:
        train_X: (N, D) tensor
        train_Y: (N, 1) tensor
        n_folds: number of CV folds
        model_cls: BoTorch model class (default: SingleTaskGP)
        likelihood_cls: likelihood class (default: GaussianLikelihood)
    Returns:
        dict with rmse, mae, nll (mean ± std across folds)
    """
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    if model_cls is None:
        model_cls = SingleTaskGP

    N = train_X.shape[0]
    fold_size = N // n_folds
    indices = torch.randperm(N)

    rmses, maes, nlls = [], [], []
    for k in range(n_folds):
        val_idx = indices[k * fold_size: (k + 1) * fold_size]
        train_idx = torch.cat([indices[:k * fold_size], indices[(k + 1) * fold_size:]])

        X_tr, Y_tr = train_X[train_idx], train_Y[train_idx]
        X_val, Y_val = train_X[val_idx], train_Y[val_idx]

        model = model_cls(X_tr, Y_tr)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        model.eval()

        with torch.no_grad():
            posterior = model.posterior(X_val)
            pred_mean = posterior.mean
            pred_var = posterior.variance

        residuals = (pred_mean - Y_val).squeeze()
        rmses.append(float(residuals.pow(2).mean().sqrt()))
        maes.append(float(residuals.abs().mean()))

        # NLL under predicted Gaussian
        nll_val = 0.5 * (
            (residuals ** 2) / (pred_var.squeeze() + 1e-8)
            + torch.log(pred_var.squeeze() + 1e-8)
            + np.log(2 * np.pi)
        ).mean()
        nlls.append(float(nll_val))

    return {
        "rmse_mean": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "mae_mean": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "nll_mean": float(np.mean(nlls)),
        "nll_std": float(np.std(nlls)),
        "n_folds": n_folds,
    }
