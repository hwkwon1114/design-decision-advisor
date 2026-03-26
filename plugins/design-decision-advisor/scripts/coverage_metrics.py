"""
Coverage and expressivity metrics for design representations.

Supports both spatial (shape/layout) and temporal (trajectory/time-series) data.
All functions return scalar metrics unless otherwise noted.
"""

import numpy as np
from scipy.stats.qmc import discrepancy
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA


# ── Spatial coverage ─────────────────────────────────────────────────────────

def space_filling_discrepancy(samples: np.ndarray) -> float:
    """
    Centered L2-discrepancy. Lower = more uniformly space-filling.

    Args:
        samples: (N, D) array, values should be in [0, 1]^D (normalize first)
    """
    samples = np.clip(samples, 0.0, 1.0)
    return float(discrepancy(samples, method="CD"))


def mean_pairwise_distance(samples: np.ndarray, metric: str = "euclidean") -> float:
    """
    Average pairwise distance. Higher = more spread / diverse.

    Args:
        samples: (N, D) array
        metric: any scipy.spatial.distance metric
    """
    return float(np.mean(pdist(samples, metric=metric)))


def effective_dimensionality(samples: np.ndarray, variance_threshold: float = 0.95) -> int:
    """
    Number of PCA components explaining `variance_threshold` of variance.
    Higher = richer representation.
    """
    pca = PCA()
    pca.fit(samples)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    return int(np.searchsorted(cumvar, variance_threshold) + 1)


def pca_projection(samples: np.ndarray, n_components: int = 2) -> np.ndarray:
    """Project samples for visualization. Returns (N, n_components)."""
    n_components = min(n_components, samples.shape[0], samples.shape[1])
    return PCA(n_components=n_components).fit_transform(samples)


def convex_hull_volume_ratio(samples: np.ndarray) -> float:
    """
    Convex hull volume / bounding box volume. Higher = more of the space is reachable.
    Falls back to range ratio for D > 3.
    """
    from scipy.spatial import ConvexHull
    n, d = samples.shape
    bbox_vol = np.prod(samples.max(axis=0) - samples.min(axis=0) + 1e-12)
    if d <= 3 and n >= d + 1:
        try:
            return float(ConvexHull(samples).volume / bbox_vol)
        except Exception:
            pass
    ranges = samples.max(axis=0) - samples.min(axis=0)
    return float(np.mean(ranges / (ranges + 1e-12)))


# ── Temporal / trajectory coverage ───────────────────────────────────────────

def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Dynamic Time Warping distance between two sequences.

    Args:
        a: (T1,) or (T1, D)
        b: (T2,) or (T2, D)
    """
    a = np.atleast_2d(a) if a.ndim == 1 else a
    b = np.atleast_2d(b) if b.ndim == 1 else b
    T1, T2 = len(a), len(b)
    dtw = np.full((T1 + 1, T2 + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    return float(dtw[T1, T2])


def mean_dtw_diversity(sequences: list) -> float:
    """
    Average pairwise DTW distance. Higher = more temporally diverse.

    Args:
        sequences: list of (T, D) arrays (may have different T)
    """
    N = len(sequences)
    if N < 2:
        return 0.0
    dists = [dtw_distance(sequences[i], sequences[j])
             for i in range(N) for j in range(i + 1, N)]
    return float(np.mean(dists))


def temporal_spread(sequences: list) -> float:
    """Mean pairwise Euclidean distance between sequence centroids. Faster than DTW."""
    centroids = np.stack([np.asarray(s).mean(axis=0) for s in sequences])
    return mean_pairwise_distance(centroids)


# ── Gradient quality ──────────────────────────────────────────────────────────

def gradient_norm_stats(grads: np.ndarray) -> dict:
    """
    Summary stats of gradient norms across sampled points.
    Low CV (coefficient of variation) = smooth, well-behaved gradients.

    Args:
        grads: (N, D) array of gradients at N sample points
    """
    norms = np.linalg.norm(grads, axis=-1)
    return {
        "mean": float(np.mean(norms)),
        "std": float(np.std(norms)),
        "median": float(np.median(norms)),
        "cv": float(np.std(norms) / (np.mean(norms) + 1e-12)),
    }


def jacobian_condition_numbers(jacobians: np.ndarray) -> np.ndarray:
    """
    Condition numbers of Jacobians at sampled points. Lower = better landscape.

    Args:
        jacobians: (N, M, D) — N points, M outputs, D inputs
    Returns:
        (N,) array
    """
    return np.array([np.linalg.cond(J) if not np.any(np.isnan(J)) else np.nan
                     for J in jacobians])


# ── Convenience wrapper ───────────────────────────────────────────────────────

def compute_spatial_metrics(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
) -> dict:
    """
    Compute all spatial coverage metrics for two representations side-by-side.

    Args:
        samples_a, samples_b: (N, D) arrays of designs from each representation
    Returns:
        nested dict {label: {metric: value}}
    """
    combined = np.vstack([samples_a, samples_b])
    lo, hi = combined.min(axis=0), combined.max(axis=0)
    scale = hi - lo + 1e-12
    a_norm = (samples_a - lo) / scale
    b_norm = (samples_b - lo) / scale

    result = {}
    for label, raw, norm in [(label_a, samples_a, a_norm), (label_b, samples_b, b_norm)]:
        result[label] = {
            "discrepancy": space_filling_discrepancy(norm),
            "mean_pairwise_dist": mean_pairwise_distance(raw),
            "effective_dim": effective_dimensionality(raw),
            "hull_volume_ratio": convex_hull_volume_ratio(raw),
        }
    return result
