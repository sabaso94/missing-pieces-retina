"""Discrimination, screening and calibration metrics for one yes/no outcome (referable retinopathy),
with bootstrap CIs that resample patients (or images, where a site has no patient IDs).

All functions take y (N,) with 0/1 labels and p (N,) with predicted probabilities. A case is called
positive when p >= t.
"""

from collections.abc import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve; NaN if only one class is present."""
    if 0 < y.sum() < len(y):
        return float(roc_auc_score(y, p))
    return float("nan")


def sensitivity(y: np.ndarray, p: np.ndarray, t: float) -> float:
    """Share of referable eyes flagged at threshold t."""
    positives = y == 1
    return float((p[positives] >= t).mean()) if positives.any() else float("nan")


def specificity(y: np.ndarray, p: np.ndarray, t: float) -> float:
    """Share of non-referable eyes not flagged at threshold t."""
    negatives = y == 0
    return float((p[negatives] < t).mean()) if negatives.any() else float("nan")


def threshold_for_sensitivity(y: np.ndarray, p: np.ndarray, target: float = 0.9) -> float:
    """The largest threshold whose sensitivity is at least `target`.

    Sorting the positives' probabilities from high to low, the k-th one is the threshold that flags exactly
    k positives; the smallest k reaching the target gives the largest such threshold (most specific).
    """
    scores = np.sort(p[y == 1])[::-1]
    k = int(np.ceil(target * len(scores)))
    return float(scores[k - 1])


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error: in equal-width probability bins, |observed rate - mean prediction|,
    weighted by the share of eyes in the bin."""
    bins = np.digitize(p, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    error = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            error += in_bin.mean() * abs(y[in_bin].mean() - p[in_bin].mean())
    return float(error)


def citl(y: np.ndarray, p: np.ndarray) -> float:
    """Calibration-in-the-large: mean prediction minus observed prevalence (> 0 = over-predicts)."""
    return float(p.mean() - y.mean())


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_slope(y: np.ndarray, p: np.ndarray) -> float:
    """Slope of a logistic regression of y on logit(p). 1 = well spread; < 1 = predictions too extreme."""
    if not 0 < y.sum() < len(y):
        return float("nan")
    model = LogisticRegression(penalty=None).fit(_logit(p)[:, None], y)
    return float(model.coef_[0, 0])


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def _cluster_resample_indices(cluster_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one bootstrap sample: clusters (patients) drawn with replacement, all their rows kept.

    Both eyes of a patient are not independent, so resampling eyes would make the CIs too narrow.
    """
    clusters, inverse = np.unique(cluster_ids, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    starts = np.cumsum(counts) - counts
    drawn = rng.integers(0, len(clusters), size=len(clusters))
    lengths = counts[drawn]
    within = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return order[np.repeat(starts[drawn], lengths) + within]


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    p: np.ndarray,
    cluster_ids: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point estimate, lower, upper) of a 95% percentile bootstrap CI, resampling clusters.
    Resamples where the metric is undefined (NaN, e.g. one class only) are ignored."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        idx = _cluster_resample_indices(cluster_ids, rng)
        samples.append(metric(y[idx], p[idx]))
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return float(metric(y, p)), float(lo), float(hi)


def diff_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_a: np.ndarray,
    p_a: np.ndarray,
    ids_a: np.ndarray,
    y_b: np.ndarray,
    p_b: np.ndarray,
    ids_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """metric(a) - metric(b) for two different sets of eyes (e.g. a site against EyePACS test), with a 95%
    CI from independent cluster resamples of the two sets."""
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        ia = _cluster_resample_indices(ids_a, rng)
        ib = _cluster_resample_indices(ids_b, rng)
        samples.append(metric(y_a[ia], p_a[ia]) - metric(y_b[ib], p_b[ib]))
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return float(metric(y_a, p_a) - metric(y_b, p_b)), float(lo), float(hi)
