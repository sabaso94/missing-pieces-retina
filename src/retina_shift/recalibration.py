"""Repairing the model at a new clinic with a few labelled images, without retraining it.

Two repairs, alone and together, against doing nothing and against a repair that needs no labels:
- platt: logistic recalibration of the probabilities (intercept and slope on the logit), so that the
  probabilities match the clinic's referral rates; the threshold stays where it was set on EyePACS;
- threshold: a new threshold, chosen on the labelled images to give 90% sensitivity again;
- platt_threshold: both (referrals are those of `threshold`; calibration is Platt's);
- em_prior: EM re-estimation of the prevalence from the clinic's unlabeled images, probabilities adjusted
  to it (no labels at all). A prior shift is a monotone change of the probabilities, so at a matching
  threshold it leaves referrals unchanged: it is scored on calibration only (sensitivity and specificity NaN).
Every repair is scored on the clinic images that were not used to fit it. Platt scaling with a slope that is
not positive (an inverted ranking, possible in small samples) falls back to the unrepaired model.
"""
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from retina_shift.labelfree import prevalence_em
from retina_shift.metrics import ece, sensitivity, specificity, threshold_for_sensitivity

METHODS = ["none", "platt", "threshold", "platt_threshold", "em_prior"]
TARGET_SENSITIVITY = 0.9


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_platt(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """(intercept, slope) of a logistic regression of y on logit(p); the identity (0, 1) if y has one class.

    The slope has a weak L2 penalty (sklearn's default, C = 1; the intercept is not penalised): small labelled
    samples are often perfectly separated, and without a penalty the slope runs off to infinity."""
    if not 0 < y.sum() < len(y):
        return np.array([0.0, 1.0])
    model = LogisticRegression(C=1.0).fit(_logit(p)[:, None], y)
    return np.array([float(model.intercept_[0]), float(model.coef_[0, 0])])


def apply_platt(p: np.ndarray, params: np.ndarray) -> np.ndarray:
    return expit(params[0] + params[1] * _logit(p))


def repair_once(y: np.ndarray, p: np.ndarray, t_source: float, fit_idx: np.ndarray,
                prior_train: float) -> list[dict]:
    """Fit every repair on the labelled images fit_idx; score each on the remaining images."""
    rest = np.ones(len(y), dtype=bool)
    rest[fit_idx] = False
    y_fit, p_fit, y_rest, p_rest = y[fit_idx], p[fit_idx], y[rest], p[rest]
    one_class = not 0 < y_fit.sum() < len(y_fit)

    params = fit_platt(y_fit, p_fit)
    platt_fallback = one_class or params[1] <= 0
    p_platt = p_rest if platt_fallback else apply_platt(p_rest, params)
    t_new = t_source if y_fit.sum() == 0 else threshold_for_sensitivity(y_fit, p_fit, TARGET_SENSITIVITY)
    _, p_em = prevalence_em(p, prior_train)  # unlabeled: uses every image of the clinic

    # method: (scores for referral, threshold, probabilities for calibration, fallback)
    variants = {
        "none": (p_rest, t_source, p_rest, False),
        "platt": (p_platt, t_source, p_platt, platt_fallback),
        "threshold": (p_rest, t_new, p_rest, y_fit.sum() == 0),
        "platt_threshold": (p_rest, t_new, p_platt, platt_fallback),
        "em_prior": (None, None, p_em[rest], False),
    }
    rows = []
    for method, (scores, t, probs, fallback) in variants.items():
        rows.append({"method": method,
                     "sens": np.nan if scores is None else sensitivity(y_rest, scores, t),
                     "spec": np.nan if scores is None else specificity(y_rest, scores, t),
                     "ece": ece(y_rest, probs), "fallback": bool(fallback), "n_scored": int(rest.sum())})
    return rows


def repair_curve(y: np.ndarray, p: np.ndarray, t_source: float, sizes: list[int], prior_train: float,
                 n_reps: int = 200, seed: int = 0) -> pd.DataFrame:
    """repair_once over n_reps random labelled samples of each size (sizes not below the site size are
    skipped). One row per repeat, size and method."""
    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        if size >= len(y):
            continue
        for rep in range(n_reps):
            fit_idx = rng.choice(len(y), size=size, replace=False)
            for row in repair_once(y, p, t_source, fit_idx, prior_train):
                rows.append({"n": size, "rep": rep, **row})
    return pd.DataFrame(rows)
