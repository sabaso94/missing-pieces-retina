import numpy as np

from retina_shift.metrics import (
    auroc,
    bootstrap_ci,
    brier,
    calibration_slope,
    citl,
    diff_ci,
    ece,
    sensitivity,
    specificity,
    threshold_for_sensitivity,
)


def _calibrated(n=20_000, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.beta(1, 3, size=n)
    y = (rng.random(n) < p).astype(int)
    return y, p


def test_threshold_is_largest_reaching_target_sensitivity():
    y, p = _calibrated()
    t = threshold_for_sensitivity(y, p, 0.9)
    assert sensitivity(y, p, t) >= 0.9
    above = np.sort(p[y == 1])
    higher = above[above > t]
    assert len(higher) == 0 or sensitivity(y, p, higher[0]) < 0.9


def test_sensitivity_specificity_count_threshold_as_positive():
    y = np.array([1, 1, 0, 0])
    p = np.array([0.5, 0.2, 0.5, 0.1])
    assert sensitivity(y, p, 0.5) == 0.5
    assert specificity(y, p, 0.5) == 0.5


def test_citl_is_zero_when_mean_prediction_equals_prevalence():
    y = np.array([1, 0, 0, 0])
    assert citl(y, np.full(4, 0.25)) == 0.0
    assert citl(y, np.full(4, 0.5)) == 0.25


def test_calibration_slope_is_one_for_calibrated_data():
    y, p = _calibrated()
    assert abs(calibration_slope(y, p) - 1) < 0.1


def test_ece_is_small_for_calibrated_and_large_for_shifted():
    y, p = _calibrated()
    assert ece(y, p) < 0.02
    assert ece(y, np.clip(p + 0.2, 0, 1)) > 0.15


def test_brier_and_auroc():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert brier(y, p) == np.mean((p - y) ** 2)
    assert auroc(y, p) == 0.75
    assert np.isnan(auroc(np.ones(4), p))


def test_clustered_bootstrap_is_wider_than_unclustered():
    y, p = _calibrated(n=2_000)
    y2, p2 = np.repeat(y, 2), np.repeat(p, 2)  # each "patient" has two identical eyes
    _, lo_c, hi_c = bootstrap_ci(auroc, y2, p2, np.repeat(np.arange(2_000), 2), n_boot=300)
    _, lo_u, hi_u = bootstrap_ci(auroc, y2, p2, np.arange(4_000), n_boot=300)
    assert (hi_c - lo_c) > 1.2 * (hi_u - lo_u)


def test_bootstrap_ignores_one_class_resamples():
    y = np.array([1, 0, 0, 0, 0, 0])
    p = np.linspace(0.1, 0.9, 6)
    _point, lo, hi = bootstrap_ci(auroc, y, p, np.arange(6), n_boot=200)
    assert np.isfinite(lo) and np.isfinite(hi)


def test_diff_ci_of_identical_sets_is_centred_on_zero():
    y, p = _calibrated(n=3_000)
    ids = np.arange(3_000)
    diff, lo, hi = diff_ci(auroc, y, p, ids, y, p, ids, n_boot=300)
    assert diff == 0 and lo < 0 < hi
