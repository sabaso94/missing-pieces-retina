import numpy as np

from retina_shift.metrics import ece, sensitivity
from retina_shift.recalibration import METHODS, apply_platt, fit_platt, repair_curve, repair_once


def _miscalibrated(n=40_000, seed=0, intercept=-1.0, slope=0.5):
    """Labels drawn from sigmoid(intercept + slope * logit(p)): the model is over-confident and too high."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.01, 0.99, size=n)
    logit = np.log(p / (1 - p))
    y = (rng.random(n) < 1 / (1 + np.exp(-(intercept + slope * logit)))).astype(int)
    return y, p


def test_platt_recovers_intercept_and_slope():
    y, p = _miscalibrated()
    a, b = fit_platt(y, p)
    assert abs(a - (-1.0)) < 0.1 and abs(b - 0.5) < 0.1
    assert ece(y, apply_platt(p, np.array([a, b]))) < ece(y, p) / 3


def test_platt_with_one_class_is_the_identity():
    assert np.array_equal(fit_platt(np.zeros(10, dtype=int), np.linspace(0.1, 0.9, 10)), np.array([0.0, 1.0]))


def test_repair_scores_only_images_not_used_for_fitting():
    y, p = _miscalibrated(n=2_000)
    fit_idx = np.arange(100)
    rows = repair_once(y, p, t_source=0.5, fit_idx=fit_idx, prior_train=0.3)
    none = next(r for r in rows if r["method"] == "none")
    assert none["ece"] == ece(y[100:], p[100:])
    assert none["sens"] == sensitivity(y[100:], p[100:], 0.5)
    assert {r["method"] for r in rows} == set(METHODS)


def test_repair_with_one_class_sample_falls_back():
    y, p = _miscalibrated(n=2_000)
    fit_idx = np.flatnonzero(y == 0)[:25]
    rows = repair_once(y, p, t_source=0.5, fit_idx=fit_idx, prior_train=0.3)
    by_method = {r["method"]: r for r in rows}
    assert by_method["platt"]["fallback"] and by_method["threshold"]["fallback"]
    assert by_method["platt"]["ece"] == by_method["none"]["ece"]
    assert not by_method["none"]["fallback"]


def test_repair_curve_skips_sizes_not_smaller_than_the_site():
    y, p = _miscalibrated(n=300)
    curve = repair_curve(y, p, t_source=0.5, sizes=[25, 300], n_reps=3, prior_train=0.3)
    assert set(curve["n"]) == {25}
    assert len(curve) == 3 * len(METHODS)


def test_new_threshold_restores_target_sensitivity_on_average():
    y, p = _miscalibrated(n=6_000)
    curve = repair_curve(y, p, t_source=0.9, sizes=[400], n_reps=30, prior_train=0.3)
    sens = curve[curve["method"] == "threshold"]["sens"].mean()
    assert abs(sens - 0.9) < 0.03
