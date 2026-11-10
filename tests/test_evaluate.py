import numpy as np

from retina_shift.evaluate import evaluate_site, reliability_bins

KEYS = ["auroc", "sens", "spec", "ece", "citl", "slope", "brier"]


def test_evaluate_site_reports_every_metric_with_a_ci():
    rng = np.random.default_rng(0)
    p = rng.random(300)
    y = (rng.random(300) < p).astype(int)
    row = evaluate_site(y, p, np.arange(300), t=0.5, n_boot=50)
    assert row["n"] == 300 and row["prevalence"] == y.mean()
    for key in KEYS:
        assert row[f"{key}_lo"] <= row[key] <= row[f"{key}_hi"], key


def test_reliability_bins_cover_every_image_once():
    rng = np.random.default_rng(1)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(int)
    bins = reliability_bins(y, p, n_bins=10)
    assert bins["n"].sum() == 500
    assert ((bins["mean_p"] >= 0) & (bins["mean_p"] <= 1)).all()
