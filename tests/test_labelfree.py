import numpy as np

from retina_shift.labelfree import (
    atc_accuracy,
    decision_accuracy,
    doc_accuracy,
    expected_sens_spec,
    fit_atc,
    prevalence_acc,
    prevalence_cc,
    prevalence_em,
)
from retina_shift.metrics import sensitivity, specificity

TRAIN_PRIOR = 0.2


def _sample(prior: float, n: int, seed: int):
    """Label shift only: x | y is fixed (N(+1, 1) or N(-1, 1)); p is calibrated for the training prior."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prior).astype(int)
    x = rng.normal(np.where(y == 1, 1.0, -1.0), 1.0)
    p = 1 / (1 + np.exp(-(2 * x + np.log(TRAIN_PRIOR / (1 - TRAIN_PRIOR)))))
    return y, p


def test_em_recovers_a_new_prevalence_and_counting_does_not():
    _y, p = _sample(0.5, 40_000, seed=1)
    prevalence, p_adjusted = prevalence_em(p, TRAIN_PRIOR)
    assert abs(prevalence - 0.5) < 0.03
    assert abs(p_adjusted.mean() - prevalence) < 1e-3
    assert abs(prevalence_cc(p, 0.5) - 0.5) > 0.05


def test_adjusted_count_recovers_the_prevalence():
    y_val, p_val = _sample(TRAIN_PRIOR, 40_000, seed=2)
    t = 0.3
    _y, p = _sample(0.5, 40_000, seed=3)
    estimate = prevalence_acc(p, t, sensitivity(y_val, p_val, t), specificity(y_val, p_val, t))
    assert abs(estimate - 0.5) < 0.03
    assert 0.0 <= prevalence_acc(np.zeros(10), t, 0.9, 0.8) <= 1.0  # clipped


def test_expected_sens_spec_match_the_truth_for_a_calibrated_model():
    y, p = _sample(TRAIN_PRIOR, 50_000, seed=4)
    t = 0.3
    sens, spec = expected_sens_spec(p, t)
    assert abs(sens - sensitivity(y, p, t)) < 0.02
    assert abs(spec - specificity(y, p, t)) < 0.02


def test_atc_and_doc_return_validation_accuracy_on_validation_data():
    y_val, p_val = _sample(TRAIN_PRIOR, 20_000, seed=5)
    t = 0.3
    true = decision_accuracy(y_val, p_val, t)
    assert abs(atc_accuracy(p_val, fit_atc(y_val, p_val, t), t) - true) < 1e-3
    assert abs(doc_accuracy(p_val, y_val, p_val, t) - true) < 1e-12


def test_expected_sens_after_em_keeps_the_original_referral_decisions():
    y, p = _sample(0.5, 50_000, seed=6)  # label shift: p is calibrated for prior 0.2, the clinic has 0.5
    t = 0.3
    _, p_em = prevalence_em(p, TRAIN_PRIOR)
    sens, spec = expected_sens_spec(p_em, t, decision_p=p)
    assert abs(sens - sensitivity(y, p, t)) < 0.02
    assert abs(spec - specificity(y, p, t)) < 0.02
