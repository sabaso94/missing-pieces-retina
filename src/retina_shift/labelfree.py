"""Estimating how a deployed model is doing at a new clinic from its predictions alone, without labels.

`p` are ensemble probabilities of referable retinopathy and `t` is the screening threshold (an eye is
referred when p >= t). Everything here is fitted on EyePACS validation data, where labels exist, and then
applied to a clinic's unlabeled images. The true values are computed only to check the estimates.

Two things change at a new clinic, and the estimators assume different things about them:
- prevalence (label shift): EM prior re-estimation and adjusted counting assume only the share of
  referable eyes changed, while images of a given grade look the same;
- the model's confidence: average-confidence estimates assume the model is still calibrated there.
"""
import numpy as np

from retina_shift.metrics import bootstrap_ci, sensitivity, specificity


def prevalence_cc(p: np.ndarray, t: float) -> float:
    """Classify and count: the share of eyes referred."""
    return float((p >= t).mean())


def prevalence_acc(p: np.ndarray, t: float, sens_val: float, spec_val: float) -> float:
    """Adjusted classify and count: corrects the referral rate for the errors measured on validation data.
    referral rate = prevalence * sens + (1 - prevalence) * (1 - spec), solved for the prevalence."""
    false_positive_rate = 1 - spec_val
    estimate = (prevalence_cc(p, t) - false_positive_rate) / (sens_val - false_positive_rate)
    return float(np.clip(estimate, 0.0, 1.0))


def prevalence_em(p: np.ndarray, prior_train: float, n_iter: int = 100,
                  tol: float = 1e-6) -> tuple[float, np.ndarray]:
    """EM re-estimation of the prevalence (Saerens et al., 2002), and the probabilities adjusted to it.

    Each round rescales every probability by the ratio of the current prevalence estimate to the training
    prevalence (Bayes' rule with a new prior), and the new estimate is the mean adjusted probability.
    """
    prior = prior_train
    adjusted = p
    for _ in range(n_iter):
        pos = p * prior / prior_train
        neg = (1 - p) * (1 - prior) / (1 - prior_train)
        adjusted = pos / (pos + neg)
        new_prior = float(adjusted.mean())
        if abs(new_prior - prior) < tol:
            prior = new_prior
            break
        prior = new_prior
    return prior, adjusted


def expected_sens_spec(p: np.ndarray, t: float, decision_p: np.ndarray | None = None) -> tuple[float, float]:
    """Sensitivity and specificity the model expects of itself if its probabilities are calibrated.

    Each eye is referable with probability p, so the expected number of referable eyes is sum(p), of which
    the referred ones contribute sum(p over referred eyes); the same with 1 - p for specificity.
    `decision_p` are the probabilities the referral decisions were made on, when `p` has been adjusted
    afterwards (e.g. to a new prevalence by EM); by default the decisions come from `p` itself.
    """
    referred = (p if decision_p is None else decision_p) >= t
    return float(p[referred].sum() / p.sum()), float((1 - p)[~referred].sum() / (1 - p).sum())


def _confidence(p: np.ndarray, t: float) -> np.ndarray:
    """The model's probability for the decision it made: p if referred, 1 - p if not."""
    return np.where(p >= t, p, 1 - p)


def decision_accuracy(y: np.ndarray, p: np.ndarray, t: float) -> float:
    """Share of eyes whose referral decision is correct."""
    return float(((p >= t) == (y == 1)).mean())


def fit_atc(y_val: np.ndarray, p_val: np.ndarray, t: float) -> float:
    """Average thresholded confidence (Garg et al., 2022): the confidence cut-off above which the share of
    validation eyes equals the validation accuracy. Only the confidence ranking needs to carry over."""
    return float(np.quantile(_confidence(p_val, t), 1 - decision_accuracy(y_val, p_val, t)))


def atc_accuracy(p: np.ndarray, threshold: float, t: float) -> float:
    return float((_confidence(p, t) > threshold).mean())


def doc_accuracy(p: np.ndarray, y_val: np.ndarray, p_val: np.ndarray, t: float) -> float:
    """Difference of confidences (Guillory et al., 2021): validation accuracy, shifted by the change in
    average confidence."""
    return decision_accuracy(y_val, p_val, t) - (_confidence(p_val, t).mean() - _confidence(p, t).mean())


def estimate_site(y: np.ndarray, p: np.ndarray, ids: np.ndarray, y_val: np.ndarray, p_val: np.ndarray,
                  t: float, prior_train: float, n_boot: int = 1000) -> dict[str, float]:
    """True prevalence, sensitivity, specificity and accuracy (with 95% CIs) next to every estimate.
    `y` is used only for the true values."""
    row = {"n": len(y)}
    truths = {
        "prevalence": lambda yy, pp: float(yy.mean()),
        "sens": lambda yy, pp: sensitivity(yy, pp, t),
        "spec": lambda yy, pp: specificity(yy, pp, t),
        "acc": lambda yy, pp: decision_accuracy(yy, pp, t),
    }
    for name, metric in truths.items():
        point, lo, hi = bootstrap_ci(metric, y, p, ids, n_boot=n_boot)
        row.update({f"true_{name}": point, f"true_{name}_lo": lo, f"true_{name}_hi": hi})
    prevalence, p_em = prevalence_em(p, prior_train)
    row["prevalence_cc"] = prevalence_cc(p, t)
    row["prevalence_acc"] = prevalence_acc(p, t, sensitivity(y_val, p_val, t), specificity(y_val, p_val, t))
    row["prevalence_em"] = prevalence
    row["sens_ac"], row["spec_ac"] = expected_sens_spec(p, t)
    row["sens_ac_em"], row["spec_ac_em"] = expected_sens_spec(p_em, t, decision_p=p)
    row["acc_ac"] = float(_confidence(p, t).mean())
    row["acc_atc"] = atc_accuracy(p, fit_atc(y_val, p_val, t), t)
    row["acc_doc"] = doc_accuracy(p, y_val, p_val, t)
    return row
