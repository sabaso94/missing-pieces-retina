"""E3: can the model's own outputs tell a clinic how well it is doing there, without labels?

    python scripts/run_e3.py

Every estimator is fitted on EyePACS validation and applied to each clinic's images; the true values (with
CIs) sit next to the estimates. Writes results/e3_estimates.csv and results/e3_errors.csv: for each
estimator, its mean and largest error over the four clinics, and whether it orders the clinics the same way
as the truth (Spearman, descriptive: four sites).
"""
import pandas as pd
from scipy.stats import spearmanr

from retina_shift.evaluate import EXTERNAL, arrays, load_predictions, threshold, train_prevalence
from retina_shift.labelfree import estimate_site
from retina_shift.splits import ROOT

ESTIMATES = {
    "prevalence": ["prevalence_cc", "prevalence_acc", "prevalence_em"],
    "sens": ["sens_ac", "sens_ac_em"],
    "spec": ["spec_ac", "spec_ac_em"],
    "acc": ["acc_ac", "acc_atc", "acc_doc"],
}


def main() -> None:
    t, prior = threshold(), train_prevalence()
    eyepacs = load_predictions("eyepacs")
    y_val, p_val, _ = arrays(eyepacs[eyepacs["split"] == "val"])
    frames = {"eyepacs_test": eyepacs[eyepacs["split"] == "test"]}
    frames.update({site: load_predictions(site) for site in EXTERNAL})

    rows = []
    for name, frame in frames.items():
        y, p, ids = arrays(frame)
        rows.append({"site": name, **estimate_site(y, p, ids, y_val, p_val, t, prior)})
    estimates = pd.DataFrame(rows)
    estimates.to_csv(ROOT / "results" / "e3_estimates.csv", index=False)
    show = ["site", "true_prevalence", "prevalence_cc", "prevalence_acc", "prevalence_em", "true_sens",
            "sens_ac", "sens_ac_em", "true_spec", "spec_ac", "spec_ac_em", "true_acc", "acc_ac", "acc_atc",
            "acc_doc"]
    print(estimates[show].round(3).to_string(index=False))

    external = estimates[estimates["site"].isin(EXTERNAL)]
    errors = []
    for truth, names in ESTIMATES.items():
        for name in names:
            error = external[name] - external[f"true_{truth}"]
            errors.append({"quantity": truth, "estimate": name, "mean_error": float(error.mean()),
                           "max_abs_error": float(error.abs().max()),
                           "spearman": float(spearmanr(external[f"true_{truth}"], external[name]).statistic),
                           "sites": len(external)})
    pd.DataFrame(errors).to_csv(ROOT / "results" / "e3_errors.csv", index=False)
    print(pd.DataFrame(errors).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
