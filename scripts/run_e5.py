"""E5: how many labelled images does a clinic need to make the model safe again?

    python scripts/run_e5.py

At each clinic, 200 times per size n, n images are labelled at random; recalibration and/or a new threshold
are fitted on them and scored on the clinic's other images. Writes results/e5_repair.csv (mean and 2.5 /
97.5 percentiles over repeats); every repeat goes to results/preds/e5_repair_reps.csv.
"""
import pandas as pd

from retina_shift.evaluate import EXTERNAL, arrays, load_predictions, threshold, train_prevalence
from retina_shift.recalibration import repair_curve
from retina_shift.splits import ROOT

SIZES = [25, 50, 100, 200, 400]


def main() -> None:
    t, prior = threshold(), train_prevalence()
    reps = []
    for seed, site in enumerate(EXTERNAL):
        y, p, _ = arrays(load_predictions(site))
        reps.append(repair_curve(y, p, t, SIZES, prior_train=prior, n_reps=200, seed=seed).assign(site=site))
    reps = pd.concat(reps, ignore_index=True)
    reps.to_csv(ROOT / "results" / "preds" / "e5_repair_reps.csv", index=False)

    rows = []
    for (site, n, method), g in reps.groupby(["site", "n", "method"], sort=False):
        row = {"site": site, "n": n, "method": method, "fallback_share": g["fallback"].mean()}
        for metric in ["sens", "spec", "ece"]:
            row[metric] = g[metric].mean()
            row[f"{metric}_lo"], row[f"{metric}_hi"] = g[metric].quantile([0.025, 0.975])
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "results" / "e5_repair.csv", index=False)
    print(summary.pivot_table(index=["site", "n"], columns="method", values="sens").round(3).to_string())


if __name__ == "__main__":
    main()
