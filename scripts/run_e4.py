"""E4: ODIR, the clinic that records age and sex. Is the model equally good for women and men and across
age bands, and does a change in the clinic's age or sex mix show up in the shift tests?

    python scripts/run_e4.py --device cuda

Writes results/e4_groups.csv, e4_gaps.csv (female - male; each age band - 50-64) and e4_mixture.csv.
For the mixture part, every repetition takes one random eye per patient and splits ODIR's patients in half
within each sex and age band: one half is the reference, the batch comes from the other, so the test compares
ODIR with ODIR and only the group mix differs.
"""

import argparse

import pandas as pd

from retina_shift.evaluate import arrays, load_features, load_predictions, site_metrics, threshold
from retina_shift.metrics import bootstrap_ci
from retina_shift.shift import fit_pca, reduce
from retina_shift.splits import ROOT
from retina_shift.subgroups import assign_age_bands, gap_ci, mixture_detection_rates

METRICS = ["auroc", "sens", "spec", "ece", "citl"]
SHARES = [0.0, 0.25, 0.5, 1.0]
SIZES = [30, 100, 300, 600, 1000]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()

    t = threshold()
    metrics = site_metrics(t)
    odir = load_predictions("odir")
    odir["age_band"] = assign_age_bands(odir["age"], odir["referable"])
    y, p, ids = arrays(odir)

    groups = {f"sex={s}": (odir["sex"] == s).to_numpy() for s in ["female", "male"]}
    groups.update({f"age={b}": (odir["age_band"] == b).to_numpy() for b in sorted(odir["age_band"].unique())})
    rows = []
    for name, mask in groups.items():
        row = {"group": name, "n": int(mask.sum()), "referable": int(y[mask].sum())}
        for metric in METRICS:
            point, lo, hi = bootstrap_ci(metrics[metric], y[mask], p[mask], ids[mask])
            row.update({metric: point, f"{metric}_lo": lo, f"{metric}_hi": hi})
        rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "e4_groups.csv", index=False)
    print(pd.DataFrame(rows)[["group", "n", "referable", *METRICS]].round(3).to_string(index=False))

    reference_band = "age=50-64" if "age=50-64" in groups else None
    comparisons = [("sex=female", "sex=male")]
    comparisons += [
        (g, reference_band) for g in groups if g.startswith("age=") and reference_band and g != reference_band
    ]
    gaps = []
    for group, ref in comparisons:
        for metric in METRICS:
            gap, lo, hi = gap_ci(metrics[metric], y, p, ids, groups[group], groups[ref])
            gaps.append({"group": group, "reference": ref, "metric": metric, "gap": gap, "lo": lo, "hi": hi})
    pd.DataFrame(gaps).to_csv(ROOT / "results" / "e4_gaps.csv", index=False)
    print(pd.DataFrame(gaps).round(3).to_string(index=False))

    z = load_features("odir")[odir["pred_row"]]
    eyepacs = load_predictions("eyepacs")
    pca = fit_pca(load_features("eyepacs")[eyepacs.loc[eyepacs["split"] == "train", "pred_row"]])
    strata = (odir["sex"] + "/" + odir["age_band"]).to_numpy()
    mix_groups = {name: mask for name, mask in groups.items() if name != reference_band}
    mixture = mixture_detection_rates(
        p, reduce(pca, z), ids, mix_groups, strata, SHARES, SIZES, n_reps=args.reps, device=args.device
    )
    mixture.to_csv(ROOT / "results" / "e4_mixture.csv", index=False)
    print(
        mixture.pivot_table(
            index=["group", "share", "batch_size"], columns="detector", values="detection_rate"
        ).to_string()
    )


if __name__ == "__main__":
    main()
