"""E1: how the EyePACS model does on EyePACS test images and at four clinics it never saw.

    python scripts/run_e1.py

Fixes the screening threshold on EyePACS validation (90% sensitivity) and writes results/threshold.json,
e1_sites.csv (every metric with CIs), e1_diffs.csv (each site minus EyePACS test), e1_reliability.csv and
e1_seeds.csv (AUROC of each ensemble member).
"""
import json

import numpy as np
import pandas as pd

from retina_shift.evaluate import (
    EXTERNAL,
    PRED_DIR,
    THRESHOLD_JSON,
    arrays,
    evaluate_site,
    load_predictions,
    reliability_bins,
    site_metrics,
)
from retina_shift.metrics import auroc, diff_ci, sensitivity, specificity, threshold_for_sensitivity
from retina_shift.splits import ROOT

RESULTS = ROOT / "results"


def main() -> None:
    eyepacs = load_predictions("eyepacs")
    val = eyepacs[eyepacs["split"] == "val"]
    y_val, p_val, _ = arrays(val)
    t = threshold_for_sensitivity(y_val, p_val, 0.9)
    THRESHOLD_JSON.write_text(json.dumps({
        "threshold": t, "target_sensitivity": 0.9, "val_sensitivity": sensitivity(y_val, p_val, t),
        "val_specificity": specificity(y_val, p_val, t), "val_auroc": auroc(y_val, p_val)}, indent=2))
    print(f"threshold {t:.4f} (validation sensitivity {sensitivity(y_val, p_val, t):.3f})")

    frames = {"eyepacs_test": eyepacs[eyepacs["split"] == "test"]}
    frames.update({site: load_predictions(site) for site in EXTERNAL})

    rows, diffs, bins, seeds = [], [], [], []
    y_ref, p_ref, ids_ref = arrays(frames["eyepacs_test"])
    metrics = site_metrics(t)
    for name, frame in frames.items():
        y, p, ids = arrays(frame)
        rows.append({"site": name, **evaluate_site(y, p, ids, t)})
        bins.append(reliability_bins(y, p).assign(site=name))
        if name != "eyepacs_test":
            for metric in ["auroc", "sens", "spec", "ece"]:
                d, lo, hi = diff_ci(metrics[metric], y, p, ids, y_ref, p_ref, ids_ref)
                diffs.append({"site": name, "metric": metric, "diff": d, "lo": lo, "hi": hi})
        site = "eyepacs" if name == "eyepacs_test" else name
        p_seeds = np.load(PRED_DIR / f"{site}.npz")["p_seeds"][frame["pred_row"].to_numpy()]
        for seed in range(p_seeds.shape[1]):
            seeds.append({"site": name, "seed": seed, "auroc": auroc(y, p_seeds[:, seed])})
        print(f"{name}: AUROC {rows[-1]['auroc']:.3f}, sensitivity {rows[-1]['sens']:.3f}, "
              f"specificity {rows[-1]['spec']:.3f}, ECE {rows[-1]['ece']:.3f}")

    pd.DataFrame(rows).to_csv(RESULTS / "e1_sites.csv", index=False)
    pd.DataFrame(diffs).to_csv(RESULTS / "e1_diffs.csv", index=False)
    pd.concat(bins)[["site", "bin", "mean_p", "frac_pos", "n"]].to_csv(RESULTS / "e1_reliability.csv", index=False)
    pd.DataFrame(seeds).to_csv(RESULTS / "e1_seeds.csv", index=False)


if __name__ == "__main__":
    main()
