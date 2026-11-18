"""E2: can a clinic's unlabeled images be told apart from the training data, and from how many images?

    python scripts/run_e2.py --device cuda

Reference: EyePACS validation images; features reduced by a PCA fitted on EyePACS training images. Batches of 10-1000
images are drawn (without replacement) from
EyePACS test (false alarms) and from each clinic, 200 times per size; three tests per batch.
Writes results/e2_detection.csv.
"""

import argparse

from retina_shift.evaluate import EXTERNAL, load_features, load_predictions
from retina_shift.shift import detection_rates, fit_pca, reduce
from retina_shift.splits import ROOT

SIZES = [10, 30, 100, 300, 1000]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reps", type=int, default=200)
    args = parser.parse_args()

    eyepacs = load_predictions("eyepacs")
    z_eyepacs = load_features("eyepacs")
    train = eyepacs[eyepacs["split"] == "train"]
    val, test = eyepacs[eyepacs["split"] == "val"], eyepacs[eyepacs["split"] == "test"]
    pca = fit_pca(z_eyepacs[train["pred_row"]])
    ref_p, ref_z = val["p"].to_numpy(), reduce(pca, z_eyepacs[val["pred_row"]])

    pools = {"eyepacs_test": (test["p"].to_numpy(), reduce(pca, z_eyepacs[test["pred_row"]]))}
    for site in EXTERNAL:
        frame = load_predictions(site)
        pools[site] = (frame["p"].to_numpy(), reduce(pca, load_features(site)[frame["pred_row"]]))
    rates = detection_rates(ref_p, ref_z, pools, SIZES, n_reps=args.reps, device=args.device)
    rates.to_csv(ROOT / "results" / "e2_detection.csv", index=False)
    print(rates.pivot_table(index=["pool", "batch_size"], columns="detector", values="detection_rate").to_string())


if __name__ == "__main__":
    main()
