"""All figures of the README, from the CSVs in results/ (and one cached image per site).

    python scripts/make_figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from retina_shift.splits import ROOT, load_meta

RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
CACHE = ROOT / "data" / "cache"
SITES = ["eyepacs_test", "aptos", "messidor2", "idrid", "odir"]
LABELS = {"eyepacs_test": "EyePACS test (USA)", "aptos": "APTOS (India)", "messidor2": "Messidor-2 (France)",
          "idrid": "IDRiD (India)", "odir": "ODIR (China)"}
COLORS = {"eyepacs_test": "#4d4d4d", "aptos": "#e08214", "messidor2": "#2166ac", "idrid": "#b2182b",
          "odir": "#1b7837"}
DETECTORS = {"ks_outputs": "KS on outputs", "mmd_features": "MMD on features",
             "c2st_features": "Classifier test on features"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def log_axis(ax, ticks: list[int]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=150)
    plt.close(fig)


def sites_figure() -> None:
    fig, axes = plt.subplots(1, 5, figsize=(12.5, 2.9))
    rng = np.random.default_rng(3)
    for ax, name in zip(axes, SITES):
        site = "eyepacs" if name == "eyepacs_test" else name
        meta = load_meta(site)
        meta = meta[(meta["split"] == "test") & (meta["referable"] == 1)]
        image_id = meta["image_id"].iloc[rng.integers(len(meta))]
        ax.imshow(Image.open(CACHE / site / f"{image_id}.jpg"))
        ax.set_title(LABELS[name], fontsize=10)
        ax.axis("off")
    save(fig, "sites.png")


def e1_figure() -> None:
    bins = pd.read_csv(RESULTS / "e1_reliability.csv")
    sites = pd.read_csv(RESULTS / "e1_sites.csv").set_index("site")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--")
    for name in SITES:
        b = bins[(bins["site"] == name) & (bins["n"] >= 10)]
        ax1.plot(b["mean_p"], b["frac_pos"], "o-", color=COLORS[name], ms=4, label=LABELS[name])
    ax1.set(xlabel="Predicted probability of referable DR", ylabel="Observed share referable",
            title="Calibration (bins with at least 10 images)", xlim=(0, 1), ylim=(0, 1))
    ax1.legend(fontsize=8, frameon=False)
    x = np.arange(len(SITES))
    for offset, metric, marker in [(-0.12, "sens", "o"), (0.12, "spec", "s")]:
        values = sites.loc[SITES, metric]
        err = [values - sites.loc[SITES, f"{metric}_lo"], sites.loc[SITES, f"{metric}_hi"] - values]
        ax2.errorbar(x + offset, values, yerr=err, fmt=marker, color="#333333" if metric == "sens" else "#999999",
                     capsize=3, label="Sensitivity" if metric == "sens" else "Specificity")
    ax2.axhline(0.9, color="#333333", lw=0.8, ls=":")
    ax2.set_xticks(x, [LABELS[s].split(" (")[0] for s in SITES])
    ax2.set(ylabel="At the EyePACS threshold", title="Screening at a fixed threshold (95% CI)", ylim=(0, 1.02))
    ax2.legend(fontsize=8, frameon=False, loc="lower left")
    save(fig, "e1_sites.png")


def e2_figure() -> None:
    rates = pd.read_csv(RESULTS / "e2_detection.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
    for ax, (detector, title) in zip(axes, DETECTORS.items()):
        for name in SITES:
            r = rates[(rates["detector"] == detector) & (rates["pool"] == name)]
            ax.plot(r["batch_size"], r["detection_rate"], "o-", color=COLORS[name], ms=4, label=LABELS[name])
        ax.axhline(0.05, color="#999999", lw=0.8, ls="--")
        log_axis(ax, [10, 30, 100, 300, 1000])
        ax.set(title=title, xlabel="Unlabeled images in the batch")
    axes[0].set_ylabel("Share of batches flagged")
    axes[0].legend(fontsize=8, frameon=False)
    save(fig, "e2_detection.png")


def e3_figure() -> None:
    est = pd.read_csv(RESULTS / "e3_estimates.csv").set_index("site").loc[SITES]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    x = np.arange(len(SITES))
    panels = [(ax1, "prevalence", [("prevalence_cc", "Share referred", "^"), ("prevalence_acc", "Adjusted count", "v"),
                                   ("prevalence_em", "EM", "D")], "Prevalence of referable DR"),
              (ax2, "spec", [("spec_ac", "Model's own expectation", "^"),
                             ("spec_ac_em", "Same, after EM prevalence", "D")], "Specificity at the threshold")]
    for ax, truth, estimates, title in panels:
        values = est[f"true_{truth}"]
        ax.errorbar(x, values, yerr=[values - est[f"true_{truth}_lo"], est[f"true_{truth}_hi"] - values], fmt="o",
                    color="black", capsize=3, label="True (95% CI)")
        for k, (column, label, marker) in enumerate(estimates):
            ax.plot(x + 0.12 * (k + 1), est[column], marker, color=f"C{k}", label=label)
        ax.set_xticks(x, [LABELS[s].split(" (")[0] for s in SITES])
        ax.set(title=title)
        ax.legend(fontsize=8, frameon=False)
    save(fig, "e3_estimates.png")


def e4_figure() -> None:
    gaps = pd.read_csv(RESULTS / "e4_gaps.csv")
    mixture = pd.read_csv(RESULTS / "e4_mixture.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.0))
    comparisons = list(dict.fromkeys(gaps["group"] + " vs " + gaps["reference"]))
    for k, metric in enumerate(["sens", "spec", "auroc", "ece"]):
        g = gaps[gaps["metric"] == metric]
        y = np.arange(len(g)) + 0.18 * (k - 1.5)
        ax1.errorbar(g["gap"], y, xerr=[g["gap"] - g["lo"], g["hi"] - g["gap"]], fmt="o", capsize=3,
                     label={"sens": "Sensitivity", "spec": "Specificity", "auroc": "AUROC",
                            "ece": "ECE (higher is worse)"}[metric])
    ax1.axvline(0, color="#999999", lw=0.8)
    ax1.set_yticks(np.arange(len(comparisons)), [c.replace("sex=", "").replace("age=", "age ") for c in comparisons])
    ax1.set(title="ODIR: gaps between groups (95% CI)", xlabel="Group minus reference")
    ax1.legend(fontsize=8, frameon=False)
    full = mixture[mixture["share"] == 1.0]
    for k, (group, g) in enumerate(full.groupby("group")):
        label = group.replace("sex=", "").replace("age=", "age ")
        for detector, style in [("mmd_features", "o-"), ("ks_outputs", "o--")]:
            d = g[g["detector"] == detector]
            ax2.plot(d["batch_size"], d["detection_rate"], style, color=f"C{k}", ms=4,
                     label=f"{label}, {'features (kernel)' if detector == 'mmd_features' else 'outputs'}")
    null = mixture[mixture["share"] == 0.0]
    ax2.plot(null.groupby("batch_size")["detection_rate"].mean(), "k:", lw=1, label="no change (false alarms)")
    log_axis(ax2, [30, 100, 300, 600, 1000])
    ax2.set(title="Batches of one group only", xlabel="Unlabeled images in the batch",
            ylabel="Share of batches flagged", ylim=(0, 1.02))
    ax2.legend(fontsize=7, frameon=False)
    save(fig, "e4_odir.png")


def e5_figure() -> None:
    repair = pd.read_csv(RESULTS / "e5_repair.csv")
    sites = ["aptos", "messidor2", "idrid", "odir"]
    methods = {"none": ("No repair", "#999999"), "em_prior": ("EM prevalence, no labels", "C2"),
               "platt": ("Recalibration only", "C4"), "platt_threshold": ("Recalibration + new threshold", "C3")}
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
    for col, site in enumerate(sites):
        for row, metric in enumerate(["sens", "ece"]):
            ax = axes[row, col]
            for method, (label, color) in methods.items():
                if metric == "ece" and method == "platt":
                    continue  # same probabilities as recalibration + new threshold
                if metric == "sens" and method == "em_prior":
                    continue  # a calibration repair: referrals unchanged at a matching threshold
                r = repair[(repair["site"] == site) & (repair["method"] == method)]
                ax.plot(r["n"], r[metric], "o-", color=color, ms=3, label=label)
                ax.fill_between(r["n"], r[f"{metric}_lo"], r[f"{metric}_hi"], color=color, alpha=0.12)
            if metric == "sens":
                ax.axhline(0.9, color="black", lw=0.8, ls=":")
                ax.set_title(LABELS[site])
            log_axis(ax, [25, 50, 100, 200, 400])
        axes[1, col].set_xlabel("Labelled images from the clinic")
    axes[0, 0].set_ylabel("Sensitivity on the other images")
    axes[1, 0].set_ylabel("ECE on the other images")
    axes[0, 0].legend(fontsize=7, frameon=False)
    axes[1, 3].legend(fontsize=7, frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.88))
    save(fig, "e5_repair.png")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    sites_figure()
    e1_figure()
    e2_figure()
    e3_figure()
    e4_figure()
    e5_figure()


if __name__ == "__main__":
    main()
