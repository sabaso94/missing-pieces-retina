"""One site's performance table, and the ingredients every experiment script needs: predictions joined to
the metadata, the screening threshold, and the training prevalence."""

import json

import numpy as np
import pandas as pd

from retina_shift.metrics import (
    auroc,
    bootstrap_ci,
    brier,
    calibration_slope,
    citl,
    ece,
    sensitivity,
    specificity,
)
from retina_shift.splits import ROOT, load_meta

PRED_DIR = ROOT / "results" / "preds"
THRESHOLD_JSON = ROOT / "results" / "threshold.json"
EXTERNAL = ["aptos", "messidor2", "idrid", "odir"]


def site_metrics(t: float) -> dict:
    return {
        "auroc": auroc,
        "sens": lambda y, p: sensitivity(y, p, t),
        "spec": lambda y, p: specificity(y, p, t),
        "ece": ece,
        "citl": citl,
        "slope": calibration_slope,
        "brier": brier,
    }


def evaluate_site(y: np.ndarray, p: np.ndarray, ids: np.ndarray, t: float, n_boot: int = 1000) -> dict:
    """Every E1 metric with its 95% CI (clusters = patients where known)."""
    row = {"n": len(y), "prevalence": float(y.mean())}
    for name, metric in site_metrics(t).items():
        point, lo, hi = bootstrap_ci(metric, y, p, ids, n_boot=n_boot)
        row.update({name: point, f"{name}_lo": lo, f"{name}_hi": hi})
    return row


def reliability_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Mean prediction and observed share of referable eyes in equal-width probability bins."""
    bins = np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1])
    rows = [
        {
            "bin": b,
            "mean_p": float(p[bins == b].mean()),
            "frac_pos": float(y[bins == b].mean()),
            "n": int((bins == b).sum()),
        }
        for b in range(n_bins)
        if (bins == b).any()
    ]
    return pd.DataFrame(rows)


def load_predictions(site: str) -> pd.DataFrame:
    """Metadata of the predicted images with the ensemble probability `p` (and row `pred_row` into z)."""
    data = np.load(PRED_DIR / f"{site}.npz")
    preds = pd.DataFrame({"image_id": data["image_id"], "p": data["p"], "pred_row": np.arange(len(data["p"]))})
    meta = load_meta(site)
    return meta.merge(preds, on="image_id", how="inner", validate="one_to_one")


def load_features(site: str) -> np.ndarray:
    return np.load(PRED_DIR / f"{site}.npz")["z"]


def threshold() -> float:
    return json.loads(THRESHOLD_JSON.read_text())["threshold"]


def train_prevalence() -> float:
    meta = load_meta("eyepacs")
    return float(meta.loc[meta["split"] == "train", "referable"].mean())


def arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y, p, cluster ids) of a predictions frame."""
    return frame["referable"].to_numpy(), frame["p"].to_numpy(), frame["patient_id"].astype(str).to_numpy()
