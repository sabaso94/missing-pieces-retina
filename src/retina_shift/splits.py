"""The EyePACS train / validation / test split, by patient, and the site metadata with that split attached.

EyePACS has both eyes of most patients. The eyes of one person look alike and usually share a grade, so a
split by image would put near-copies of test images into training. Splitting by patient avoids that.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
META_DIR = ROOT / "data" / "meta"
SPLITS_CSV = ROOT / "results" / "splits.csv"


def patient_split(patient_ids: np.ndarray, seed: int = 0, fractions=(0.7, 0.1, 0.2)) -> np.ndarray:
    """"train" / "val" / "test" for every row; all rows of a patient get the same split."""
    patients = np.unique(patient_ids)
    shuffled = np.random.default_rng(seed).permutation(patients)
    cut = np.cumsum(fractions)[:-1] * len(patients)
    names = np.empty(len(patients), dtype=object)
    names[:] = "test"
    position = np.arange(len(patients))
    names[position < cut[1]] = "val"
    names[position < cut[0]] = "train"
    split_of = dict(zip(shuffled, names))
    return np.array([split_of[p] for p in patient_ids])


def load_meta(site: str) -> pd.DataFrame:
    """Metadata of one site's cached images, with a `split` column.

    EyePACS rows get their saved patient split; every image of the other sites is "test" (never trained on).
    """
    meta = pd.read_csv(META_DIR / f"{site}.csv", dtype={"image_id": str, "patient_id": str})
    if site == "eyepacs":
        splits = pd.read_csv(SPLITS_CSV, dtype={"image_id": str})
        meta = meta.merge(splits, on="image_id", how="left", validate="one_to_one")
        if meta["split"].isna().any():
            raise ValueError("EyePACS images without a split; rerun scripts/cohort_report.py")
    else:
        meta["split"] = "test"
    return meta
