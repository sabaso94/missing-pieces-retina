"""One label table per site, read straight from the Kaggle archives.

Every table has the same columns:
    image_id   unique within the site, also the cache file name
    member     path of the image inside the zip
    zip        archive file name (in the zip directory)
    grade      ICDR diabetic retinopathy grade 0-4
    referable  1 if grade >= 2 (moderate non-proliferative retinopathy or worse)
    patient_id the patient, where the site records it; otherwise the image itself
    eye        "left" / "right" where known
    age, sex   ODIR only
"""

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SITES = ["eyepacs", "aptos", "messidor2", "idrid", "odir"]
REFERABLE_GRADE = 2

ZIPS = {
    "eyepacs": "diabetic-retinopathy-resized.zip",
    "aptos": "aptos2019.zip",
    "messidor2": "messidor2preprocess.zip",
    "messidor2_grades": "messidor2-dr-grades.zip",
    "idrid": "indian-diabetic-retinopathy-image-dataset.zip",
    "odir": "ocular-disease-recognition-odir5k.zip",
}
KAGGLE = {
    "eyepacs": ["tanlikesmath/diabetic-retinopathy-resized"],
    "aptos": ["mariaherrerot/aptos2019"],
    "messidor2": ["mariaherrerot/messidor2preprocess", "google-brain/messidor2-dr-grades"],
    "idrid": ["aaryapatel98/indian-diabetic-retinopathy-image-dataset"],
    "odir": ["andrewmvd/ocular-disease-recognition-odir5k"],
}

_EYEPACS_LABELS = "trainLabels.csv"
_EYEPACS_DIR = "resized_train/resized_train/"
_APTOS_LABELS = {"train": "train_1.csv", "val": "valid.csv", "test": "test.csv"}
_MESSIDOR_GRADES = "messidor_data.csv"
_IDRID_ROOT = "B.%20Disease%20Grading/B. Disease Grading/"
_IDRID_LABELS = {
    "train": _IDRID_ROOT + "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
    "test": _IDRID_ROOT + "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
}
_IDRID_IMAGES = {
    "train": _IDRID_ROOT + "1. Original Images/a. Training Set/",
    "test": _IDRID_ROOT + "1. Original Images/b. Testing Set/",
}
_ODIR_TABLE = "ODIR-5K/ODIR-5K/data.xlsx"
_ODIR_IMAGES = "ODIR-5K/ODIR-5K/Training Images/"

COLUMNS = ["image_id", "member", "grade", "referable", "patient_id", "eye", "age", "sex"]


def _finish(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["grade"] = rows["grade"].astype(int)
    rows["referable"] = (rows["grade"] >= REFERABLE_GRADE).astype(int)
    for column in ["eye", "age", "sex"]:
        if column not in rows:
            rows[column] = np.nan
    if "patient_id" not in rows:
        rows["patient_id"] = rows["image_id"]
    rows["patient_id"] = rows["patient_id"].astype(str)
    return rows[COLUMNS].reset_index(drop=True)


def odir_grade(keywords: str) -> float:
    """ICDR grade of one ODIR eye from its diagnostic keywords; NaN when retinopathy is named but not graded.

    ODIR writes one keyword per finding, separated by Chinese or Latin commas, for example
    "moderate non proliferative retinopathy，cataract". The highest graded DR keyword wins. An eye whose only
    DR keyword is ungraded ("diabetic retinopathy") or uncertain ("suspected ...") cannot be labelled and is
    excluded. Eyes with no DR keyword (normal or other diseases only) are grade 0.
    """
    grades, unspecified = [], False
    for keyword in re.split(r"[，,]", str(keywords)):
        k = keyword.strip().lower()
        nonproliferative = "nonproliferative" in k or "non proliferative" in k
        if "suspect" in k:
            unspecified = unspecified or "retinopathy" in k
        elif "proliferative" in k and not nonproliferative:
            grades.append(4)
        elif nonproliferative:
            if "severe" in k:
                grades.append(3)
            elif "moderate" in k:
                grades.append(2)
            elif "mild" in k:
                grades.append(1)
            else:
                unspecified = True
        elif "diabetic retinopathy" in k:
            unspecified = True
    if grades:
        return float(max(grades))
    return float("nan") if unspecified else 0.0


def eyepacs_rows(labels: pd.DataFrame, members: list[str]) -> pd.DataFrame:
    """EyePACS (Kaggle 2015 training set): names are "<patient>_<eye>"."""
    available = {m.rsplit("/", 1)[-1].rsplit(".", 1)[0]: m for m in members if m.startswith(_EYEPACS_DIR)}
    labels = labels[labels["image"].isin(available)]
    patient_eye = labels["image"].str.rsplit("_", n=1)
    return _finish(
        pd.DataFrame(
            {
                "image_id": labels["image"].to_numpy(),
                "member": labels["image"].map(available).to_numpy(),
                "grade": labels["level"].to_numpy(),
                "patient_id": patient_eye.str[0].to_numpy(),
                "eye": patient_eye.str[1].to_numpy(),
            }
        )
    )


def messidor_rows(grades: pd.DataFrame, members: list[str]) -> pd.DataFrame:
    """Messidor-2 with the adjudicated grades of Krause et al. (2018); gradable images only.

    The Kaggle copy writes some extensions in upper case (".JPG"), so names are matched case-insensitively.
    """
    by_name = {m.rsplit("/", 1)[-1].lower(): m for m in members if not m.endswith(".csv")}
    grades = grades[(grades["adjudicated_gradable"] == 1) & grades["image_id"].str.lower().isin(by_name)]
    return _finish(
        pd.DataFrame(
            {
                "image_id": grades["image_id"].str.rsplit(".", n=1).str[0].to_numpy(),
                "member": grades["image_id"].str.lower().map(by_name).to_numpy(),
                "grade": grades["adjudicated_dr_grade"].to_numpy(),
            }
        )
    )


def idrid_rows(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """IDRiD disease grading, training and test sets together. Both sets number their images from
    IDRiD_001, so the set name is part of the image id."""
    parts = []
    for name, table in [("train", train), ("test", test)]:
        table = table.dropna(subset=["Image name"])
        parts.append(
            pd.DataFrame(
                {
                    "image_id": name + "_" + table["Image name"].astype(str),
                    "member": _IDRID_IMAGES[name] + table["Image name"].astype(str) + ".jpg",
                    "grade": table["Retinopathy grade"].to_numpy(),
                }
            )
        )
    return _finish(pd.concat(parts, ignore_index=True))


def odir_rows(table: pd.DataFrame) -> pd.DataFrame:
    """ODIR-5K training set, one row per eye, grade from the eye's diagnostic keywords."""
    parts = []
    for side, eye in [("Left", "left"), ("Right", "right")]:
        parts.append(
            pd.DataFrame(
                {
                    "image_id": table[f"{side}-Fundus"].str.rsplit(".", n=1).str[0],
                    "member": _ODIR_IMAGES + table[f"{side}-Fundus"],
                    "grade": table[f"{side}-Diagnostic Keywords"].map(odir_grade),
                    "patient_id": table["ID"].astype(str),
                    "eye": eye,
                    "age": table["Patient Age"],
                    "sex": table["Patient Sex"].str.lower(),
                    "_order": table["ID"] * 2 + (eye == "right"),
                }
            )
        )
    rows = pd.concat(parts).sort_values("_order").dropna(subset=["grade"])
    return _finish(rows)


def aptos_rows(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """APTOS 2019 (Aravind Eye Hospital): the competition's 3,662 labelled images, which this Kaggle copy
    splits into train / val / test folders. All three are used: the model never trains on APTOS."""
    parts = [
        pd.DataFrame(
            {
                "image_id": table["id_code"].astype(str),
                "member": f"{split}_images/{split}_images/" + table["id_code"].astype(str) + ".png",
                "grade": table["diagnosis"].to_numpy(),
            }
        )
        for split, table in tables.items()
    ]
    return _finish(pd.concat(parts, ignore_index=True))


def site_table(site: str, zip_dir: Path) -> pd.DataFrame:
    """The label table of one site, read from its archive(s) in zip_dir."""
    zip_dir = Path(zip_dir)
    archive = zipfile.ZipFile(zip_dir / ZIPS[site])
    members = archive.namelist()

    def read_csv(name: str, zf: zipfile.ZipFile = archive) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(zf.read(name)))

    if site == "eyepacs":
        rows = eyepacs_rows(read_csv(_EYEPACS_LABELS), members)
    elif site == "aptos":
        rows = aptos_rows({split: read_csv(name) for split, name in _APTOS_LABELS.items()})
    elif site == "messidor2":
        grades = read_csv(_MESSIDOR_GRADES, zipfile.ZipFile(zip_dir / ZIPS["messidor2_grades"]))
        rows = messidor_rows(grades, members)
    elif site == "idrid":
        rows = idrid_rows(read_csv(_IDRID_LABELS["train"]), read_csv(_IDRID_LABELS["test"]))
    elif site == "odir":
        rows = odir_rows(pd.read_excel(io.BytesIO(archive.read(_ODIR_TABLE))))
    else:
        raise ValueError(f"unknown site {site}")
    rows.insert(2, "zip", ZIPS[site])
    return rows
