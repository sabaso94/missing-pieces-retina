"""Who is in each site, and the EyePACS patient split.

    python scripts/cohort_report.py

Writes results/splits.csv (EyePACS image -> train / val / test), results/cohort.csv (one row per site and,
for EyePACS, per split) and results/cohort_odir.csv (ODIR by sex and age band).
"""

import pandas as pd

from retina_shift.sites import SITES
from retina_shift.splits import META_DIR, ROOT, SPLITS_CSV, load_meta, patient_split

AGE_BANDS = [("<50", 0, 50), ("50-64", 50, 65), ("65+", 65, 200)]


def summary(name: str, meta: pd.DataFrame) -> dict:
    row = {
        "site": name,
        "images": len(meta),
        "patients": meta["patient_id"].nunique(),
        "referable": int(meta["referable"].sum()),
        "referable_pct": round(100 * meta["referable"].mean(), 1),
    }
    for grade in range(5):
        row[f"grade_{grade}"] = int((meta["grade"] == grade).sum())
    return row


def main() -> None:
    eyepacs = pd.read_csv(META_DIR / "eyepacs.csv", dtype={"image_id": str, "patient_id": str})
    pd.DataFrame({"image_id": eyepacs["image_id"], "split": patient_split(eyepacs["patient_id"].to_numpy())}).to_csv(
        SPLITS_CSV, index=False
    )

    rows = []
    for site in SITES:
        meta = load_meta(site)
        if site == "eyepacs":
            for split in ["train", "val", "test"]:
                rows.append(summary(f"eyepacs_{split}", meta[meta["split"] == split]))
        else:
            rows.append(summary(site, meta))
    cohort = pd.DataFrame(rows)
    cohort.to_csv(ROOT / "results" / "cohort.csv", index=False)
    print(cohort.to_string(index=False))

    odir = load_meta("odir")
    groups = [("all", odir)] + [(f"sex={s}", odir[odir["sex"] == s]) for s in ["female", "male"]]
    groups += [(f"age={name}", odir[(odir["age"] >= lo) & (odir["age"] < hi)]) for name, lo, hi in AGE_BANDS]
    odir_rows = pd.DataFrame([summary(name, g) for name, g in groups]).rename(columns={"site": "group"})
    odir_rows.to_csv(ROOT / "results" / "cohort_odir.csv", index=False)
    print(odir_rows.to_string(index=False))


if __name__ == "__main__":
    main()
