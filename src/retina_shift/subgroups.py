"""Subgroups at ODIR (the one site that records age and sex): gaps between groups, and whether a change in
the age or sex mix of a clinic's patients can be detected without labels."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from retina_shift.metrics import _cluster_resample_indices
from retina_shift.shift import c2st_test, group_count, ks_test, median_bandwidth, mixture_batch, mmd_test

AGE_EDGES = [(None, 50, "<50"), (50, 65, "50-64"), (65, None, "65+")]


def age_band(age: float) -> str:
    for lo, hi, name in AGE_EDGES:
        if (lo is None or age >= lo) and (hi is None or age < hi):
            return name
    raise ValueError(age)


def assign_age_bands(ages: pd.Series, referable: pd.Series, min_referable: int = 50) -> pd.Series:
    """Age band per eye. A band with fewer than `min_referable` referable eyes gives sensitivities too
    noisy to compare, so it is merged with the middle band ("<50" -> "<65", "65+" -> "50+")."""
    bands = ages.map(age_band)
    counts = referable.groupby(bands).sum()
    if counts.get("<50", 0) < min_referable:
        bands = bands.replace({"<50": "<65", "50-64": "<65"})
    elif counts.get("65+", 0) < min_referable:
        bands = bands.replace({"65+": "50+", "50-64": "50+"})
    return bands


def gap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    p: np.ndarray,
    ids: np.ndarray,
    in_group: np.ndarray,
    in_ref: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """metric(group) - metric(reference group), with a 95% CI from patient resamples of the whole site
    (each resample is split into the two groups, so the groups keep their natural sizes on average)."""
    rng = np.random.default_rng(seed)
    point = metric(y[in_group], p[in_group]) - metric(y[in_ref], p[in_ref])
    samples = []
    for _ in range(n_boot):
        idx = _cluster_resample_indices(ids, rng)
        g, r = idx[in_group[idx]], idx[in_ref[idx]]
        samples.append(metric(y[g], p[g]) - metric(y[r], p[r]))
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def split_one_eye(ids: np.ndarray, strata: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """(reference, pool) image indices: one random eye per patient, and the patients of every stratum split
    in half (floor to the reference). `strata` must be constant within a patient."""
    order = rng.permutation(len(ids))
    _, first = np.unique(ids[order], return_index=True)
    eyes = order[first]
    ref, pool = [], []
    for stratum in np.unique(strata[eyes]):
        members = rng.permutation(eyes[strata[eyes] == stratum])
        ref.append(members[: len(members) // 2])
        pool.append(members[len(members) // 2 :])
    return np.concatenate(ref), np.concatenate(pool)


def mixture_detection_rates(
    p: np.ndarray,
    z: np.ndarray,
    ids: np.ndarray,
    groups: dict[str, np.ndarray],
    strata: np.ndarray,
    shares: list[float],
    sizes: list[int],
    n_reps: int = 200,
    seed: int = 0,
    device: str = "cpu",
    alpha: float = 0.05,
    n_perm: int = 200,
) -> pd.DataFrame:
    """Alarm rates of the three tests on batches enriched in one group, within one site (share 0 = false
    alarms). `z` must already be in the reduced (PCA) space.

    Every repetition draws a new reference and pool with `split_one_eye`: the two eyes of a patient look
    alike, and a single fixed split would make every batch share that split's chance difference, so neither
    gives the nominal false-alarm rate. Each group must be a union of strata, so that the pool always holds
    the same number of its images; a batch that the pool cannot hold is skipped. Batches take an exact dose
    of the group (`mixture_batch`): share 0 is the natural mix, share 1 the group only.
    """
    rng = np.random.default_rng(seed)
    patient_strata = pd.Series(strata).groupby(ids).first()
    pool_size = {s: n - n // 2 for s, n in patient_strata.value_counts().items()}
    rows = []
    for group, in_group in groups.items():
        group_strata = set(strata[in_group])
        if group_strata & set(strata[~in_group]):
            raise ValueError(f"{group} is not a union of strata")
        n_pool, n_group = sum(pool_size.values()), sum(pool_size[s] for s in group_strata)
        for share in shares:
            for size in sizes:
                k = group_count(share, n_group / n_pool, size)
                if size > n_pool or k > n_group or size - k > n_pool - n_group:
                    continue
                alarms = {"ks_outputs": 0, "mmd_features": 0, "c2st_features": 0}
                for rep in range(n_reps):
                    ref, pool = split_one_eye(ids, strata, rng)
                    batch = pool[mixture_batch(in_group[pool], share, size, rng)]
                    alarms["ks_outputs"] += ks_test(p[ref], p[batch]) < alpha
                    alarms["mmd_features"] += (
                        mmd_test(z[ref], z[batch], median_bandwidth(z[ref]), n_perm, seed + rep, device) < alpha
                    )
                    alarms["c2st_features"] += c2st_test(z[ref], z[batch], seed + rep) < alpha
                for detector, count in alarms.items():
                    rows.append(
                        {
                            "group": group,
                            "share": share,
                            "batch_size": size,
                            "detector": detector,
                            "detection_rate": count / n_reps,
                            "n_reps": n_reps,
                        }
                    )
    return pd.DataFrame(rows)
