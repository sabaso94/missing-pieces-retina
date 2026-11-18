import numpy as np
import pandas as pd

from retina_shift.metrics import auroc
from retina_shift.shift import mixture_batch
from retina_shift.subgroups import age_band, assign_age_bands, gap_ci, mixture_detection_rates, split_one_eye


def test_age_band_edges():
    assert [age_band(a) for a in [18, 49, 50, 64, 65, 90]] == ["<50", "<50", "50-64", "50-64", "65+", "65+"]


def test_small_age_band_is_merged_into_its_neighbour():
    ages = pd.Series([30] * 100 + [55] * 100 + [70] * 100)
    referable = pd.Series([1] * 10 + [0] * 90 + [1] * 60 + [0] * 40 + [1] * 60 + [0] * 40)
    bands = assign_age_bands(ages, referable, min_referable=50)
    assert set(bands) == {"<65", "65+"}  # "<50" has 10 referable eyes, merged with "50-64"
    assert set(assign_age_bands(ages, referable, min_referable=5)) == {"<50", "50-64", "65+"}


def test_gap_of_a_group_with_itself_is_zero():
    rng = np.random.default_rng(0)
    y, p = (rng.random(400) < 0.3).astype(int), rng.random(400)
    ids = np.repeat(np.arange(200), 2)
    everyone = np.ones(400, dtype=bool)
    assert gap_ci(auroc, y, p, ids, everyone, everyone, n_boot=50) == (0.0, 0.0, 0.0)


def test_mixture_batch_takes_an_exact_dose_of_the_group():
    # share s on top of the natural share pi: the group makes up exactly s + (1 - s) * pi of the batch
    in_group = np.zeros(1000, dtype=bool)
    in_group[:200] = True  # pi = 0.2
    rng = np.random.default_rng(0)
    for share, n, expected in [(0.0, 100, 20), (0.5, 300, 180), (0.25, 400, 160), (1.0, 150, 150)]:
        batch = mixture_batch(in_group, share=share, n=n, rng=rng)
        assert len(np.unique(batch)) == n and in_group[batch].sum() == expected, (share, n)


def test_mixture_batch_draws_without_replacement_and_enriches_the_group():
    rng = np.random.default_rng(0)
    in_group = np.zeros(1000, dtype=bool)
    in_group[:200] = True
    full = mixture_batch(in_group, share=1.0, n=150, rng=rng)
    assert in_group[full].all() and len(np.unique(full)) == 150
    half = mixture_batch(in_group, share=0.5, n=300, rng=rng)
    assert len(np.unique(half)) == 300 and in_group[half].mean() >= 0.5


def _paired_eyes(n_patients=600, seed=0):
    """Two eyes per patient that look alike (a shared patient effect), as at ODIR."""
    rng = np.random.default_rng(seed)
    ids = np.repeat(np.arange(n_patients), 2)
    z = rng.normal(size=(n_patients, 8))[ids] + 0.3 * rng.normal(size=(2 * n_patients, 8))
    return ids, z, 1 / (1 + np.exp(-z[:, 0]))


def test_split_one_eye_takes_one_eye_per_patient_and_halves_every_stratum():
    ids, _, _ = _paired_eyes(n_patients=101)
    strata = (ids % 3).astype(str)
    ref, pool = split_one_eye(ids, strata, np.random.default_rng(0))
    assert len(np.unique(ids[ref])) == len(ref) and len(np.unique(ids[pool])) == len(pool)
    assert not set(ids[ref]) & set(ids[pool]) and len(ref) + len(pool) == 101
    for s in np.unique(strata):
        n = len(np.unique(ids[strata == s]))
        assert (strata[ref] == s).sum() == n // 2 and (strata[pool] == s).sum() == n - n // 2


def test_split_one_eye_changes_between_repetitions():
    ids, _, _ = _paired_eyes(n_patients=100)
    rng = np.random.default_rng(0)
    first, second = (
        split_one_eye(ids, np.zeros(len(ids), dtype=str), rng),
        split_one_eye(ids, np.zeros(len(ids), dtype=str), rng),
    )
    assert set(first[0]) != set(second[0])


def test_mixture_null_false_alarms_near_alpha_with_paired_eyes():
    ids, z, p = _paired_eyes()
    in_g = ids % 2 == 0
    rates = mixture_detection_rates(
        p, z, ids, {"g": in_g}, in_g.astype(str), shares=[0.0], sizes=[200], n_reps=100, n_perm=50
    )
    assert (rates["detection_rate"] <= 0.12).all(), rates


def test_mixture_skips_batches_larger_than_the_pool_can_hold():
    ids, z, p = _paired_eyes(n_patients=100)
    in_g = ids < 40  # 40 patients -> 20 of the 50 images in every pool (pi = 0.4)
    rates = mixture_detection_rates(
        p, z, ids, {"g": in_g}, in_g.astype(str), shares=[0.0, 0.5, 1.0], sizes=[20, 40], n_reps=2, n_perm=10
    )
    # share 0.5 needs 0.7 of the batch from the group: 14 of 20 fits, 28 of 40 does not
    assert set(zip(rates["share"], rates["batch_size"])) == {(0.0, 20), (0.0, 40), (0.5, 20), (1.0, 20)}
