import numpy as np
import pandas as pd

from retina_shift.metrics import auroc
from retina_shift.shift import mixture_batch
from retina_shift.subgroups import age_band, assign_age_bands, gap_ci


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


def test_mixture_batch_draws_without_replacement_and_enriches_the_group():
    rng = np.random.default_rng(0)
    in_group = np.zeros(1000, dtype=bool)
    in_group[:200] = True
    full = mixture_batch(in_group, share=1.0, n=150, rng=rng)
    assert in_group[full].all() and len(np.unique(full)) == 150
    half = mixture_batch(in_group, share=0.5, n=300, rng=rng)
    assert len(np.unique(half)) == 300 and in_group[half].mean() >= 0.5
