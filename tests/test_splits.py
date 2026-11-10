import numpy as np

from retina_shift.splits import patient_split


def _ids(n_patients=10_000):
    return np.repeat(np.arange(n_patients).astype(str), 2)  # two eyes per patient


def test_both_eyes_of_a_patient_share_a_split():
    ids = _ids()
    split = patient_split(ids)
    assert (split[0::2] == split[1::2]).all()


def test_split_shares_are_70_10_20():
    split = patient_split(_ids())
    for name, share in [("train", 0.7), ("val", 0.1), ("test", 0.2)]:
        assert abs((split == name).mean() - share) <= 0.01


def test_split_is_fixed_by_seed():
    ids = _ids(500)
    assert (patient_split(ids, seed=3) == patient_split(ids, seed=3)).all()
    assert (patient_split(ids, seed=3) != patient_split(ids, seed=4)).any()
