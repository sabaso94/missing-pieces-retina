import numpy as np

from retina_shift.shift import c2st_test, detection_rates, fit_pca, ks_test, median_bandwidth, mmd_test

REPS = 100


def _rates(shift: float, n: int = 100, dims: int = 32, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    ref_z = rng.normal(size=(1000, dims))
    ref_p = 1 / (1 + np.exp(-ref_z[:, 0]))
    bandwidth = median_bandwidth(ref_z)
    alarms = {"ks": 0, "mmd": 0, "c2st": 0}
    for rep in range(REPS):
        z = rng.normal(size=(n, dims)) + shift
        p = 1 / (1 + np.exp(-z[:, 0]))
        alarms["ks"] += ks_test(ref_p, p) < 0.05
        alarms["mmd"] += mmd_test(ref_z, z, bandwidth, n_perm=100, seed=rep) < 0.05
        alarms["c2st"] += c2st_test(ref_z, z, seed=rep) < 0.05
    return {k: v / REPS for k, v in alarms.items()}


def test_false_alarms_near_alpha_when_nothing_changed():
    for name, rate in _rates(shift=0.0).items():
        assert rate <= 0.10, (name, rate)


def test_every_test_detects_a_clear_shift():
    for name, rate in _rates(shift=0.5).items():
        assert rate >= 0.8, (name, rate)


def test_fit_pca_reduces_to_32_dims():
    z = np.random.default_rng(0).normal(size=(500, 100))
    assert fit_pca(z).transform(z).shape == (500, 32)


def test_detection_rates_skips_sizes_larger_than_the_pool():
    rng = np.random.default_rng(0)
    ref_z, pool_z = rng.normal(size=(300, 8)), rng.normal(size=(500, 8))
    ref_p, pool_p = rng.random(300), rng.random(500)
    rates = detection_rates(ref_p, ref_z, {"site": (pool_p, pool_z)}, sizes=[30, 1000], n_reps=5, n_perm=20)
    assert set(rates["batch_size"]) == {30}
    assert set(rates["detector"]) == {"ks_outputs", "mmd_features", "c2st_features"}
    assert rates["n_reps"].eq(5).all()
