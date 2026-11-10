import numpy as np

from retina_shift.shift import c2st_test, detection_rates, fit_pca, ks_test, median_bandwidth, mmd_test, reduce

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


def _noise(n, dims, rng):
    return rng.normal(size=(n, dims))


def test_pca_must_be_fitted_on_separate_training_features():
    """PCA fitted on the reference overfits it: the reference then looks more spread out along the kept
    directions than any new sample from the same distribution, and the feature tests raise false alarms.
    Fitted on separate (training) features, false alarms stay near alpha."""
    rng = np.random.default_rng(0)
    train, ref, pool = _noise(2000, 256, rng), _noise(500, 256, rng), _noise(3000, 256, rng)
    ref_p, pool_p = rng.random(500), rng.random(3000)

    def rates(pca):
        r = detection_rates(ref_p, reduce(pca, ref), {"same": (pool_p, reduce(pca, pool))}, sizes=[100],
                            n_reps=40, n_perm=100)
        return r.set_index("detector")["detection_rate"]

    assert rates(fit_pca(ref))["mmd_features"] >= 0.5
    assert (rates(fit_pca(train)) <= 0.15).all()
