"""Detecting that a clinic's images differ from the training data, without labels (two-sample tests).

A deployed model sees batches of new images. Each test compares a batch with a reference sample (EyePACS
validation images) and returns a p-value; an alarm is p < alpha. Three tests, at two places in the model:

- outputs: KS test on the predicted probabilities ("black-box shift detection", Rabanser et al., 2019);
- features: kernel MMD test (Gretton et al., 2012) and a classifier two-sample test (C2ST, Lopez-Paz &
  Oquab, 2017) on the network's image features, reduced to 32 dimensions by PCA fitted on EyePACS
  training images.
  Classifier-based tests did best in Koch et al.'s retinal subgroup-shift studies.
"""
import numpy as np
import pandas as pd
import torch
from scipy.stats import binom, ks_2samp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

PCA_DIMS = 32


def fit_pca(train_z: np.ndarray, dims: int = PCA_DIMS) -> PCA:
    """PCA for the feature tests, fitted on training-image features: never on the reference or a batch.

    Fitted on the reference, PCA overfits it: the reference then spreads more along the kept directions than
    any new sample from the same distribution, and the tests flag that difference as a shift.
    """
    return PCA(n_components=dims, random_state=0).fit(train_z)


def reduce(pca: PCA, z: np.ndarray) -> np.ndarray:
    """Features projected onto the PCA directions, as float64."""
    return pca.transform(z).astype(np.float64)


def ks_test(ref_p: np.ndarray, batch_p: np.ndarray) -> float:
    """p-value of the two-sample Kolmogorov-Smirnov test on predicted probabilities."""
    return float(ks_2samp(ref_p, batch_p).pvalue)


def median_bandwidth(z: np.ndarray, max_n: int = 1000, seed: int = 0) -> float:
    """Median pairwise distance in the reference: the usual default kernel width for MMD."""
    rng = np.random.default_rng(seed)
    if len(z) > max_n:
        z = z[rng.choice(len(z), size=max_n, replace=False)]
    d = torch.cdist(torch.from_numpy(z).double(), torch.from_numpy(z).double())
    return float(d[torch.triu_indices(len(z), len(z), offset=1).unbind()].median())


def mmd_test(ref_z: np.ndarray, batch_z: np.ndarray, bandwidth: float, n_perm: int = 200, seed: int = 0,
             device: str = "cpu") -> float:
    """Permutation p-value of the (biased) squared MMD with a Gaussian kernel.

    The kernel matrix is built once. A split of the pooled sample into "reference" and "batch" is a weight
    vector a (1/n_ref on one side, -1/n_batch on the other), and MMD^2 = a^T K a, so all permutations come
    from one matrix product.
    """
    z = torch.from_numpy(np.concatenate([ref_z, batch_z])).to(device, torch.float64)
    n_ref, n = len(ref_z), len(z)
    kernel = torch.exp(-torch.cdist(z, z) ** 2 / (2 * bandwidth**2))
    generator = torch.Generator().manual_seed(seed)
    is_ref = torch.zeros(n_perm + 1, n, dtype=torch.bool)
    is_ref[0, :n_ref] = True  # row 0: the observed split
    for i in range(1, n_perm + 1):
        is_ref[i, torch.randperm(n, generator=generator)[:n_ref]] = True
    weights = torch.where(is_ref, 1.0 / n_ref, -1.0 / (n - n_ref)).T.to(device, torch.float64)
    stats = ((kernel @ weights) * weights).sum(dim=0)
    return float((1 + (stats[1:] >= stats[0]).sum()) / (1 + n_perm))


def c2st_test(ref_z: np.ndarray, batch_z: np.ndarray, seed: int = 0) -> float:
    """Classifier two-sample test (Lopez-Paz & Oquab 2017): can a classifier tell batch images from
    reference images?

    An equal number of reference images is drawn; a random half of the 2n images fits a logistic
    regression, which then classifies the other half. If the two samples come from the same distribution,
    the number of correct calls on the held-out half is Binomial(n_held_out, 1/2); the p-value is the chance
    of at least as many correct calls. (Cross-validated predictions are not independent enough for this
    binomial test: on heavy-tailed features they raise too many false alarms.)
    """
    rng = np.random.default_rng(seed)
    n = len(batch_z)
    ref = ref_z[rng.choice(len(ref_z), size=n, replace=False)]
    x = np.concatenate([ref, batch_z])
    y = np.r_[np.zeros(n), np.ones(n)]
    order = rng.permutation(2 * n)
    fit, held_out = order[:n], order[n:]
    if len(np.unique(y[fit])) < 2:
        return 1.0
    predicted = LogisticRegression(max_iter=1000).fit(x[fit], y[fit]).predict(x[held_out])
    correct = int((predicted == y[held_out]).sum())
    return float(binom.sf(correct - 1, len(held_out), 0.5))


def detection_rates(ref_p: np.ndarray, ref_z: np.ndarray, pools: dict[str, tuple[np.ndarray, np.ndarray]],
                    sizes: list[int], n_reps: int = 200, seed: int = 0, device: str = "cpu",
                    alpha: float = 0.05, n_perm: int = 200) -> pd.DataFrame:
    """Share of batches drawn from each pool in which each test raises an alarm (p < alpha).

    `ref_z` and the pools' z must already be in the reduced (PCA) space. Batches are drawn without
    replacement; a batch size larger than the pool is skipped. A pool from the training distribution gives
    the false-alarm rate. All three tests see the same batches.
    """
    rng = np.random.default_rng(seed)
    bandwidth = median_bandwidth(ref_z)
    rows = []
    for name, (pool_p, pool_z) in pools.items():
        for size in sizes:
            if size > len(pool_p):
                continue
            alarms = {"ks_outputs": 0, "mmd_features": 0, "c2st_features": 0}
            for rep in range(n_reps):
                idx = rng.choice(len(pool_p), size=size, replace=False)
                alarms["ks_outputs"] += ks_test(ref_p, pool_p[idx]) < alpha
                alarms["mmd_features"] += mmd_test(ref_z, pool_z[idx], bandwidth, n_perm, seed + rep, device) < alpha
                alarms["c2st_features"] += c2st_test(ref_z, pool_z[idx], seed + rep) < alpha
            for detector, count in alarms.items():
                rows.append({"pool": name, "batch_size": size, "detector": detector,
                             "detection_rate": count / n_reps, "n_reps": n_reps})
    return pd.DataFrame(rows)


def mixture_batch(in_group: np.ndarray, share: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Indices of one batch of n images in which a subgroup is over-represented.

    round(share * n) images come from the subgroup; the rest from the whole pool (which contains the
    subgroup at its natural share), all without replacement. share = 0 is an unshifted batch.
    """
    n_group = int(round(share * n))
    group = rng.choice(np.flatnonzero(in_group), size=n_group, replace=False)
    left = np.setdiff1d(np.arange(len(in_group)), group)
    return np.concatenate([group, rng.choice(left, size=n - n_group, replace=False)])
