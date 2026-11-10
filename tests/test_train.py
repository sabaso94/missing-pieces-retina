import numpy as np
import torch

from retina_shift.dataset import gpu_augment, normalise
from retina_shift.metrics import auroc
from retina_shift.model import make_model
from retina_shift.predict import predict_probs, predict_site
from retina_shift.train import TrainConfig, train_model


def _synthetic(n=400, size=64, seed=0):
    """Dark images; positives carry a bright square somewhere in the image."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 60, size=(n, 3, size, size), dtype=np.uint8)
    y = (rng.random(n) < 0.3).astype(int)
    for i in np.flatnonzero(y):
        r, c = rng.integers(8, size - 24, size=2)
        x[i, :, r : r + 14, c : c + 14] = 230
    return torch.from_numpy(x), y


def _config(seed=0):
    return TrainConfig(seed=seed, device="cpu", epochs=3, batch_size=32, lr=1e-3, patience=3, pretrained=False)


def test_training_learns_a_visible_lesion():
    x, y = _synthetic()
    model, history = train_model(_config(), x[:300], y[:300], x[300:], y[300:])
    assert history["val_auroc"].max() > 0.9
    p = predict_probs(model, x[300:], device="cpu")
    assert auroc(y[300:], p) == history["val_auroc"].max()


def test_training_is_reproducible():
    x, y = _synthetic(n=200)
    _, h1 = train_model(_config(seed=1), x[:150], y[:150], x[150:], y[150:])
    _, h2 = train_model(_config(seed=1), x[:150], y[:150], x[150:], y[150:])
    assert h1.equals(h2)


def test_predict_site_returns_seed_probabilities_and_features():
    x, _ = _synthetic(n=10)
    models = [make_model(pretrained=False).eval() for _ in range(2)]
    p_seeds, z = predict_site(models, x, device="cpu", batch_size=4)
    assert p_seeds.shape == (10, 2) and z.shape == (10, 2048) and z.dtype == np.float32
    assert ((p_seeds > 0) & (p_seeds < 1)).all()


def test_augmentation_keeps_shape_and_is_seeded():
    x, _ = _synthetic(n=4)
    a = gpu_augment(x, torch.Generator().manual_seed(0))
    b = gpu_augment(x, torch.Generator().manual_seed(0))
    assert a.shape == (4, 3, 64, 64) and a.dtype == torch.float32 and torch.equal(a, b)
    assert normalise(x).shape == a.shape
