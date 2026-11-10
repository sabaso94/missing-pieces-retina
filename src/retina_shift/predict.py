"""Running trained models over a set of images."""
import numpy as np
import torch

from retina_shift.dataset import normalise


@torch.no_grad()
def _forward(model, x: torch.Tensor, device: str, batch_size: int, with_features: bool):
    model.to(device).eval()
    probs, feats = [], []
    for start in range(0, len(x), batch_size):
        batch = normalise(x[start : start + batch_size].to(device, non_blocking=True))
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            z = model.features(batch)
            logits = model.head(z).squeeze(1)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        if with_features:
            feats.append(z.float().cpu().numpy())
    p = np.concatenate(probs).astype(np.float64)
    return (p, np.concatenate(feats).astype(np.float32)) if with_features else p


def predict_probs(model, x: torch.Tensor, device: str = "cuda", batch_size: int = 128) -> np.ndarray:
    """Predicted probability of referable retinopathy for each image."""
    return _forward(model, x, device, batch_size, with_features=False)


def predict_site(models, x: torch.Tensor, device: str = "cuda",
                 batch_size: int = 128) -> tuple[np.ndarray, np.ndarray]:
    """(N, n_models) probabilities, and the (N, 2048) features of the first model (seed 0)."""
    p0, z = _forward(models[0], x, device, batch_size, with_features=True)
    others = [predict_probs(m, x, device, batch_size) for m in models[1:]]
    return np.stack([p0, *others], axis=1), z
