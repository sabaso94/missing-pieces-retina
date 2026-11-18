"""Training one ensemble member: AdamW, cosine learning rate, mixed precision, early stopping on validation
AUROC. Seeded and with deterministic cuDNN, so a seed fixes the result."""

import copy
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from retina_shift.dataset import gpu_augment
from retina_shift.metrics import auroc
from retina_shift.model import make_model
from retina_shift.predict import predict_probs

# Deterministic cuBLAS matrix products (must be set before the first CUDA call).
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "cuda"
    epochs: int = 15
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 4
    pretrained: bool = True


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(cfg: TrainConfig, x_train: torch.Tensor, y_train: np.ndarray, x_val: torch.Tensor, y_val: np.ndarray):
    """Returns the model at the epoch with the best validation AUROC, and the per-epoch history."""
    _seed_everything(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    model = make_model(cfg.pretrained).to(cfg.device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = int(np.ceil(len(x_train) / cfg.batch_size))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.epochs * steps_per_epoch)
    use_amp = cfg.device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    y_train_t = torch.from_numpy(np.asarray(y_train, dtype=np.float32))

    best_auc, best_state, since_best, history = -1.0, None, 0, []
    for epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(len(x_train), generator=generator)
        losses = []
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            x = gpu_augment(x_train[idx].to(cfg.device, non_blocking=True), generator)
            y = y_train_t[idx].to(cfg.device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = loss_fn(model(x).float(), y)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
        val_auc = auroc(np.asarray(y_val), predict_probs(model, x_val, device=cfg.device))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_auroc": val_auc})
        if val_auc > best_auc:
            best_auc, best_state, since_best = val_auc, copy.deepcopy(model.state_dict()), 0
        else:
            since_best += 1
            if since_best >= cfg.patience:
                break
    if best_state is None:
        raise RuntimeError("the validation score never improved: no checkpoint to restore")
    model.load_state_dict(best_state)
    return model.eval(), pd.DataFrame(history)
