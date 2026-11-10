"""Train the EyePACS ensemble members.

    python scripts/train.py --seeds 0 1 2 --device cuda

One model per seed, saved to checkpoints/seed<k>.pt with its training history. Seeds whose checkpoint
already exists are skipped, so the five members can be trained in parallel on two GPUs.
"""
import argparse

import torch

from retina_shift.dataset import load_images
from retina_shift.splits import ROOT, load_meta
from retina_shift.train import TrainConfig, train_model

CKPT_DIR = ROOT / "checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    todo = [s for s in args.seeds if not (CKPT_DIR / f"seed{s}.pt").exists()]
    if not todo:
        print("all requested checkpoints exist")
        return
    meta = load_meta("eyepacs")
    train, val = meta[meta["split"] == "train"], meta[meta["split"] == "val"]
    x_train, x_val = load_images("eyepacs", train["image_id"]), load_images("eyepacs", val["image_id"])
    print(f"loaded {len(train)} train and {len(val)} validation images")
    CKPT_DIR.mkdir(exist_ok=True)
    for seed in todo:
        model, history = train_model(TrainConfig(seed=seed, device=args.device), x_train,
                                     train["referable"].to_numpy(), x_val, val["referable"].to_numpy())
        torch.save(model.state_dict(), CKPT_DIR / f"seed{seed}.pt")
        history.to_csv(CKPT_DIR / f"seed{seed}_history.csv", index=False)
        print(f"seed {seed}: best validation AUROC {history['val_auroc'].max():.4f} "
              f"after {len(history)} epochs")


if __name__ == "__main__":
    main()
