"""Predictions of the trained ensemble at every site.

    python scripts/predict.py --device cuda

Writes results/preds/<site>.npz with image_id, split, p_seeds (one column per seed), p (ensemble mean)
and z (seed-0 features, 2048-d). EyePACS training images are included only for their features: the shift
tests reduce features with a PCA fitted on them (never evaluated).
"""
import argparse

import numpy as np
import torch

from retina_shift.dataset import load_images
from retina_shift.model import make_model
from retina_shift.predict import predict_site
from retina_shift.sites import SITES
from retina_shift.splits import ROOT, load_meta

CKPT_DIR = ROOT / "checkpoints"
PRED_DIR = ROOT / "results" / "preds"
SEEDS = [0, 1, 2, 3, 4]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    models = []
    for seed in SEEDS:
        model = make_model(pretrained=False)
        model.load_state_dict(torch.load(CKPT_DIR / f"seed{seed}.pt", map_location="cpu"))
        models.append(model.eval())
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    for site in SITES:
        meta = load_meta(site)
        p_seeds, z = predict_site(models, load_images(site, meta["image_id"]), device=args.device)
        np.savez_compressed(PRED_DIR / f"{site}.npz", image_id=meta["image_id"].to_numpy(dtype=str),
                            split=meta["split"].to_numpy(dtype=str), p_seeds=p_seeds, p=p_seeds.mean(axis=1), z=z)
        print(f"{site}: {len(meta)} images predicted")


if __name__ == "__main__":
    main()
