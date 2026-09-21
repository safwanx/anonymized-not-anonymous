#!/usr/bin/env python3
"""Focused Privacy Beyond Pixels (PBP) AAM trainer for NTU120-Privacy-21K.

PBP's own multitask_train_fa.py unconditionally loads THUMOS + UCF-Crime even
when their loss weights are 0, so it cannot run NTU-only. This script reproduces
the exact AAM objective PBP applies on the action-recognition dataset, dropping
the TAD/AD co-training tasks:

    loss = fa_w   * MSE(x, AAM(x))                  # reconstruction (keep utility)
         + ar_w   * CE(head(AAM(x)), action_label)  # action utility
         - fb_w   * NTXent(AAM(f_i), AAM(f_j))       # SSL privacy (push apart 2 frames)

It uses PBP's own AAM module (models/anonymizer.py) and PBP's NTXentLoss, so the
trained anonymizer is faithful to the paper. The action head is a plain
Linear(768, num_classes), identical to PBP's load_ft 'maev2' path.

Inputs are the HDF5 files written by export_protocol_features_to_pbp.py.
The fb (frame-bank) file is required for the privacy term; without it, pass
--fb-weight 0 to train a recon+utility-only ablation.

GPU recommended but small enough to run on CPU (AAM is a 2-layer MLP / shallow
transformer over 768-d vectors).

Run:
    python pbp_integration/train_pbp_aam_ntu.py \
        --train-split train_original_r1_allcams --epochs 100
"""

from __future__ import annotations

import argparse
import json
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pbp_paths as P


def _device(prefer_cpu: bool) -> torch.device:
    if prefer_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


class NTUFeatureSplit(Dataset):
    """Serves (clip_feature, action_label, fb_pair) from our exported HDF5.

    fb_pair is [2, 768]: two distinct frames of the same video for the SSL
    privacy loss. If the fb file is absent, fb_pair is a zero tensor and the
    caller must use --fb-weight 0.
    """

    def __init__(self, split: str, use_fb: bool):
        self.clip = h5py.File(P.feat_h5(split, fb=False), "r")
        labels = json.loads((P.PBP_LABELS_DIR / f"labels_ntu_{split}.json").read_text())
        self.keys = [k for k in self.clip.keys() if k in labels]
        self.labels = labels
        self.use_fb = use_fb
        self.fb = h5py.File(P.feat_h5(split, fb=True), "r") if use_fb else None

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, i: int):
        key = self.keys[i]
        clip = torch.from_numpy(self.clip[key][...]).float()
        clip = clip[np.random.randint(clip.shape[0])] if clip.shape[0] > 1 else clip.squeeze(0)
        label = int(self.labels[key]["action_label"])
        if self.use_fb and key in self.fb:
            bank = torch.from_numpy(self.fb[key][...]).float()
            j, k = np.random.choice(bank.shape[0], size=2, replace=bank.shape[0] < 2)
            fb_pair = torch.stack([bank[j], bank[k]])
        else:
            fb_pair = torch.zeros(2, clip.shape[-1])
        return clip, label, fb_pair


def build_aam(arch: str) -> nn.Module:
    MLP, Transformer = P.import_anonymizer()
    if arch == "mlp":
        return MLP(P.FEATURE_DIM, P.FEATURE_DIM)
    if arch == "transformer":
        return Transformer(P.FEATURE_DIM, num_heads=8, num_layers=3)
    raise ValueError(f"Unknown AAM arch: {arch}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-split", default="train_original_r1_allcams")
    ap.add_argument("--arch", choices=["mlp", "transformer"], default="mlp",
                    help="mlp = per-vector AAM (matches our single-clip features); "
                         "transformer = PBP default (expects a sequence).")
    ap.add_argument("--num-classes", type=int, default=120)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--fa-weight", type=float, default=100.0, help="reconstruction (PBP default)")
    ap.add_argument("--ar-weight", type=float, default=1.0, help="action utility (PBP default)")
    ap.add_argument("--fb-weight", type=float, default=1.0, help="SSL privacy (PBP default); 0 disables")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--run-id", default="pbp_aam_ntu")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dev = _device(args.cpu)
    use_fb = args.fb_weight > 0
    print(f"device={dev} fb={'on' if use_fb else 'off'} arch={args.arch}")

    ds = NTUFeatureSplit(args.train_split, use_fb=use_fb)
    if len(ds) == 0:
        raise SystemExit(f"No features for split {args.train_split}. Run the exporter first.")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )

    aam = build_aam(args.arch).to(dev)
    head = nn.Linear(P.FEATURE_DIM, args.num_classes).to(dev)
    opt = torch.optim.AdamW(list(aam.parameters()) + list(head.parameters()), lr=args.lr)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()

    # PBP's NTXent (imported lazily; lives in the repo root).
    import sys
    if str(P.PBP_REPO) not in sys.path:
        sys.path.insert(0, str(P.PBP_REPO))
    from nt_xent_original import NTXentLoss  # type: ignore
    crit_fb = (
        NTXentLoss(
            device=str(dev),
            batch_size=args.batch_size,
            temperature=args.temperature,
            use_cosine_similarity=True,
        )
        if use_fb
        else None
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        aam.train(); head.train()
        agg = {"fa": [], "ar": [], "fb": []}
        for clip, label, fb_pair in loader:
            clip, label = clip.to(dev), label.to(dev)
            opt.zero_grad()

            anon = aam(clip)
            loss_fa = mse(clip, anon)
            loss_ar = ce(head(anon), label)

            if use_fb:
                fb_pair = fb_pair.to(dev)                      # [B,2,768]
                z0, z1 = aam(fb_pair[:, 0]), aam(fb_pair[:, 1])
                assert crit_fb is not None
                loss_fb = crit_fb(z0, z1)
            else:
                loss_fb = torch.zeros((), device=dev)

            loss = args.fa_weight * loss_fa + args.ar_weight * loss_ar - args.fb_weight * loss_fb
            loss.backward()
            opt.step()
            agg["fa"].append(loss_fa.item()); agg["ar"].append(loss_ar.item()); agg["fb"].append(loss_fb.detach().item())

        epoch_row = {"epoch": epoch, **{key: float(np.mean(values)) for key, values in agg.items()}}
        if not all(np.isfinite(value) for key, value in epoch_row.items() if key != "epoch"):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}: {epoch_row}")
        history.append(epoch_row)
        print(f"epoch {epoch:3d}  fa={epoch_row['fa']:.4f}  "
              f"ar={epoch_row['ar']:.4f}  fb={epoch_row['fb']:.4f}", flush=True)

    out = P.PBP_MODELS_DIR / f"{args.run_id}.pth"
    torch.save({"fa_model_state_dict": aam.state_dict(),
                "head_state_dict": head.state_dict(),
                "arch": args.arch, "args": vars(args), "history": history}, out)
    history_out = P.PBP_RESULTS_DIR / f"{args.run_id}_training_history.json"
    history_out.write_text(json.dumps(history, indent=2))
    print(f"saved AAM -> {out}")
    print(f"saved history -> {history_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
