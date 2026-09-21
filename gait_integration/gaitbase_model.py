"""Standalone GaitBase (OpenGait Baseline) for inference.

OpenGait's Baseline rides on a BaseModel framework (distributed training,
message manager, optimizer). For a pure feature extractor we build the same
network blocks directly and replicate Baseline.forward, then load the matching
weights (Backbone.* / FCs.* / BNNecks.*) from the checkpoint.

Verified: strict load of GaitBase_DA-60000.pt, forward on a [N,S,64,44]
silhouette sequence -> embedding [N, 256, 16] (flatten to 4096-d for retrieval).
"""

from __future__ import annotations

import sys
import types
import importlib.machinery as _mach

import numpy as np
import torch
import torch.nn as nn

import gait_paths as G


def _import_opengait_blocks():
    """Import the GaitBase building blocks, robust to broken optional deps.

    On a clean env (SLURM) the imports just work. On machines with a broken
    tensorflow/jax/tensorboard stack, importing opengait transitively explodes;
    we stub those optional modules (GaitBase needs none of them) and retry.
    """
    if str(G.OPENGAIT_PKG) not in sys.path:
        sys.path.insert(0, str(G.OPENGAIT_PKG))

    def _do_import():
        from modeling.modules import (  # type: ignore
            SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper,
            SeparateFCs, SeparateBNNecks,
        )
        from modeling.backbones.resnet import ResNet9  # type: ignore
        return (SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper,
                SeparateFCs, SeparateBNNecks, ResNet9)

    try:
        return _do_import()
    except Exception:
        for name in ("tensorflow", "jax", "jaxlib"):
            mod = types.ModuleType(name)
            mod.__spec__ = _mach.ModuleSpec(name, None)
            sys.modules.setdefault(name, mod)
        tb = types.ModuleType("torch.utils.tensorboard")
        tb.__spec__ = _mach.ModuleSpec("torch.utils.tensorboard", None)
        tb.SummaryWriter = object
        sys.modules.setdefault("torch.utils.tensorboard", tb)
        return _do_import()


class GaitBase(nn.Module):
    """Inference-only GaitBase matching configs/gaitbase/gaitbase_da_casiab.yaml."""

    def __init__(self, checkpoint: str | None = None):
        super().__init__()
        (SetBlockWrapper, HPP, PackSeq, SeparateFCs, SeparateBNNecks, ResNet9) = _import_opengait_blocks()
        self.Backbone = SetBlockWrapper(
            ResNet9(block="BasicBlock", channels=[64, 128, 256, 512],
                    layers=[1, 1, 1, 1], strides=[1, 2, 2, 1], maxpool=False)
        )
        self.FCs = SeparateFCs(in_channels=512, out_channels=256, parts_num=16)
        self.BNNecks = SeparateBNNecks(class_num=74, in_channels=256, parts_num=16)
        self.TP = PackSeq(torch.max)
        self.HPP = HPP(bin_num=[16])
        if checkpoint:
            self.load_weights(checkpoint)
        self.eval()

    def load_weights(self, checkpoint: str) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        sd = state.get("model", state)
        for module, prefix in [(self.Backbone, "Backbone."), (self.FCs, "FCs."), (self.BNNecks, "BNNecks.")]:
            sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            module.load_state_dict(sub, strict=True)

    @torch.no_grad()
    def embed(self, sil_seq: np.ndarray, device: torch.device) -> np.ndarray:
        """sil_seq: [S, 64, 44] float in [0,1] -> 4096-d L2-normalized embedding."""
        x = torch.from_numpy(sil_seq).float().to(device)        # [S,H,W]
        x = x.unsqueeze(0).unsqueeze(0)                          # [N=1, C=1, S, H, W]
        outs = self.Backbone(x)                                  # [N,C,S,H,W]
        outs = self.TP(outs, None, options={"dim": 2})[0]        # [N,C,H,W]
        feat = self.HPP(outs)                                    # [N,C,P]
        embed_1 = self.FCs(feat)                                 # [N,C,P]
        vec = embed_1.flatten(1).cpu().numpy()[0]               # [C*P]
        norm = np.linalg.norm(vec) + 1e-8
        return (vec / norm).astype(np.float32)


def load_gaitbase(device: torch.device) -> GaitBase:
    ckpt = str(G.find_gaitbase_checkpoint())
    print(f"Loading GaitBase from {ckpt}")
    return GaitBase(ckpt).to(device)
