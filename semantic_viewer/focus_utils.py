#
# Copyright (C) 2026, GS4City
# All rights reserved.
#

import numpy as np
import torch

torch.backends.cudnn.enabled = False


def build_color_map(n_cls: int):
    rng = np.random.RandomState(0)
    colors = rng.rand(n_cls, 3).astype(np.float32)
    if n_cls > 0:
        colors[0] = 0.0
    return colors


def estimate_focus_from_gaussians(gaussians, top_ratio: float = 0.05):
    try:
        xyz = gaussians._xyz.detach().cpu()
        op = gaussians._opacity.detach().view(-1).cpu()

        N = xyz.shape[0]
        if N == 0:
            raise RuntimeError("No gaussians in model.")

        k = max(int(N * top_ratio), 1000)
        k = min(k, N)
        _, topk_idx = torch.topk(op, k)
        xyz_focus = xyz[topk_idx]

        center = xyz_focus.mean(0)
        dist = torch.norm(xyz_focus - center, dim=1)
        radius = dist.quantile(0.9).item()
        radius = max(radius * 2.0, 0.1)

        return center.numpy().astype(np.float32), float(radius)
    except Exception as e:
        print(f"[GUI] estimate_focus_from_gaussians filed, back to default: {e}")
        return np.zeros(3, dtype=np.float32), 2.0
