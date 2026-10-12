"""Secant (finite-difference) roughness over a step-size sweep.

This is a GRADIENT-FREE complement to the Jacobian metrics. Instead of the
tangent map df/dz at a point (which is 0/undefined for non-differentiable maps
like vq/scramble), it measures the secant ratio

        r(z, Δ) = ‖f(z + Δ) − f(z)‖ / ‖Δ‖

along each trajectory, at several step scales Δ. The *shape* of r vs scale is
diagnostic of the kind of (non-)smoothness:

  * smooth map:           r(scale) ≈ const  → loglog slope ≈ 0
                          (and small-scale r recovers the Jacobian norm)
  * jump discontinuity:   r(scale) ↑ as scale→0  → loglog slope ≈ −1
  * quantized/scrambled:  r ≈ 0 at tiny scale (steps stay inside a codebook
                          cell), then rises near the cell scale → positive slope

Defined and finite for EVERY variant, including the non-differentiable controls.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from latent_smoothness.utils import eval_mode


@dataclass
class SecantRoughnessResult:
    scales: Tensor          # (S,) step length as a fraction of the chord
    mean_roughness: Tensor  # (S,) mean secant ratio at each scale
    max_roughness: Tensor   # (S,) max secant ratio at each scale (secant Lipschitz)
    loglog_slope: float     # OLS slope of log(mean_roughness) vs log(scale)


def _compute_secant_roughness_raw(
    decoder: nn.Module,
    z_start: Tensor,
    z_end: Tensor,
    n_steps: int = 32,
) -> SecantRoughnessResult:
    z_start, z_end = z_start.detach(), z_end.detach()
    M = z_start.shape[0]
    d = z_end - z_start                                  # (M, dz)
    chord = torch.linalg.norm(d, dim=-1).clamp_min(1e-12)  # (M,)
    ts = torch.linspace(0.0, 1.0, n_steps + 1, device=z_start.device, dtype=z_start.dtype)

    # Decode the whole path once (gradient-free); reuse for every scale.
    frames = []
    with torch.no_grad():
        for k in range(n_steps + 1):
            z_k = z_start + ts[k] * d                    # (M, dz)
            frames.append(decoder(z_k).reshape(M, -1))   # (M, dx)
    X = torch.stack(frames, dim=0)                       # (K+1, M, dx)

    strides = []
    s = 1
    while s <= n_steps:
        strides.append(s)
        s *= 2

    scales, means, maxes = [], [], []
    for stride in strides:
        diff = X[stride:] - X[:-stride]                  # (K+1-stride, M, dx)
        num = torch.linalg.norm(diff, dim=-1)            # (K+1-stride, M)
        step_len = (stride / n_steps) * chord            # (M,)
        ratio = num / step_len.unsqueeze(0)              # (K+1-stride, M)
        scales.append(stride / n_steps)
        means.append(float(ratio.mean()))
        maxes.append(float(ratio.max()))

    scales_t = torch.tensor(scales)
    means_t = torch.tensor(means)
    maxes_t = torch.tensor(maxes)

    lx = torch.log(scales_t.clamp_min(1e-12))
    ly = torch.log(means_t.clamp_min(1e-12))
    denom = ((lx - lx.mean()) ** 2).sum().clamp_min(1e-12)
    slope = float(((lx - lx.mean()) * (ly - ly.mean())).sum() / denom)

    return SecantRoughnessResult(
        scales=scales_t,
        mean_roughness=means_t,
        max_roughness=maxes_t,
        loglog_slope=slope,
    )


def compute_secant_roughness(
    decoder: nn.Module,
    z_start: Tensor,
    z_end: Tensor,
    n_steps: int = 32,
) -> SecantRoughnessResult:
    """Gradient-free secant-roughness sweep along z_start -> z_end trajectories."""
    with eval_mode(decoder):
        return _compute_secant_roughness_raw(decoder, z_start, z_end, n_steps)
