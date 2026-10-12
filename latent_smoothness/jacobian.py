from dataclasses import dataclass

import warnings

import torch
import torch.nn as nn
from torch import Tensor

from latent_smoothness.utils import eval_mode


@dataclass
class JacobianResult:
    singular_values: Tensor
    mean_spectrum: Tensor
    spectral_variation: Tensor
    lipschitz_estimate: float
    gradient_degenerate: bool = False  # Jacobian ~0 everywhere: gradient lens carries no signal
    n_zero_fill: int = 0               # points with no gradient path (zero-filled)


def _has_train_batchnorm(module: nn.Module) -> bool:
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.training:
            return True
    return False


def _compute_jacobian_raw(
    decoder: nn.Module,
    z_points: Tensor,
) -> JacobianResult:
    """Per-sample Jacobian computation compatible with all module types.

    For modules in train mode with BatchNorm, uses a 2-sample batch to satisfy
    BN's requirement of batch size > 1 while computing the Jacobian for each point.
    """
    N, dz = z_points.shape
    all_svs = []
    n_zero_fill = 0

    needs_batch_trick = _has_train_batchnorm(decoder)
    # Use last sample as fixed reference when padding to batch size 2
    z_ref = z_points[-1:].detach()

    for i in range(N):
        # Keep z as 1D (dz,) so grads have shape (dx, dz) not (dx, 1, dz)
        z = z_points[i].detach().clone().requires_grad_(True)  # (dz,)

        if needs_batch_trick:
            batch_input = torch.cat([z.unsqueeze(0), z_ref], dim=0)  # (2, dz)
            out = decoder(batch_input)
            out_i = out[0].reshape(-1)
        else:
            out_i = decoder(z.unsqueeze(0)).reshape(-1)

        dx = out_i.shape[0]
        eye = torch.eye(dx, device=out_i.device, dtype=out_i.dtype)
        J = torch.autograd.grad(
            out_i, z,
            grad_outputs=eye,
            is_grads_batched=True,
            allow_unused=True,
        )[0]  # (dx, dz), or None when z is unused (non-differentiable
              # maps like the vq/scramble warps snap to a codebook -> flat cells)
        if J is None:
            J = torch.zeros(dx, dz, device=out_i.device, dtype=out_i.dtype)
            n_zero_fill += 1

        svs = torch.linalg.svdvals(J.unsqueeze(0))  # (1, min(dx,dz))
        all_svs.append(svs)

    singular_values = torch.cat(all_svs, dim=0)  # (N, min(dx,dz))
    mean_spectrum = singular_values.mean(dim=0)
    spectral_variation = singular_values.std(dim=0)
    lipschitz_estimate = float(singular_values.detach().max())

    # Degenerate = the Jacobian field is ~0, so the gradient-based lens carries
    # no signal. Covers BOTH the piecewise-constant case (no gradient path ->
    # zero-filled) and the collapsed/constant case (a real but ~0 Jacobian).
    # NOTE: a 0 here is NOT a smoothness signal; use output-based metrics
    # (PPL, secant_roughness, decision-region count) for such maps.
    gradient_degenerate = bool(lipschitz_estimate < 1e-8 or n_zero_fill == N)
    if gradient_degenerate:
        warnings.warn(
            f"compute_jacobian_spectrum: Jacobian is degenerate "
            f"(lipschitz={lipschitz_estimate:.2e}, zero_filled={n_zero_fill}/{N}). "
            f"This is a non-differentiable/collapsed map; the ~0 is NOT a "
            f"smoothness signal. Read output-based metrics instead."
        )

    return JacobianResult(
        singular_values=singular_values,
        mean_spectrum=mean_spectrum,
        spectral_variation=spectral_variation,
        lipschitz_estimate=lipschitz_estimate,
        gradient_degenerate=gradient_degenerate,
        n_zero_fill=n_zero_fill,
    )


def compute_jacobian_spectrum(
    decoder: nn.Module,
    z_points: Tensor,
) -> JacobianResult:
    """Singular values of the decoder Jacobian at each of the ``(N, dz)`` points."""
    with eval_mode(decoder):
        return _compute_jacobian_raw(decoder, z_points)
