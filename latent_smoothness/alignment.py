"""Alignment between latent displacements and the classifier's decision-margin gradient."""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from latent_smoothness.utils import decision_margin, eval_mode


@dataclass
class AlignmentResult:
    cosine_similarities: Tensor
    mean_alignment: float
    fraction_aligned: float


def _compute_gradient_alignment_raw(
    classifier: nn.Module,
    mu_before: Tensor,
    mu_after: Tensor,
    class_idx: int = 0,
    eval_point: str = "mu_before",
) -> AlignmentResult:
    N, dz = mu_before.shape
    delta_mu = mu_after - mu_before  # (N, dz)

    if eval_point == "mu_before":
        z_eval = mu_before
    elif eval_point == "mu_after":
        z_eval = mu_after
    elif eval_point == "midpoint":
        z_eval = (mu_before + mu_after) / 2.0
    else:
        raise ValueError(f"Unknown eval_point: {eval_point!r}")

    z_eval = z_eval.detach().clone().requires_grad_(True)
    f = decision_margin(classifier(z_eval), class_idx)  # (N,)
    grad = torch.autograd.grad(f.sum(), z_eval, allow_unused=True)[0]  # (N, dz)
    if grad is None:  # non-differentiable map (e.g. vq/scramble): zero gradient
        grad = torch.zeros_like(z_eval)

    # Cosine similarity per sample
    delta_norm = delta_mu.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    grad_norm = grad.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    cos_sim = (delta_mu / delta_norm * grad / grad_norm).sum(dim=-1)  # (N,)

    return AlignmentResult(
        cosine_similarities=cos_sim.detach(),
        mean_alignment=float(cos_sim.mean().detach()),
        fraction_aligned=float((cos_sim > 0).float().mean().detach()),
    )


def compute_gradient_alignment(
    classifier: nn.Module,
    mu_before: Tensor,
    mu_after: Tensor,
    class_idx: int = 0,
    eval_point: str = "mu_before",
) -> AlignmentResult:
    """Cosine similarity between ``mu_after - mu_before`` and the gradient of the
    decision margin of ``class_idx`` (see :func:`decision_margin`).

    ``eval_point`` selects where the gradient is taken: ``"mu_before"``,
    ``"mu_after"`` or ``"midpoint"``. ``fraction_aligned`` is the share of
    displacements with positive cosine, i.e. that increase the margin to first order.
    """
    with eval_mode(classifier):
        return _compute_gradient_alignment_raw(classifier, mu_before, mu_after, class_idx, eval_point)
