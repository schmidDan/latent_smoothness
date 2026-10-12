"""Label coherence of a classifier along latent trajectories.

A LABEL-EXPLICIT complement to ``ppl`` / ``secant_roughness``: instead of
pixel/perceptual change of the DECODER output along z_start->z_end, it tracks the
CLASSIFIER's response along the interpolated path. Two complementary measures
are reported:

  * ``label_flips``    — how many times ``argmax`` changes along the path
                         (0 = coherent, many = the label thrashes).
  * ``monotonic_gain`` — net climb of P(target) toward the endpoint class,
                         in [-1, 1] (mean sign of the per-step change; +1 = climbs
                         every step, 0 = jumpy, -1 = falls).

The target class is the endpoint's predicted class (``argmax`` at z_end), so the
metric measures whether the PATH to the endpoint is coherent. Unlike gradient
``alignment`` it needs no gradients, so it is defined for non-differentiable
latent maps (e.g. the ``vq`` / ``scramble`` warps) too.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from latent_smoothness.utils import eval_mode, linear_interpolate


@dataclass
class LabelCoherenceResult:
    label_flips: Tensor        # (M,) int — # argmax changes along each path
    monotonic_gain: Tensor     # (M,) float in [-1, 1] — net climb of P(target)
    target_prob_gain: Tensor   # (M,) float — P_target(end) - P_target(start)
    mean_label_flips: float
    mean_monotonic_gain: float
    mean_target_prob_gain: float


def _compute_label_coherence_raw(
    classifier: nn.Module,
    z_start: Tensor,
    z_end: Tensor,
    n_steps: int = 64,
) -> LabelCoherenceResult:
    M, dz = z_start.shape
    z_path = linear_interpolate(z_start, z_end, n_steps)  # (M, n_steps, dz)

    with torch.no_grad():
        logits = classifier(z_path.reshape(M * n_steps, dz))  # (M*n_steps, C)
        logits = logits.reshape(M, n_steps, -1)
        labels = logits.argmax(dim=-1)                        # (M, n_steps)
        probs = F.softmax(logits, dim=-1)                     # (M, n_steps, C)

    # Realization 1 — label-flip count.
    label_flips = (labels[:, 1:] != labels[:, :-1]).sum(dim=1)  # (M,)

    # Realization 2 — monotonic gain of P(endpoint class) along the path.
    target = labels[:, -1]                                        # (M,)
    p_target = probs.gather(-1, target.view(M, 1, 1).expand(M, n_steps, 1)).squeeze(-1)  # (M, n_steps)
    d = p_target[:, 1:] - p_target[:, :-1]                        # (M, n_steps-1)
    monotonic_gain = torch.sign(d).mean(dim=1)                   # (M,) in [-1, 1]
    target_prob_gain = p_target[:, -1] - p_target[:, 0]           # (M,)

    return LabelCoherenceResult(
        label_flips=label_flips.detach().cpu(),
        monotonic_gain=monotonic_gain.detach().cpu(),
        target_prob_gain=target_prob_gain.detach().cpu(),
        mean_label_flips=float(label_flips.float().mean()),
        mean_monotonic_gain=float(monotonic_gain.mean()),
        mean_target_prob_gain=float(target_prob_gain.mean()),
    )


def compute_label_coherence(
    classifier: nn.Module,
    z_start: Tensor,
    z_end: Tensor,
    n_steps: int = 64,
) -> LabelCoherenceResult:
    """Label coherence of the classifier along z_start -> z_end trajectories."""
    with eval_mode(classifier):
        return _compute_label_coherence_raw(classifier, z_start, z_end, n_steps)
