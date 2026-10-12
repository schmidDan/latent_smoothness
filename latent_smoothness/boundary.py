"""Decision-boundary geometry of a latent classifier.

All quantities are computed on the decision margin of ``class_idx``,
``m(z) = l_c(z) - max_{j != c} l_j(z)`` (see :func:`decision_margin`), whose zero
level set is the classifier's decision boundary for that class. A single-output
classifier is treated as binary and its logit is used directly.
"""
import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from latent_smoothness.utils import decision_margin, eval_mode


@dataclass
class BoundaryPointResult:
    margins: Tensor          # (N,) decision margin m(z); > 0 iff class_idx is predicted
    gradient_norms: Tensor   # (N,) ||∇m(z)||
    curvature: Tensor        # (N,) ||H_m(z)||_F^2 / dz^2


@dataclass
class BoundaryTrajectoryResult:
    n_crossings: int                 # total decision-boundary crossings over all paths
    sign_change_locations: Tensor    # (M, n_steps) bool, True at the step after a crossing


def _compute_boundary_geometry_raw(
    classifier: nn.Module,
    mu_points: Tensor,
    class_idx: int = 0,
) -> BoundaryPointResult:
    N, dz = mu_points.shape

    margins = []
    gradient_norms = []
    curvatures = []
    n_hessian_fail = 0

    def scalar_fn(z_vec):
        return decision_margin(classifier(z_vec.unsqueeze(0)), class_idx)[0]

    for i in range(N):
        z = mu_points[i:i+1].detach().clone().requires_grad_(True)
        f = decision_margin(classifier(z), class_idx)[0]

        # Gradient
        grad = torch.autograd.grad(f, z, create_graph=False, allow_unused=True)[0]  # (1, dz)
        if grad is None:  # non-differentiable map: zero gradient
            grad = torch.zeros_like(z)
        grad_flat = grad.squeeze(0)  # (dz,)

        # Hessian via torch.func.hessian (forward-over-reverse AD). The forward
        # pass has no rule for cdist, so it raises on non-differentiable maps
        # such as the vq/scramble warps. Those are piecewise-constant -> ~zero
        # curvature, so fall back to 0 rather than crash the whole suite.
        try:
            H = torch.func.hessian(scalar_fn)(mu_points[i].detach())  # (dz, dz)
            curv = (H ** 2).sum() / (dz ** 2)
        except NotImplementedError:  # forward-AD unsupported (e.g. cdist); real
            n_hessian_fail += 1       # RuntimeErrors must surface, not be hidden
            curv = torch.zeros((), device=mu_points.device, dtype=mu_points.dtype)
        except RuntimeError as e:
            # Some maps break functorch tracing in ways other than
            # NotImplementedError: piecewise_affine converts to numpy internally
            # (per-cell permutation hashing) -> "Cannot access data pointer of
            # Tensor that doesn't have storage" under torch.func. Fall back to 0
            # for this tracing incompatibility only; re-raise anything else so
            # genuine errors still surface.
            msg = str(e).lower()
            if "data pointer" in msg or "storage" in msg:
                n_hessian_fail += 1
                curv = torch.zeros((), device=mu_points.device, dtype=mu_points.dtype)
            else:
                raise

        margins.append(f.detach())
        gradient_norms.append(grad_flat.norm().detach())
        curvatures.append(curv.detach())

    if n_hessian_fail:
        warnings.warn(
            f"compute_boundary_geometry: Hessian unavailable at {n_hessian_fail}/{N} "
            f"points (map not traceable by torch.func; curvature set to 0)."
        )

    return BoundaryPointResult(
        margins=torch.stack(margins),
        gradient_norms=torch.stack(gradient_norms),
        curvature=torch.stack(curvatures),
    )


def compute_boundary_geometry(
    classifier: nn.Module,
    mu_points: Tensor,
    class_idx: int = 0,
) -> BoundaryPointResult:
    """Margin, margin-gradient norm and Hessian curvature at each latent point."""
    with eval_mode(classifier):
        return _compute_boundary_geometry_raw(classifier, mu_points, class_idx)


def _compute_boundary_trajectory_raw(
    classifier: nn.Module,
    trajectories: Tensor,
    class_idx: int = 0,
) -> BoundaryTrajectoryResult:
    M, n_steps, dz = trajectories.shape

    with torch.no_grad():
        flat = trajectories.reshape(M * n_steps, dz)
        margins = decision_margin(classifier(flat), class_idx).reshape(M, n_steps)

    # A crossing is a change of "is class_idx predicted" between consecutive steps.
    positive = margins > 0
    sign_changes = torch.zeros(M, n_steps, dtype=torch.bool, device=trajectories.device)
    sign_changes[:, 1:] = positive[:, 1:] != positive[:, :-1]

    return BoundaryTrajectoryResult(
        n_crossings=int(sign_changes.sum().item()),
        sign_change_locations=sign_changes,
    )


def compute_boundary_trajectory(
    classifier: nn.Module,
    trajectories: Tensor,
    class_idx: int = 0,
) -> BoundaryTrajectoryResult:
    """Count decision-boundary crossings of ``class_idx`` along ``(M, n_steps, dz)`` paths."""
    with eval_mode(classifier):
        return _compute_boundary_trajectory_raw(classifier, trajectories, class_idx)
