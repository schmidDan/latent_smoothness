"""Class-collapse warp — a smooth but non-injective, class-merging map.

A *linear* map (secant slope ~= 0, LID barely changes) that flattens the
class-centroid arrangement along its top ``n_directions`` principal axes, so the
most-separated classes overlap. It is the counterpart to the scrambling warps:
perfectly smooth, yet it destroys the class structure. The collapsed axis cannot
be restored by ``backproject``, and a classifier retrained on the warped latents
cannot recover it either.

    W(z) = z - ((z - mu_bar) @ B.T) @ B

where ``B`` (``n_directions x dz``) are the top principal directions of the
centred class-mean cloud and ``mu_bar`` its centroid. Linear => differentiable;
non-injective => ``backproject`` is a best-effort identity.
"""
from pathlib import Path
from typing import Union

import torch
from torch import Tensor

from latent_smoothness.warp.base import Warp


def _fit_collapse_basis(fit_data: Tensor, labels: Tensor, n_directions: int):
    """Return ``(mu_bar (dz,), B (k, dz))`` — top-k PCA axes of the class means."""
    z = fit_data.detach().float()
    labels = labels.to(torch.long)
    means = [z[labels == c].mean(dim=0) for c in labels.unique().tolist()
             if int((labels == c).sum()) >= 1]
    M = torch.stack(means, dim=0)              # (Kq, dz)
    mu_bar = M.mean(dim=0)
    Mc = M - mu_bar
    k = min(int(n_directions), max(M.shape[0] - 1, 0))
    if k <= 0:
        return mu_bar, torch.zeros(0, z.shape[1])
    _, _, Vh = torch.linalg.svd(Mc, full_matrices=False)
    return mu_bar, Vh[:k]                       # (k, dz), orthonormal


class ClassCollapseWarp(Warp):
    """Project out the top principal axes of the class-mean cloud.

    Args:
        fit_data: ``(N, d)`` latents.
        labels: ``(N,)`` integer class label per latent (e.g. classifier argmax).
        n_directions: number of between-class axes to remove; capped at
            ``n_classes - 1``.
    """

    name = "class_collapse"
    is_invertible = False
    is_differentiable = True

    def __init__(self, fit_data: Tensor, labels: Tensor, n_directions: int = 1):
        mu_bar, B = _fit_collapse_basis(fit_data, labels, n_directions)
        self._mu_bar = mu_bar.detach().cpu()
        self._B = B.detach().cpu()             # (n_directions, dz)
        self.n_directions = int(B.shape[0])

    def forward(self, z: Tensor) -> Tensor:
        if self._B.shape[0] == 0:
            return z
        mu = self._mu_bar.to(device=z.device, dtype=z.dtype)
        B = self._B.to(device=z.device, dtype=z.dtype)
        centered = z - mu
        return z - (centered @ B.t()) @ B

    def backproject(self, z_w: Tensor) -> Tensor:
        """Collapsed coordinates are unrecoverable — best-effort identity."""
        return z_w

    def save(self, path: Union[str, Path]) -> None:
        torch.save(
            {"mu_bar": self._mu_bar, "B": self._B, "n_directions": self.n_directions},
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ClassCollapseWarp":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls.__new__(cls)
        obj._mu_bar = blob["mu_bar"]
        obj._B = blob["B"]
        obj.n_directions = int(blob["n_directions"])
        return obj
