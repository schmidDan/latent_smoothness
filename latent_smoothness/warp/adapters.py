"""Adapters over the functional warps + the ``build_warp`` factory.

Each adapter wraps one functional warp (``quantize.py`` / ``scramble.py`` /
``flow_warp.py``) behind the uniform :class:`Warp` interface.
``build_warp(name, fit_data, **kw)`` is the single entry point over all warps
(see :data:`WARP_NAMES`).

All adapters are device-preserving: they move their fitted tensors onto the
input's device on the fly and return on that device.
"""
from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor

from latent_smoothness.utils import set_global_seed
from latent_smoothness.warp.base import Warp
from latent_smoothness.warp.flow_warp import FlowWarp
from latent_smoothness.warp.piecewise_affine import PiecewiseAffineWarp
from latent_smoothness.warp.quantize import posthoc_vq
from latent_smoothness.warp.scramble import _fit_scramble, _fit_class_scramble
from latent_smoothness.warp.class_collapse import ClassCollapseWarp


def _nearest_index(z: Tensor, codebook: Tensor) -> Tensor:
    """Index of the nearest codebook row for each row of ``z`` (on ``z.device``)."""
    cb = codebook.to(device=z.device, dtype=torch.float32)
    return torch.cdist(z.to(torch.float32), cb).argmin(dim=1)


class VQWarp(Warp):
    """Vector-quantization warp: snap each latent to its nearest codebook entry.

    ``backproject`` is best-effort recovery (re-snap), NOT a true inverse — the
    quantization is lossy. ``posthoc_vq`` is seed-independent (``random_state=42``
    is hardcoded), so ``seed`` is accepted only for interface symmetry and does
    not change the codebook; vq is a single deterministic instance.

    Args:
        fit_data: ``(N, d)`` latents to cluster (needs ``N >= codebook_size``).
        codebook_size: number of codebook entries ``K``.
        seed: accepted for symmetry; has no effect (see above).
    """

    name = "vq"
    is_invertible = False
    is_differentiable = False

    def __init__(self, fit_data: Tensor, codebook_size: int, seed: int = 42):
        _, codebook = posthoc_vq(fit_data, codebook_size)
        self._codebook = codebook.detach().cpu()
        self.codebook_size = int(codebook_size)
        self.seed = int(seed)

    def forward(self, z: Tensor) -> Tensor:
        """Snap each latent to its nearest codebook entry."""
        cb = self._codebook.to(device=z.device, dtype=z.dtype)
        return cb[_nearest_index(z, self._codebook)]

    def backproject(self, z_w: Tensor) -> Tensor:
        """Best-effort recovery: read the snapped value (idempotent re-snap)."""
        cb = self._codebook.to(device=z_w.device, dtype=z_w.dtype)
        return cb[_nearest_index(z_w, self._codebook)]

    def save(self, path: Union[str, Path]) -> None:
        torch.save(
            {"codebook": self._codebook, "codebook_size": self.codebook_size, "seed": self.seed},
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "VQWarp":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls.__new__(cls)
        obj._codebook = blob["codebook"]
        obj.codebook_size = int(blob["codebook_size"])
        obj.seed = int(blob["seed"])
        return obj


class ScrambleWarp(Warp):
    """Bijective-scramble warp: quantize, then permute the codebook cells.

    Same construction as :func:`bijective_scramble`: ``forward`` is its encode
    (snap → permuted codebook entry), ``backproject`` its decode (snap →
    inverse-permuted entry).
    ``backproject`` exactly undoes the permutation, but the warp is reported
    ``is_invertible=False`` because it is preceded by lossy quantization (the
    original pre-snap latent is not recoverable). The codebook + permutation are
    persisted so a reloaded warp reproduces forward/backproject without the
    training data.

    Args:
        fit_data: ``(N, d)`` latents to cluster (needs ``N >= codebook_size``).
        codebook_size: number of codebook entries ``K``.
        seed: selects the cell permutation.
    """

    name = "scramble"
    is_invertible = False
    is_differentiable = False

    def __init__(self, fit_data: Tensor, codebook_size: int, seed: int = 42):
        _, codebook, perm = _fit_scramble(fit_data, codebook_size, seed)
        self._codebook = codebook.detach().cpu()
        self._perm = perm.detach().cpu()
        self.codebook_size = int(codebook_size)
        self.seed = int(seed)
        self._build_inverse_perm()

    def _build_inverse_perm(self) -> None:
        K = self._codebook.shape[0]
        inv = torch.zeros(K, dtype=torch.long)
        inv[self._perm] = torch.arange(K, dtype=torch.long)
        self._inv_perm = inv

    def forward(self, z: Tensor) -> Tensor:
        """Encode: snap to a cell, then map to the permuted cell's codebook entry."""
        cb = self._codebook.to(device=z.device, dtype=z.dtype)
        cell = _nearest_index(z, self._codebook)
        return cb[self._perm.to(z.device)[cell]]

    def backproject(self, z_w: Tensor) -> Tensor:
        """Decode: snap to a cell, then map back through the inverse permutation."""
        cb = self._codebook.to(device=z_w.device, dtype=z_w.dtype)
        cell = _nearest_index(z_w, self._codebook)
        return cb[self._inv_perm.to(z_w.device)[cell]]

    def save(self, path: Union[str, Path]) -> None:
        torch.save(
            {
                "codebook": self._codebook,
                "perm": self._perm,
                "codebook_size": self.codebook_size,
                "seed": self.seed,
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ScrambleWarp":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls.__new__(cls)
        obj._codebook = blob["codebook"]
        obj._perm = blob["perm"]
        obj.codebook_size = int(blob["codebook_size"])
        obj.seed = int(blob["seed"])
        obj._build_inverse_perm()
        return obj


class FlowWarpAdapter(Warp):
    """Adapter over :class:`FlowWarp` (the bijective normalizing-flow warp).

    ``forward``/``inverse`` delegate to the wrapped flow; ``backproject`` is the
    exact inverse (base default). ``save``/``load`` delegate to
    :meth:`FlowWarp.save`/:meth:`FlowWarp.load`. The flow is moved onto the
    input's device on every call so a CPU-loaded warp works on a CUDA input.
    """

    name = "flow_warp"
    is_invertible = True
    is_differentiable = True

    def __init__(self, warp: FlowWarp):
        self._warp = warp

    @classmethod
    def fit(
        cls,
        fit_data: Tensor,
        target: str = "mixture_of_gaussians",
        target_samples: Optional[Tensor] = None,
        n_epochs: int = 500,
        standardize: bool = False,
        seed: int = 0,
    ) -> "FlowWarpAdapter":
        """Fit a fresh :class:`FlowWarp` (seeded via ``set_global_seed``) and wrap it."""
        set_global_seed(seed)
        warp = FlowWarp()
        warp.fit(
            fit_data,
            target=target,
            target_samples=target_samples,
            n_epochs=n_epochs,
            standardize=standardize,
        )
        return cls(warp)

    def forward(self, z: Tensor) -> Tensor:
        self._warp.to(z.device)
        return self._warp.forward(z)

    def inverse(self, z_w: Tensor) -> Tensor:
        self._warp.to(z_w.device)
        return self._warp.inverse(z_w)

    def save(self, path: Union[str, Path]) -> None:
        self._warp.save(path)

    @classmethod
    def load(cls, path: Union[str, Path], map_location="cpu") -> "FlowWarpAdapter":
        return cls(FlowWarp.load(path, map_location=map_location))


class ClassScrambleWarp(Warp):
    """Class-constrained scramble: quantise, then permute cells by disjoint
    transpositions restricted to the **same** class (``class_mode="within"``, an
    aligned but rough warp) or **different** classes (``class_mode="across"``).

    The permutation is an involution (``inv_perm == perm``), so ``backproject``
    reuses ``perm`` and exactly undoes ``forward`` at codebook points (up to the
    quantisation). Needs per-point ``labels`` at fit time (e.g. classifier
    argmax).
    """

    name = "class_scramble"
    is_invertible = False
    is_differentiable = False

    def __init__(
        self,
        fit_data: Tensor,
        codebook_size: int,
        labels: Tensor,
        class_mode: str = "within",
        seed: int = 42,
    ):
        codebook, perm, cell_cls = _fit_class_scramble(
            fit_data, codebook_size, labels, class_mode, seed
        )
        self._codebook = codebook.detach().cpu()
        self._perm = perm.detach().cpu()          # involution: inv_perm == perm
        self._cell_cls = cell_cls.detach().cpu()
        self.codebook_size = int(codebook_size)
        self.class_mode = str(class_mode)
        self.seed = int(seed)

    def forward(self, z: Tensor) -> Tensor:
        cb = self._codebook.to(device=z.device, dtype=z.dtype)
        return cb[self._perm.to(z.device)[_nearest_index(z, self._codebook)]]

    def backproject(self, z_w: Tensor) -> Tensor:
        cb = self._codebook.to(device=z_w.device, dtype=z_w.dtype)
        return cb[self._perm.to(z_w.device)[_nearest_index(z_w, self._codebook)]]

    def save(self, path: Union[str, Path]) -> None:
        torch.save({
            "codebook": self._codebook, "perm": self._perm, "cell_cls": self._cell_cls,
            "codebook_size": self.codebook_size, "class_mode": self.class_mode,
            "seed": self.seed,
        }, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ClassScrambleWarp":
        blob = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls.__new__(cls)
        obj._codebook = blob["codebook"]
        obj._perm = blob["perm"]
        obj._cell_cls = blob["cell_cls"]
        obj.codebook_size = int(blob["codebook_size"])
        obj.class_mode = str(blob["class_mode"])
        obj.seed = int(blob["seed"])
        return obj


_BUILDERS = {
    "vq": VQWarp,
    "scramble": ScrambleWarp,
    "class_scramble": ClassScrambleWarp,
    "flow_warp": FlowWarpAdapter.fit,
    "piecewise_affine": PiecewiseAffineWarp,
    "class_collapse": ClassCollapseWarp,
}

#: Names accepted by :func:`build_warp`.
WARP_NAMES = tuple(_BUILDERS)


def build_warp(name: str, fit_data: Tensor, **kw) -> Warp:
    """Factory: build the warp named ``name`` fitted on ``fit_data``.

    Args:
        name: one of :data:`WARP_NAMES` — ``"vq"``, ``"scramble"``,
            ``"class_scramble"``, ``"flow_warp"``, ``"piecewise_affine"``,
            ``"class_collapse"``.
        fit_data: ``(N, d)`` latents used to fit/define the warp.
        **kw: forwarded to the chosen warp's constructor (or
            :meth:`FlowWarpAdapter.fit`), e.g. ``codebook_size`` + ``seed`` for
            vq/scramble; additionally ``labels`` + ``class_mode`` for
            class_scramble; ``bins_per_dim`` + ``seed`` for piecewise_affine;
            ``target``/``n_epochs``/``standardize``/``seed`` for flow_warp;
            ``labels`` + ``n_directions`` for class_collapse.

    Returns:
        A fitted :class:`Warp`.
    """
    if name not in _BUILDERS:
        raise ValueError(f"Unknown warp name {name!r}; expected one of {WARP_NAMES}.")
    return _BUILDERS[name](fit_data, **kw)
