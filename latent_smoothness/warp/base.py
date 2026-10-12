"""The public ``Warp`` interface — the contract every warp implements.

A ``Warp`` maps latents ``z`` of shape ``(..., d)`` to warped latents of the
same shape and back. Concrete warps are ``vq``, ``scramble``,
``class_scramble``, ``flow_warp``, ``piecewise_affine`` and ``class_collapse``;
all are built through :func:`latent_smoothness.warp.build_warp`.

Every method is **device-preserving**: a warp moves its own fitted tensors onto
the input's device on the fly and returns a result on that device.
"""
import abc
from pathlib import Path
from typing import Union

from torch import Tensor


class Warp(abc.ABC):
    """Abstract base for all warps.

    Subclasses set three class attributes and implement :meth:`forward`,
    :meth:`save`, :meth:`load`:

    Attributes:
        name: short identifier, e.g. ``"vq"``, ``"piecewise_affine"``.
        is_invertible: ``True`` iff :meth:`backproject` is an exact inverse of
            :meth:`forward` (flow_warp, piecewise_affine). ``False`` for the
            lossy warps (vq, scramble, class_scramble, class_collapse), whose
            ``backproject`` is best-effort.
        is_differentiable: ``True`` iff :meth:`forward` carries a usable
            (nonzero) gradient and needs no straight-through surrogate
            (flow_warp, piecewise_affine, class_collapse); ``False`` for the
            quantizing warps (vq, scramble, class_scramble).
    """

    name: str
    is_invertible: bool
    is_differentiable: bool

    @abc.abstractmethod
    def forward(self, z: Tensor) -> Tensor:
        """Apply the warp ``W(z)``. Shape- and device/dtype-preserving."""
        raise NotImplementedError

    def backproject(self, z_w: Tensor) -> Tensor:
        """Map a warped latent back toward latent space.

        Default: the exact inverse (invertible warps, via :meth:`inverse`).
        Non-invertible warps override this with best-effort recovery (reading
        the snapped value); that is NOT a true inverse.
        """
        return self.inverse(z_w)

    def inverse(self, z_w: Tensor) -> Tensor:
        """Exact inverse for invertible warps; non-invertible warps instead
        override :meth:`backproject` and never call this."""
        raise NotImplementedError(
            f"{type(self).__name__} is not invertible; call backproject() instead."
        )

    def forward_ste(self, z: Tensor) -> Tensor:
        """Straight-through forward.

        Differentiable warps return :meth:`forward` unchanged. Non-differentiable
        warps return ``z + (forward(z) - z).detach()`` — forward value is the
        snapped output, gradient is the identity surrogate.
        """
        if self.is_differentiable:
            return self.forward(z)
        return z + (self.forward(z) - z).detach()

    @property
    def needs_ste(self) -> bool:
        """``True`` iff :meth:`forward_ste` must use a straight-through
        surrogate, i.e. the warp is not differentiable."""
        return not self.is_differentiable

    @abc.abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """Persist fitted state to ``path``."""
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def load(cls, path: Union[str, Path]) -> "Warp":
        """Reconstruct a warp from :meth:`save` output."""
        raise NotImplementedError
