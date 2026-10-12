"""The ``piecewise_affine`` warp.

A map that is **affine within each grid cell** (so it has a real, nonzero local
Jacobian — a controller can exploit local linear structure, no straight-through
needed), **discontinuous across cells** (globally non-smooth), and **exactly
invertible / information-preserving**.

Construction (grid + per-cell signed permutation):

* Work in per-dim-NORMALIZED coordinates so every cell is unit-sized. From
  ``fit_data`` store per-dim ``lo = min`` and ``bw = (max - lo) / bins_per_dim``;
  then ``u = (z - lo) / bw`` makes every unit cell width-1 in every dim.
* Cell index ``c = floor(u)`` — NO clamping. Cells are an unbounded grid
  (``bins_per_dim`` only sets cell *size* via ``bw``), so the centered offset
  ``o = u - (c + 0.5)`` lies in ``[-0.5, 0.5]^d`` for EVERY point, even outside
  the fit range. That keeps the per-cell map permutation-safe → invertible
  everywhere.
* The per-cell map is a random **signed permutation** ``S_c`` (a permutation of
  the ``d`` axes + a ±1 sign per axis), derived deterministically from an
  explicit, process-stable hash of ``(c, seed)``. Because all dims share the box
  ``[-0.5, 0.5)``, permuting + sign-flipping keeps ``o`` inside the box → the
  output stays in cell ``c`` → invertible by re-gridding. The result is reduced
  mod 1 back into ``[-0.5, 0.5)`` (see :meth:`PiecewiseAffineWarp._wrap`) so the
  single boundary value a sign-flip can escape (``+0.5``, only from an input
  exactly on a cell edge) folds back in — making invertibility exact everywhere.

The local Jacobian is ``J = diag(bw) · S_c · diag(1/bw)`` a.e. — a scaled signed
permutation: exactly one nonzero per row/column, all nonzero. This is the
"differentiable cousin of vq": same cellular partition, affine (orthogonal-up-to
-scaling) within a cell instead of constant.
"""
import hashlib
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from torch import Tensor

from latent_smoothness.warp.base import Warp


def _cell_signed_permutation(cell: np.ndarray, seed: int, d: int) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic ``(perm, signs)`` for one integer cell index.

    Args:
        cell: length-``d`` integer vector (a cell index ``c``).
        seed: warp seed.
        d: latent dimension.

    Returns:
        ``perm`` (length-``d`` permutation of ``range(d)``) and ``signs``
        (length-``d`` vector of ±1, float32).

    The cell+seed are hashed with SHA-256 — an **explicit, process-stable** hash
    — to seed a local ``RandomState``. Python's builtin ``hash()`` is salted per
    process (``PYTHONHASHSEED``) and would NOT be reproducible across runs, so it
    is deliberately not used here.
    """
    digest = hashlib.sha256(
        np.asarray(cell, dtype=np.int64).tobytes() + np.int64(seed).tobytes()
    ).digest()
    h = int.from_bytes(digest[:8], "little")
    # Seed a RandomState with the full 64-bit hash via two uint32 words
    # (RandomState rejects a single >2**32 int but accepts a uint32 array).
    rs = np.random.RandomState([h & 0xFFFFFFFF, (h >> 32) & 0xFFFFFFFF])
    perm = rs.permutation(d).astype(np.int64)
    signs = rs.randint(0, 2, size=d).astype(np.float32) * 2.0 - 1.0
    return perm, signs


class PiecewiseAffineWarp(Warp):
    """Affine-within-cell, discontinuous-across-cell, exactly invertible warp.

    Args:
        fit_data: ``(N, d)`` latents whose per-dim min/range define the grid.
        bins_per_dim: cell-size divisor ``B``; ``bw = (max - min) / B`` per dim.
            Sets cell *size*, not a boundary — the grid is unbounded.
        seed: selects the per-cell signed permutations.

    Note:
        ``forward``/``backproject`` derive the per-cell signed permutation via a
        NumPy hash (:func:`_cell_signed_permutation`), so this warp is **not
        traceable under torch.func / functorch** — ``torch.func.hessian`` (used
        by ``compute_boundary_geometry``) raises "Cannot access data pointer…"
        and that metric degrades to a flagged ``curvature = 0``. Ordinary
        ``torch.autograd`` is unaffected (the local Jacobian is exact), and the
        gradient-free ``secant_roughness`` metric works for all warps. The 0 is a
        tooling limitation here, not a real smoothness reading — see the README
        section "Known limitations".
    """

    name = "piecewise_affine"
    is_invertible = True
    is_differentiable = True

    def __init__(self, fit_data: Tensor, bins_per_dim: int = 4, seed: int = 0):
        fit = fit_data.detach().float()
        if fit.dim() != 2:
            raise ValueError(f"fit_data must be (N, d); got shape {tuple(fit.shape)}.")
        lo = fit.min(dim=0).values
        hi = fit.max(dim=0).values
        # clamp_min avoids a zero bin-width (→ div-by-zero) on constant dims.
        bw = ((hi - lo) / bins_per_dim).clamp_min(1e-8)
        self._lo = lo.cpu()
        self._bw = bw.cpu()
        self.bins_per_dim = int(bins_per_dim)
        self.seed = int(seed)
        self.d = int(fit.shape[1])

    @staticmethod
    def _wrap(o: Tensor) -> Tensor:
        """Reduce an offset into the half-open cell box ``[-0.5, 0.5)`` (mod 1).

        A signed permutation maps ``[-0.5, 0.5)`` to ``(-0.5, 0.5]``; the lone
        escaped value ``+0.5`` (only reachable from an input exactly on a cell
        boundary, e.g. a per-dim data extreme) would floor into the *next* cell
        and break re-gridding. Wrapping sends ``+0.5 → -0.5`` so the point stays
        in its cell, making the warp exactly invertible **everywhere**, boundary
        points included. ``floor`` has zero gradient, so this is the identity on
        cell interiors — the affine local Jacobian is unaffected.
        """
        return o - torch.floor(o + 0.5)

    def _perm_sign(self, c: Tensor) -> Tuple[Tensor, Tensor]:
        """Per-row signed permutation for a batch of cell indices ``c`` (N, d).

        Returns ``perm_idx`` (N, d, long) and ``sign`` (N, d, float) on
        ``c.device``. Computed by hashing the *unique* cells in numpy; these are
        detached index/scale constants and carry no gradient.
        """
        c_np = c.detach().cpu().numpy().astype(np.int64)
        uniq, inv = np.unique(c_np, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        perms = np.empty((uniq.shape[0], self.d), dtype=np.int64)
        signs = np.empty((uniq.shape[0], self.d), dtype=np.float32)
        for k in range(uniq.shape[0]):
            perms[k], signs[k] = _cell_signed_permutation(uniq[k], self.seed, self.d)
        perm_idx = torch.from_numpy(perms[inv]).to(c.device)
        sign = torch.from_numpy(signs[inv]).to(c.device)
        return perm_idx, sign

    def forward(self, z: Tensor) -> Tensor:
        """Apply the warp. Device/dtype-preserving; accepts any ``(..., d)``."""
        lo = self._lo.to(device=z.device, dtype=z.dtype)
        bw = self._bw.to(device=z.device, dtype=z.dtype)
        orig_shape = z.shape
        u = (z.reshape(-1, self.d) - lo) / bw
        c = torch.floor(u)
        center = c + 0.5
        o = u - center                                   # in [-0.5, 0.5)^d
        perm_idx, sign = self._perm_sign(c)
        o_prime = self._wrap(sign.to(o.dtype) * torch.gather(o, 1, perm_idx))
        out = lo + bw * (center + o_prime)
        return out.reshape(orig_shape).to(z.dtype)

    def backproject(self, z_w: Tensor) -> Tensor:
        """Exact inverse: ``z`` such that ``forward(z) == z_w`` (re-gridding).

        ``z_w`` lies in the same cell ``c`` as ``z`` did (the signed permutation
        keeps the offset inside the cell box), so flooring recovers ``c`` and the
        signed permutation is undone by an inverse-permutation gather.
        """
        lo = self._lo.to(device=z_w.device, dtype=z_w.dtype)
        bw = self._bw.to(device=z_w.device, dtype=z_w.dtype)
        orig_shape = z_w.shape
        u = (z_w.reshape(-1, self.d) - lo) / bw
        c = torch.floor(u)                               # == forward's cell
        center = c + 0.5
        o_prime = u - center
        perm_idx, sign = self._perm_sign(c)
        # Invert the signed permutation. forward did o'_k = sign_k · o_{perm_k},
        # so o_j = sign_{inv_j} · o'_{inv_j} with inv = perm^-1. Use an
        # inverse-permutation gather (not scatter_) so the op stays deterministic
        # under torch.use_deterministic_algorithms(True).
        inv = torch.argsort(perm_idx, dim=1)
        o = self._wrap(
            torch.gather(sign, 1, inv).to(o_prime.dtype) * torch.gather(o_prime, 1, inv)
        )
        out = lo + bw * (center + o)
        return out.reshape(orig_shape).to(z_w.dtype)

    def inverse(self, z_w: Tensor) -> Tensor:
        """Alias of :meth:`backproject` (the warp is exactly invertible)."""
        return self.backproject(z_w)

    def forward_ste(self, z: Tensor) -> Tensor:
        """Identical to :meth:`forward` — the warp is differentiable (no STE)."""
        return self.forward(z)

    def save(self, path: Union[str, Path]) -> None:
        """Persist ``lo, bw, bins_per_dim, seed`` to ``path``."""
        torch.save(
            {
                "lo": self._lo,
                "bw": self._bw,
                "bins_per_dim": self.bins_per_dim,
                "seed": self.seed,
                "d": self.d,
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PiecewiseAffineWarp":
        """Reconstruct a warp from :meth:`save` output."""
        blob = torch.load(path, map_location="cpu", weights_only=True)
        obj = cls.__new__(cls)
        obj._lo = blob["lo"]
        obj._bw = blob["bw"]
        obj.bins_per_dim = int(blob["bins_per_dim"])
        obj.seed = int(blob["seed"])
        obj.d = int(blob["d"])
        return obj
