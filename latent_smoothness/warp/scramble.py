from typing import Callable, Tuple

import numpy as np
import torch
from torch import Tensor

from latent_smoothness.warp.quantize import posthoc_vq


def _fit_scramble(z: Tensor, codebook_size: int, seed: int = 42) -> Tuple[Tensor, Tensor, Tensor]:
    """Quantise ``z`` and draw a seeded random permutation of the codebook cells.

    Returns ``(quantized_z, codebook (K, dz), perm (K,) long)``.
    """
    quantized_z, codebook = posthoc_vq(z, codebook_size)
    rng = np.random.RandomState(seed)
    perm = torch.from_numpy(rng.permutation(codebook.shape[0]).astype(np.int64))
    return quantized_z, codebook, perm


def bijective_scramble(
    z: Tensor,
    codebook_size: int,
    seed: int = 42,
) -> Tuple[Tensor, Callable, Callable]:
    """Quantise ``z`` to ``codebook_size`` cells, then permute the cells.

    Returns ``(scrambled_z, encode_fn, decode_fn)``: ``encode_fn`` snaps points to
    their cell and maps them to the permuted cell's centroid; ``decode_fn``
    applies the inverse permutation, so ``decode_fn(encode_fn(q)) == q`` for any
    codebook point ``q``.
    """
    quantized_z, codebook, perm = _fit_scramble(z, codebook_size, seed)
    K = codebook.shape[0]
    inv_perm = torch.zeros(K, dtype=torch.long)
    inv_perm[perm] = torch.arange(K, dtype=torch.long)

    # Map each input point to its cell index
    def _assign(pts: Tensor) -> Tensor:
        """Returns cell index (original space) for each point."""
        dists = torch.cdist(pts.float(), codebook.to(pts.device).float())  # (N, K)
        return dists.argmin(dim=1)  # (N,)

    def encode_fn(pts: Tensor) -> Tensor:
        cell_ids = _assign(pts)
        permuted_ids = perm.to(pts.device)[cell_ids]
        return codebook.to(pts.device)[permuted_ids].to(dtype=pts.dtype)

    def decode_fn(pts: Tensor) -> Tensor:
        cell_ids = _assign(pts)
        original_ids = inv_perm.to(pts.device)[cell_ids]
        return codebook.to(pts.device)[original_ids].to(dtype=pts.dtype)

    scrambled_z = encode_fn(quantized_z)
    return scrambled_z, encode_fn, decode_fn


def _majority_cell_classes(cell_ids: Tensor, labels: Tensor, K: int) -> Tensor:
    """Per-cell class = majority label of its points (ties -> lowest class; empty -> -1)."""
    labels = labels.to(torch.long)
    cell_cls = torch.full((K,), -1, dtype=torch.long)
    for c in range(K):
        lab_c = labels[cell_ids == c]
        if lab_c.numel() == 0:
            continue
        vals, counts = lab_c.unique(return_counts=True)  # vals ascending
        # argmax returns the FIRST maximal index -> lowest class value on ties.
        cell_cls[c] = int(vals[int(counts.argmax())])
    return cell_cls


def _fit_class_scramble(z: Tensor, codebook_size: int, labels: Tensor,
                        class_mode: str, seed: int = 42):
    """Fit a class-constrained scramble: quantise, then permute cells by **disjoint
    transpositions** whose two cells share a class (``within``) or differ in class
    (``across``). Transpositions are involutions -> the permutation is a valid
    bijection with ``inv_perm == perm`` and both modes swap a comparable number of
    cells. Empty cells (class -1) are never paired (left fixed).

    Returns ``(codebook (K,dz), perm (K,) long involution, cell_cls (K,) long)``.
    """
    if class_mode not in ("within", "across"):
        raise ValueError(f"class_mode must be 'within' or 'across', got {class_mode!r}")
    quantized_z, codebook = posthoc_vq(z, codebook_size)
    K = codebook.shape[0]
    cell_ids = torch.cdist(z.detach().cpu().float(), codebook.float()).argmin(dim=1)
    cell_cls = _majority_cell_classes(cell_ids, labels, K)

    rng = np.random.RandomState(seed)
    pairs = []
    if class_mode == "within":
        for v in sorted({int(x) for x in cell_cls.tolist() if x >= 0}):
            cells = torch.nonzero(cell_cls == v, as_tuple=False).squeeze(1).tolist()
            rng.shuffle(cells)
            for i in range(0, len(cells) - 1, 2):
                pairs.append((cells[i], cells[i + 1]))
    else:  # across: greedily pair each cell with a still-free cell of a different class
        candidates = [int(c) for c in rng.permutation(K) if cell_cls[int(c)] >= 0]
        used = set()
        for a in candidates:
            if a in used:
                continue
            for b in candidates:
                if b != a and b not in used and cell_cls[b] != cell_cls[a]:
                    pairs.append((a, b))
                    used.add(a)
                    used.add(b)
                    break

    perm = torch.arange(K, dtype=torch.long)
    for a, b in pairs:
        perm[a] = b
        perm[b] = a
    return codebook, perm, cell_cls


def class_constrained_scramble(z: Tensor, codebook_size: int, labels: Tensor,
                               class_mode: str, seed: int = 42):
    """Functional class-constrained scramble (parallels :func:`bijective_scramble`).

    Because the permutation is an involution, ``decode_fn`` reuses the same ``perm``
    (it is its own inverse), so ``decode_fn(encode_fn(z))`` recovers the quantised z.
    """
    codebook, perm, _ = _fit_class_scramble(z, codebook_size, labels, class_mode, seed)

    def _assign(pts: Tensor) -> Tensor:
        dists = torch.cdist(pts.float(), codebook.to(pts.device).float())
        return dists.argmin(dim=1)

    def encode_fn(pts: Tensor) -> Tensor:
        cell = _assign(pts)
        return codebook.to(pts.device)[perm.to(pts.device)[cell]].to(dtype=pts.dtype)

    # perm is an involution (inv_perm == perm) -> decode == encode.
    decode_fn = encode_fn

    # encode_fn(z) snaps z to cells then permutes — identical to encoding the
    # quantised z (no second K-means needed).
    scrambled_z = encode_fn(z)
    return scrambled_z, encode_fn, decode_fn
