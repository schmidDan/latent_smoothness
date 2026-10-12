"""Label-conditioned dimensionality of a latent point cloud.

A LABEL-EXPLICIT complement to ``lid``: plain intrinsic dimensionality only sees
the point set, so it cannot distinguish two latent spaces that contain the same
points with a different class arrangement (e.g. the ``vq`` and ``scramble``
warps). Conditioning dimensionality on the class label makes the arrangement
visible. Two complementary measures are reported:

  * within-class spread / effective rank — are same-class latents a tight,
    low-dimensional blob or smeared across the space?
  * class-difference vector-field rank / consistency — is the class-centroid
    arrangement low-rank / dominated by one axis (so "toward class A" is a
    consistent direction) or full-rank / scattered?

Labels come from the classifier itself (``argmax``), so the metric is
self-contained (no external labels) and defined for any latent map.
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from latent_smoothness.utils import eval_mode


@dataclass
class LabelDimensionalityResult:
    within_class_effrank: float    # realization 1: mean per-class participation ratio
    within_class_spread: float     # realization 1 companion: mean per-class dispersion
    class_diff_effrank: float      # realization 2: eff. rank of the class-mean cloud
    class_diff_consistency: float  # realization 2 companion: top-direction variance fraction
    fisher_ratio: float            # auxiliary: trace(between)/trace(within) separability
    n_classes_used: int


def _participation_ratio(evals: Tensor) -> float:
    """(Σλ)² / Σλ² — the effective number of dimensions (∈ [1, len])."""
    evals = evals.clamp_min(0.0)
    s1 = evals.sum()
    s2 = (evals * evals).sum()
    if s2 <= 1e-12:
        return float("nan")
    return float((s1 * s1) / s2)


def _compute_label_dimensionality_raw(
    classifier: nn.Module,
    z_points: Tensor,
    min_class_count: int = 5,
) -> LabelDimensionalityResult:
    with torch.no_grad():
        labels = classifier(z_points).argmax(dim=-1)  # (N,)

    z = z_points.detach()

    # One pass: per qualifying class collect (mean, count, effrank, spread, ss).
    means, counts, effranks, spreads = [], [], [], []
    within_ss = 0.0
    for c in labels.unique().tolist():
        Xc = z[labels == c]
        n_c = Xc.shape[0]
        if n_c < min_class_count:
            continue
        m_c = Xc.mean(dim=0)
        centered = Xc - m_c
        cov = centered.t() @ centered / (n_c - 1)       # (dz, dz)
        effranks.append(_participation_ratio(torch.linalg.eigvalsh(cov)))
        spreads.append(float((centered * centered).sum(dim=1).mean()))
        within_ss += float((centered * centered).sum())
        means.append(m_c)
        counts.append(n_c)

    n_used = len(means)
    within_class_effrank = float(torch.tensor(effranks).mean()) if effranks else float("nan")
    within_class_spread = float(torch.tensor(spreads).mean()) if spreads else float("nan")

    if n_used < 2:
        return LabelDimensionalityResult(
            within_class_effrank=within_class_effrank,
            within_class_spread=within_class_spread,
            class_diff_effrank=float("nan"),
            class_diff_consistency=float("nan"),
            fisher_ratio=float("nan"),
            n_classes_used=n_used,
        )

    M = torch.stack(means, dim=0)                 # (Kq, dz)
    Mc = M - M.mean(dim=0, keepdim=True)
    e = torch.linalg.svdvals(Mc).clamp_min(0.0) ** 2   # eigenvalues of the class-mean scatter
    class_diff_effrank = _participation_ratio(e)
    class_diff_consistency = float(e[0] / e.sum()) if float(e.sum()) > 1e-12 else float("nan")

    # Fisher separability: point-weighted trace(between) / trace(within).
    total_n = sum(counts)
    global_mean = sum(n_c * m_c for n_c, m_c in zip(counts, means)) / total_n
    between = sum(n_c * float(((m_c - global_mean) ** 2).sum())
                  for n_c, m_c in zip(counts, means)) / total_n
    within = within_ss / total_n
    fisher_ratio = between / max(within, 1e-12)

    return LabelDimensionalityResult(
        within_class_effrank=within_class_effrank,
        within_class_spread=within_class_spread,
        class_diff_effrank=class_diff_effrank,
        class_diff_consistency=class_diff_consistency,
        fisher_ratio=fisher_ratio,
        n_classes_used=n_used,
    )


def compute_label_dimensionality(
    classifier: nn.Module,
    z_points: Tensor,
    min_class_count: int = 5,
) -> LabelDimensionalityResult:
    """Label-conditioned dimensionality of ``z_points`` (labels = classifier argmax)."""
    with eval_mode(classifier):
        return _compute_label_dimensionality_raw(classifier, z_points, min_class_count)
