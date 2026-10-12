from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class LIDResult:
    lid_twonn: Tensor
    lid_mle: Tensor
    mean_lid_twonn: float
    mean_lid_mle: float


def _twonn_lid(X: np.ndarray):
    """Returns per-point TwoNN estimates (noisy) and global TwoNN estimate (stable).

    Per-point: d_i = 1/log(r2_i/r1_i)
    Global (mean_lid_twonn): N / sum(log(r2_i/r1_i))  — harmonic mean of per-point estimates.
    """
    from sklearn.neighbors import NearestNeighbors

    nbrs = NearestNeighbors(n_neighbors=3, algorithm="auto").fit(X)
    dists, _ = nbrs.kneighbors(X)
    r1 = dists[:, 1]  # 1st nearest neighbour
    r2 = dists[:, 2]  # 2nd nearest neighbour
    log_mu = np.log(r2 / np.maximum(r1, 1e-12))
    log_mu_clamped = np.maximum(log_mu, 1e-12)

    lid_per_point = (1.0 / log_mu_clamped).astype(np.float32)
    # Global estimate: N / sum(log mu) — equivalent to harmonic mean of per-point values
    mean_lid = float(len(X) / np.maximum(log_mu_clamped.sum(), 1e-12))
    return lid_per_point, mean_lid


def _mle_lid(X: np.ndarray, k: int = 10):
    """Levina-Bickel MLE intrinsic dimension estimator."""
    from sklearn.neighbors import NearestNeighbors

    N = X.shape[0]
    k_actual = min(k + 1, N)
    nbrs = NearestNeighbors(n_neighbors=k_actual, algorithm="auto").fit(X)
    dists, _ = nbrs.kneighbors(X)
    # dists[:, 0] is self (dist=0); dists[:, 1..k] are k nearest neighbors
    dists = dists[:, 1:]  # (N, k)
    r_k = dists[:, -1]   # (N,) — k-th nearest neighbor distance

    # log(r_k / r_j) for j=1..k-1; all >= 0 since r_j <= r_k
    log_ratios = np.log(r_k[:, None] / np.maximum(dists[:, :-1], 1e-12))  # (N, k-1)
    lid = (k - 1) / np.maximum(log_ratios.sum(axis=1), 1e-12)
    return lid.astype(np.float32)


def compute_lid(z_points: Tensor, k_mle: int = 10) -> LIDResult:
    X = z_points.detach().cpu().numpy()

    lid_twonn_np, mean_twonn = _twonn_lid(X)
    lid_mle_np = _mle_lid(X, k=k_mle)

    lid_twonn = torch.from_numpy(lid_twonn_np)
    lid_mle = torch.from_numpy(lid_mle_np)

    return LIDResult(
        lid_twonn=lid_twonn,
        lid_mle=lid_mle,
        mean_lid_twonn=mean_twonn,
        mean_lid_mle=float(lid_mle.mean()),
    )
