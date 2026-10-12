from typing import Tuple

import numpy as np
import torch
from torch import Tensor


def posthoc_vq(z: Tensor, codebook_size: int, n_iter: int = 100) -> Tuple[Tensor, Tensor]:
    """Post-hoc vector quantization of ``(N, d)`` latents with K-means.

    Returns ``(quantized_z, codebook)``: each latent replaced by its nearest of
    ``codebook_size`` centroids, and the ``(codebook_size, d)`` centroids.
    Deterministic (fixed K-means ``random_state``).
    """
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits

    X = z.detach().cpu().numpy()
    km = KMeans(n_clusters=codebook_size, n_init=10, random_state=42, max_iter=n_iter)
    # Single-threaded OpenMP for the fit: avoids a KMeans thread-pool deadlock
    # seen with some sklearn versions on Windows, scoped to this call only.
    with threadpool_limits(limits=1, user_api="openmp"):
        km.fit(X)
    labels = km.labels_  # (N,)
    codebook = torch.from_numpy(km.cluster_centers_.astype(np.float32))
    quantized_z = codebook[labels]
    return quantized_z.to(device=z.device, dtype=z.dtype), codebook.to(dtype=z.dtype)
