import numpy as np
import torch
import tempfile
import os

from latent_smoothness.mine import estimate_mi
from latent_smoothness.warp.quantize import posthoc_vq
from latent_smoothness.warp.scramble import bijective_scramble
from tests.conftest import DZ, SEED

# MINE requires larger N for stable estimation: N=128 (conftest) is too small
# for reliable logsumexp bounds in 8D input space (DZ=4+4). Tests create N=1024
# tensors locally, mirroring the K=256 warp-test pattern.
N_MINE = 1024


def test_mi_independent_gaussians():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    z1 = torch.randn(N_MINE, DZ)
    torch.manual_seed(SEED + 1)
    z2 = torch.randn(N_MINE, DZ)
    mi = estimate_mi(z1, z2, n_epochs=200)
    # MINE has a known finite-sample positive bias (the logsumexp partition term
    # is biased by Jensen's inequality): ~0.2 rather than 0 is what N=1024 and
    # 200 epochs achieve, so 0.3 is used as the independence threshold.
    assert mi < 0.3, f"Expected MI < 0.3 for independent Gaussians, got {mi}"


def test_mi_correlated_pairs():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    z1 = torch.randn(N_MINE, DZ)
    z2 = z1 + torch.randn(N_MINE, DZ) * 0.1  # highly correlated
    mi = estimate_mi(z1, z2, n_epochs=200)
    assert mi > 0.5, f"Expected MI > 0.5 for correlated pairs, got {mi}"


def test_mi_cache_hit_matches_cache_miss():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    z1 = torch.randn(N_MINE, DZ)
    z2 = z1 + torch.randn(N_MINE, DZ) * 0.1

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "mine_cache.pt")
        # Cache miss: train and save
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        mi_miss = estimate_mi(z1, z2, n_epochs=200, save_path=path)
        # Cache hit: load and infer
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        mi_hit = estimate_mi(z1, z2, n_epochs=200, save_path=path)

    assert abs(mi_miss - mi_hit) <= 0.05, \
        f"Cache hit ({mi_hit:.4f}) should match cache miss ({mi_miss:.4f}) within atol=0.05"


def test_mi_bijective_vs_lossy():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    z = torch.randn(N_MINE, DZ)
    z_bijective = bijective_scramble(z, codebook_size=64)[0]
    z_lossy = posthoc_vq(z, codebook_size=4)[0]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    mi_bijective = estimate_mi(z, z_bijective, n_epochs=200)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    mi_lossy = estimate_mi(z, z_lossy, n_epochs=200)

    assert mi_bijective - mi_lossy > 0.3, \
        f"Expected MI(bijective) - MI(lossy) > 0.3, got {mi_bijective:.4f} - {mi_lossy:.4f} = {mi_bijective - mi_lossy:.4f}"
