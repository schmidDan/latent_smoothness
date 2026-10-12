import torch
from latent_smoothness import compute_lid
from tests.conftest import DZ, N, SEED


def test_lid_approx_dz_for_gaussian(unit_gaussian_latents):
    result = compute_lid(unit_gaussian_latents)
    assert abs(result.mean_lid_twonn - DZ) <= 1.0, \
        f"Expected mean TwoNN LID ≈ {DZ}, got {result.mean_lid_twonn}"


def test_lid_approx_1_for_1d_submanifold():
    torch.manual_seed(SEED)
    coords_1d = torch.randn(N, 1)
    z = torch.zeros(N, DZ)
    z[:, 0] = coords_1d[:, 0]
    result = compute_lid(z)
    assert abs(result.mean_lid_twonn - 1.0) <= 0.5, \
        f"Expected mean TwoNN LID ≈ 1.0 for 1D submanifold, got {result.mean_lid_twonn}"
