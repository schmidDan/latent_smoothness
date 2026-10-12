import pytest
import torch
import torch.nn as nn
from latent_smoothness import compute_ppl
from tests.conftest import M, SEED

pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("ensure_lpips_weights")]


def test_no_ppl_spikes_for_linear_spatial(linear_spatial_decoder, unit_gaussian_latents, device):
    torch.manual_seed(SEED)
    z_start = unit_gaussian_latents[:M]
    z_end = unit_gaussian_latents[M:2*M]
    result = compute_ppl(linear_spatial_decoder, z_start, z_end, n_steps=50, device=device)
    assert result.ppl_spikes.sum() == 0, \
        f"Expected no PPL spikes for linear spatial decoder, got {result.ppl_spikes.sum()}"


def test_ppl_spikes_at_correct_location_for_rough(unit_gaussian_latents, device):
    """Verify PPL spike locations are near ||z||=1.0.

    Uses a very narrow spike (exponent -1000 vs -100 in the conftest fixture) so
    that the LPIPS maximum segment's midpoint reliably falls within atol=0.1 of
    ||z||=1.0 despite VGG's perceptual non-linearity. The broader conftest fixture
    is sufficient for detecting that spikes EXIST (test_geodesic_discrepancy_positive)
    but too wide for location accuracy with LPIPS.
    """
    torch.manual_seed(SEED)

    class NarrowSpikeSpatial(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 64, bias=False)
            self.unflatten = nn.Unflatten(-1, (1, 8, 8))

        def forward(self, z):
            scale = 1.0 + 10.0 * torch.exp(-1000 * (z.norm(dim=-1, keepdim=True) - 1.0) ** 2)
            return self.unflatten(self.linear(z * scale))

    decoder = NarrowSpikeSpatial().to(device)
    # Radial paths: same direction, ||z_start||=0.5, ||z_end||=2.0
    z_dir = unit_gaussian_latents[:M] / unit_gaussian_latents[:M].norm(dim=-1, keepdim=True)
    z_start = z_dir * 0.5
    z_end = z_dir * 2.0

    result = compute_ppl(decoder, z_start, z_end, n_steps=200, device=device)
    assert result.ppl_spikes.sum() > 0, "Expected PPL spikes for rough spatial decoder"
    norms = result.spike_z_locations.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=0.1), \
        f"Spike locations should be near ||z||=1.0, got norms: {norms}"


def test_spike_path_indices_in_range(rough_spatial_decoder, unit_gaussian_latents, device):
    torch.manual_seed(SEED)
    z_dir = unit_gaussian_latents[:M] / unit_gaussian_latents[:M].norm(dim=-1, keepdim=True)
    z_start = z_dir * 0.5
    z_end = z_dir * 2.0
    result = compute_ppl(rough_spatial_decoder, z_start, z_end, n_steps=200, device=device)
    if result.spike_path_indices.numel() > 0:
        assert result.spike_path_indices.min() >= 0
        assert result.spike_path_indices.max() <= M - 1


def test_compute_ppl_raises_for_flat_decoder(linear_decoder, unit_gaussian_latents, device):
    torch.manual_seed(SEED)
    z_start = unit_gaussian_latents[:M]
    z_end = unit_gaussian_latents[M:2*M]
    with pytest.raises(ValueError, match="4D"):
        compute_ppl(linear_decoder, z_start, z_end, device=device)


def test_ppl_spike_only_at_local_rough_zone(unit_gaussian_latents, device):
    """Local-contrast detector fires at the single rough segment and nowhere else.

    Uses a very narrow multiplicative spike (width≈0.05 centred at ||z||=1).
    The path crosses this zone once; the detector must find exactly one spike per
    path and must NOT fire on the smooth regions on either side.  This tests the
    case that is NOT covered by the two existing tests:
      - test_no_ppl_spikes_for_linear_spatial: globally smooth → 0 spikes
      - test_ppl_spikes_at_correct_location_for_rough: globally rough → spikes ≥ 1
    Here: locally rough (one isolated zone), globally smooth → exactly 1 spike.
    """
    torch.manual_seed(SEED)

    class LocallyRoughSpatial(nn.Module):
        """Spatial decoder that is linear everywhere except a narrow band at ||z||=1."""
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 64, bias=False)
            self.unflatten = nn.Unflatten(-1, (1, 8, 8))

        def forward(self, z):
            # Spike width ≈ 0.05: only the single path segment crossing ||z||=1 is affected
            scale = 1.0 + 50.0 * torch.exp(-200.0 * (z.norm(dim=-1, keepdim=True) - 1.0) ** 2)
            return self.unflatten(self.linear(z * scale))

    decoder = LocallyRoughSpatial().to(device)

    # Radial paths: ||z_start||=0.3, ||z_end||=2.0 — each path crosses ||z||=1 exactly once
    z_dir = unit_gaussian_latents[:M] / unit_gaussian_latents[:M].norm(dim=-1, keepdim=True)
    z_start = z_dir * 0.3
    z_end = z_dir * 2.0

    result = compute_ppl(decoder, z_start, z_end, n_steps=200, device=device)

    # Each of the M paths must fire exactly one spike
    spikes_per_path = result.ppl_spikes.sum(dim=-1)  # (M,)
    assert (spikes_per_path == 1).all(), (
        f"Expected exactly 1 spike per path for locally-rough decoder, "
        f"got per-path counts: {spikes_per_path.tolist()}"
    )
