import pytest
import torch
import torch.nn as nn
from latent_smoothness import compute_geodesic_discrepancy
from tests.conftest import M, SEED


def test_geodesic_discrepancy_zero_for_linear(linear_decoder, unit_gaussian_latents):
    torch.manual_seed(SEED)
    z_start = unit_gaussian_latents[:M]
    z_end = unit_gaussian_latents[M:2*M]
    result = compute_geodesic_discrepancy(linear_decoder, z_start, z_end, solver="euler", n_steps=20)
    assert torch.allclose(result.discrepancy, torch.zeros_like(result.discrepancy), atol=1e-4), \
        f"Expected near-zero discrepancy for linear decoder, got {result.discrepancy}"


def test_geodesic_discrepancy_positive_for_rough(rough_nonlinear_decoder, unit_gaussian_latents):
    torch.manual_seed(SEED)
    z_start = unit_gaussian_latents[:M]
    z_end = unit_gaussian_latents[M:2*M]
    result = compute_geodesic_discrepancy(rough_nonlinear_decoder, z_start, z_end, solver="euler", n_steps=20)
    assert result.discrepancy.mean() > 0, \
        "Expected positive discrepancy for rough nonlinear decoder"


@pytest.mark.slow
def test_all_solvers_agree(unit_gaussian_latents):
    """All four solvers agree to atol=5e-3 on a mildly nonlinear decoder.

    The rough_nonlinear_decoder fixture has an extremely sharp spike (width≈0.1)
    that prevents Euler convergence at any practical step count. We use a
    milder inline decoder (width≈0.5) so the integrand is smooth enough for all
    solvers to agree, while still being nonlinear.
    """
    torch.manual_seed(SEED)
    z_start = unit_gaussian_latents[:M]
    z_end = unit_gaussian_latents[M:2*M]

    class MildNonlinear(nn.Module):
        def forward(self, z):
            scale = 1.0 + 0.5 * torch.exp(-2.0 * (z.norm(dim=-1, keepdim=True) - 1.0) ** 2)
            return z * scale

    decoder = MildNonlinear()

    solvers = ["euler", "rk4", "torchdiffeq", "torchdiffeq_adjoint"]
    n_steps_map = {
        "euler": 2000,
        "rk4": 500,
        "torchdiffeq": 20,
        "torchdiffeq_adjoint": 20,
    }
    lengths = {}
    for solver in solvers:
        result = compute_geodesic_discrepancy(
            decoder, z_start, z_end, solver=solver, n_steps=n_steps_map[solver],
            rtol=1e-5, atol=1e-7
        )
        lengths[solver] = result.geodesic_path_length

    ref = lengths["torchdiffeq"]
    for solver, length in lengths.items():
        # atol=5e-3 because first-order Euler convergence with a smooth nonlinear
        # decoder still has small residual error at 2000 steps
        assert torch.allclose(length, ref, atol=5e-3), \
            f"Solver {solver!r} disagrees with torchdiffeq: max diff = {(length - ref).abs().max()}"
