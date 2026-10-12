"""Tests for the Jacobian degeneracy flag and the secant-roughness sweep.

Smooth reference: a linear decoder f(z) = W z, where every local Jacobian is
exactly W and every secant ratio is ‖W d̂‖ (constant across scales).
Degenerate reference: a quantized decoder f(z) = W * round(z/c), piecewise
constant in z (zero gradient a.e.), the same family as vq/scramble.
"""
import torch
import torch.nn as nn

from latent_smoothness.jacobian import compute_jacobian_spectrum
from latent_smoothness.roughness import compute_secant_roughness
from tests.conftest import SEED


class LinearDecoder(nn.Module):
    def __init__(self, dz, dx, seed=SEED):
        super().__init__()
        torch.manual_seed(seed)
        self.W = nn.Parameter(torch.randn(dx, dz))

    def forward(self, z):
        return z @ self.W.T


class QuantizedDecoder(nn.Module):
    """Piecewise-constant in z: round() has zero gradient a.e. -> degenerate."""
    def __init__(self, dz, dx, cell=1.0, seed=SEED):
        super().__init__()
        torch.manual_seed(seed)
        self.W = nn.Parameter(torch.randn(dx, dz))
        self.cell = cell

    def forward(self, z):
        return (torch.round(z / self.cell) * self.cell) @ self.W.T


# ---- degeneracy flag ----

def test_smooth_decoder_not_degenerate():
    dz, dx = 6, 12
    dec = LinearDecoder(dz, dx)
    z = torch.randn(40, dz)
    res = compute_jacobian_spectrum(dec, z)
    assert not res.gradient_degenerate, "smooth linear map must not be flagged degenerate"
    assert res.n_zero_fill == 0
    # max singular value of the Jacobian == largest singular value of W
    sv_W = torch.linalg.svdvals(dec.W.detach())[0].item()
    assert abs(res.lipschitz_estimate - sv_W) < 1e-3, \
        f"lipschitz {res.lipschitz_estimate} != svmax(W) {sv_W}"


def test_quantized_decoder_is_degenerate():
    dz, dx = 6, 12
    dec = QuantizedDecoder(dz, dx, cell=1.0)
    # spread z across cells; interiors give exactly-zero Jacobian
    z = torch.randn(40, dz) * 0.1 + 0.5
    res = compute_jacobian_spectrum(dec, z)
    assert res.gradient_degenerate, "piecewise-constant map must be flagged degenerate"
    assert res.lipschitz_estimate < 1e-6


# ---- secant roughness ----

def test_secant_roughness_matches_jacobian_for_linear():
    dz, dx = 6, 12
    dec = LinearDecoder(dz, dx)
    z0 = torch.randn(30, dz)
    z1 = z0 + torch.randn(30, dz)
    res = compute_secant_roughness(dec, z0, z1, n_steps=32)
    # linear map -> secant ratio is scale-independent: flat curve, slope ~ 0
    assert torch.isfinite(res.mean_roughness).all()
    rel_spread = (res.mean_roughness.max() - res.mean_roughness.min()) / res.mean_roughness.mean()
    assert rel_spread < 1e-4, f"linear roughness should be flat across scales, spread={rel_spread:.2e}"
    assert abs(res.loglog_slope) < 1e-2, f"linear slope should be ~0, got {res.loglog_slope}"


def test_secant_roughness_finite_and_defined_for_quantized():
    dz, dx = 6, 12
    dec = QuantizedDecoder(dz, dx, cell=1.0)
    z0 = torch.randn(30, dz)
    z1 = z0 + torch.randn(30, dz) * 3.0  # long paths cross several cell boundaries
    res = compute_secant_roughness(dec, z0, z1, n_steps=32)
    # gradient-free: must be defined (finite) even though the Jacobian is degenerate
    assert torch.isfinite(res.mean_roughness).all()
    assert torch.isfinite(res.max_roughness).all()
    assert float(res.max_roughness.max()) > 0.0, "quantized jumps must produce nonzero roughness"


# ---- graceful degradation: boundary Hessian & geodesic solver ----

class CdistClassifier(nn.Module):
    """Routes through torch.cdist, like the vq/scramble snap. torch.func.hessian
    needs forward-mode AD, which cdist does not implement -> exercises the guard."""
    def __init__(self, dz, k=8, seed=SEED):
        super().__init__()
        torch.manual_seed(seed)
        self.register_buffer("codebook", torch.randn(k, dz))

    def forward(self, z):
        return -torch.cdist(z, self.codebook)  # (B, k) logits = -distance


def test_boundary_curvature_survives_nondifferentiable_hessian():
    from latent_smoothness.boundary import compute_boundary_geometry
    dz = 6
    clf = CdistClassifier(dz)
    z = torch.randn(20, dz)
    res = compute_boundary_geometry(clf, z)          # must not raise
    assert torch.isfinite(res.curvature).all(), "curvature must be finite (0-filled) on Hessian failure"
    assert torch.allclose(res.curvature, torch.zeros_like(res.curvature))
    assert torch.isfinite(res.gradient_norms).all()  # reverse-mode grad still works through cdist


def test_geodesic_solver_failure_returns_nan(monkeypatch):
    import latent_smoothness.geodesic as geo
    from latent_smoothness.geodesic import compute_geodesic_discrepancy

    def boom(*a, **k):
        raise AssertionError("underflow in dt 0.0")
    monkeypatch.setattr(geo, "_arc_length_torchdiffeq", boom)

    dec = LinearDecoder(6, 12)
    z0 = torch.randn(10, 6)
    z1 = z0 + torch.randn(10, 6)
    res = compute_geodesic_discrepancy(dec, z0, z1, solver="torchdiffeq")  # must not raise
    assert res.solver_failed is True
    assert torch.isnan(res.discrepancy).all(), "failed solver must yield NaN discrepancy"
