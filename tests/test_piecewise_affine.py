"""Tests for the piecewise_affine warp.

Pins the four defining properties: exact invertibility (information-preserving),
a real scaled-signed-permutation local Jacobian (vs vq's zero Jacobian),
discontinuity across cells (global non-smoothness), and determinism in `seed`.
"""
import torch

from latent_smoothness.warp.piecewise_affine import PiecewiseAffineWarp


def _grid_warp(d, span, bins_per_dim, seed=0):
    """A warp whose grid has lo=0 and bw=span/bins_per_dim per dim."""
    fit = torch.zeros(2, d)
    fit[1] = float(span)
    return PiecewiseAffineWarp(fit, bins_per_dim=bins_per_dim, seed=seed)


def test_invertible_recon_zero():
    """backproject(forward(z)) == z exactly (information-preserving)."""
    d = 4
    g = torch.Generator().manual_seed(0)
    z = torch.randn(64, d, generator=g).float() * 2.0
    warp = PiecewiseAffineWarp(z, bins_per_dim=4, seed=0)

    recon = warp.backproject(warp.forward(z))
    err = (recon - z).abs().max()
    assert err < 1e-5, f"piecewise_affine must be exactly invertible: max err {err:.2e}"


def test_local_jacobian_generalized_permutation():
    """The autograd Jacobian of forward at an interior point is a scaled signed
    permutation: exactly one nonzero entry per row AND per column, all nonzero."""
    d = 3
    warp = _grid_warp(d, span=4.0, bins_per_dim=4, seed=0)  # lo=0, bw=1
    # An interior point: offsets ~ {-0.2, +0.2, -0.1}, no boundary.
    z0 = torch.tensor([0.3, 1.7, 2.4])

    J = torch.autograd.functional.jacobian(warp.forward, z0)
    assert J.shape == (d, d)

    nz = J.abs() > 1e-6
    assert torch.equal(nz.sum(dim=1), torch.ones(d, dtype=torch.long)), \
        f"each row must have exactly one nonzero (rows={nz.sum(dim=1).tolist()})\nJ={J}"
    assert torch.equal(nz.sum(dim=0), torch.ones(d, dtype=torch.long)), \
        f"each column must have exactly one nonzero (cols={nz.sum(dim=0).tolist()})\nJ={J}"
    # Real, nonzero local-linear map (contrast: vq's Jacobian is all zeros).
    assert J.abs().sum() > 0


def test_discontinuous_across_cells():
    """A tiny input step that crosses a cell boundary causes a large output jump
    (global non-smoothness), unlike the smooth within-cell behaviour."""
    d = 4
    warp = _grid_warp(d, span=8.0, bins_per_dim=8, seed=0)  # lo=0, bw=1
    g = torch.Generator().manual_seed(0)

    k = torch.randint(1, 6, (64,), generator=g).float()         # integer cell along dim 0
    z_lo = torch.rand(64, d, generator=g) * 6.0 + 1.0           # other dims interior
    z_lo[:, 0] = k + 0.999                                       # just below boundary k+1
    z_hi = z_lo.clone()
    z_hi[:, 0] = k + 1.001                                       # just above -> next cell

    dz = (z_hi - z_lo).norm(dim=1)                              # ~2e-3 each
    jump = (warp.forward(z_hi) - warp.forward(z_lo)).norm(dim=1)

    assert float(dz.max()) < 1e-2, "control: the input step must be small"
    # Median output jump must dwarf the input step (a genuine discontinuity).
    assert float(jump.median()) > 50 * float(dz.max()), \
        f"crossing a cell boundary should jump: median jump {float(jump.median()):.3f}, dz {float(dz.max()):.4f}"


def test_deterministic():
    """Same seed => identical forward; different seed => different forward."""
    d = 4
    g = torch.Generator().manual_seed(0)
    z = torch.randn(128, d, generator=g).float() * 2.0
    fit = z

    w_a = PiecewiseAffineWarp(fit, bins_per_dim=4, seed=0)
    w_b = PiecewiseAffineWarp(fit, bins_per_dim=4, seed=0)
    w_c = PiecewiseAffineWarp(fit, bins_per_dim=4, seed=1)

    assert torch.allclose(w_a.forward(z), w_b.forward(z)), \
        "same seed must give identical forward"
    assert not torch.allclose(w_a.forward(z), w_c.forward(z)), \
        "different seed must give different forward"
