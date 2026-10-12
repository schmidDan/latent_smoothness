"""Regression tests for FlowWarp's non-finite guard and optional z-scoring.

These cover:
  * verify() must tolerate non-finite warped samples (drop them, or report
    NaN if the warp fully diverged) instead of raising in the kNN-KL estimator.
  * fit(standardize=True) must z-score the z-flow input and undo it exactly in
    inverse(), while standardize=False (the default) changes nothing.
"""
import numpy as np
import torch

from latent_smoothness.warp.flow_warp import FlowWarp
from tests.conftest import SEED


def _fit_small(z, standardize=False, target="mixture_of_gaussians"):
    """Fit with few epochs — the round-trip is exact by bijectivity regardless
    of training quality, so 50 epochs keeps the test fast."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    warp = FlowWarp()
    warp.fit(z.float(), target=target, n_epochs=50, standardize=standardize)
    return warp


# --- z-scoring option -------------------------------------------------------

def test_standardize_default_is_off(unit_gaussian_latents):
    """Default fit() stores no mean/std, i.e. behaviour is unchanged."""
    warp = _fit_small(unit_gaussian_latents, standardize=False)
    assert warp._z_mean is None and warp._z_std is None, \
        "standardize must default to off (no stored normalisation stats)"


def test_standardize_roundtrip_recovers_input(unit_gaussian_latents):
    """With heavy per-dim scaling (the overflow-prone case), z-scoring must
    still give an exact inverse: inverse(forward(z)) == z."""
    scale = torch.tensor([10.0, 0.1, 5.0, 0.2])
    z = unit_gaussian_latents.float() * scale + 3.0

    warp = _fit_small(z, standardize=True)
    assert warp._z_mean is not None and tuple(warp._z_mean.shape) == (1, z.shape[1])
    assert warp._z_std is not None and bool((warp._z_std > 0).all())

    with torch.no_grad():
        z_rt = warp.inverse(warp.forward(z))
    rel = torch.norm(z_rt - z, dim=-1) / (torch.norm(z, dim=-1) + 1e-6)
    assert (rel < 1e-3).all(), \
        f"z-scored round-trip must recover input: max rel err={rel.max():.2e}"


# --- non-finite guard in verify() ------------------------------------------

def test_verify_drops_nonfinite_and_returns_finite_kl(unit_gaussian_latents, monkeypatch):
    """A few inf/NaN warped rows must be dropped, not crash NearestNeighbors."""
    z = unit_gaussian_latents.float()
    warp = _fit_small(z)

    corrupt = warp.forward(z).detach().clone()
    corrupt[2] = float("inf")     # whole-row inf
    corrupt[5, 1] = float("nan")  # single-element NaN
    monkeypatch.setattr(warp, "forward", lambda _z: corrupt)

    result = warp.verify(z)  # must not raise
    assert np.isfinite(result.kl_to_target), "kl_to_target should be finite after dropping bad rows"
    assert np.isfinite(result.kl_to_original), "kl_to_original should be finite after dropping bad rows"
    assert result.sample_plot_data.shape[0] == corrupt.shape[0] - 2, \
        "plot data should contain exactly the surviving (finite) rows"
    assert result.n_input == corrupt.shape[0] and result.n_dropped == 2, \
        f"drop counts must be reported: n_input={result.n_input}, n_dropped={result.n_dropped}"


def test_verify_all_nonfinite_returns_nan(unit_gaussian_latents, monkeypatch):
    """If essentially everything is non-finite the warp diverged — report NaN
    KLs gracefully rather than throwing."""
    z = unit_gaussian_latents.float()
    warp = _fit_small(z)

    allbad = torch.full_like(warp.forward(z), float("nan"))
    monkeypatch.setattr(warp, "forward", lambda _z: allbad)

    result = warp.verify(z)  # must not raise
    assert np.isnan(result.kl_to_target) and np.isnan(result.kl_to_original), \
        "Fully diverged warp should yield NaN KLs, not a crash"
    assert result.n_dropped == result.n_input == allbad.shape[0], \
        "a fully diverged warp must report all rows dropped"


def test_training_that_never_succeeds_raises(unit_gaussian_latents, monkeypatch):
    """If every training step fails, fit() must raise instead of returning an
    untrained flow."""
    import pytest
    import normflows as nf

    def always_nan(self, x):
        return torch.tensor(float("nan"), requires_grad=True)
    monkeypatch.setattr(nf.NormalizingFlow, "forward_kld", always_nan)
    with pytest.raises(RuntimeError, match="Flow training failed"):
        FlowWarp().fit(unit_gaussian_latents.float(), n_epochs=3)


def test_all_named_targets_fit(unit_gaussian_latents):
    for target in ("mixture_of_gaussians", "banana", "uniform", "random_affine_cells"):
        warp = _fit_small(unit_gaussian_latents, target=target)
        assert torch.isfinite(warp.forward(unit_gaussian_latents.float())).all(), target
