"""FlowWarp save/load, buffer registration, ensemble, and set_global_seed.

The headline regression is the standardized round-trip: the per-dim mean/std
must be registered buffers, otherwise state_dict() drops them and a reloaded
standardized flow silently loses its normalisation. Few epochs are used throughout —
the save/load round-trip is exact regardless of training quality.
"""
import random

import numpy as np
import torch

from latent_smoothness.utils import set_global_seed
from latent_smoothness.warp.flow_warp import FlowWarp, fit_flow_ensemble


def _fixed_latents(n=128, dz=4, seed=0):
    """A fixed, reproducible batch of latents to fit on."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, dz, generator=g).float()


def _fit(z, standardize=False, n_epochs=50, seed=0):
    set_global_seed(seed)
    warp = FlowWarp()
    warp.fit(z, target="mixture_of_gaussians", n_epochs=n_epochs, standardize=standardize)
    return warp


# --- save / load round-trips ------------------------------------------------

def test_save_load_roundtrip_unstandardized(tmp_path):
    """forward / inverse outputs are identical before save and after load."""
    z = _fixed_latents()
    warp = _fit(z, standardize=False)

    with torch.no_grad():
        fwd_before = warp.forward(z)
        inv_before = warp.inverse(fwd_before)

    path = tmp_path / "flow.pt"
    warp.save(path)
    loaded = FlowWarp.load(path)

    assert loaded._z_mean is None and loaded._z_std is None, \
        "unstandardized load must keep mean/std as None"
    with torch.no_grad():
        fwd_after = loaded.forward(z)
        inv_after = loaded.inverse(fwd_after)

    assert torch.allclose(fwd_before, fwd_after, atol=1e-6), \
        f"forward changed across save/load: max diff={(fwd_before - fwd_after).abs().max():.2e}"
    assert torch.allclose(inv_before, inv_after, atol=1e-6), \
        f"inverse changed across save/load: max diff={(inv_before - inv_after).abs().max():.2e}"


def test_save_load_roundtrip_standardized(tmp_path):
    """The bug-fix regression: a standardized flow's mean/std survive save/load
    and the warp behaves identically afterwards."""
    scale = torch.tensor([10.0, 0.1, 5.0, 0.2])
    z = _fixed_latents() * scale + 3.0
    warp = _fit(z, standardize=True)
    assert warp._z_mean is not None  # standardization actually stored

    with torch.no_grad():
        fwd_before = warp.forward(z)

    path = tmp_path / "flow_std.pt"
    warp.save(path)
    loaded = FlowWarp.load(path)

    assert loaded._z_mean is not None, "standardization stats dropped on load (the bug)"
    assert torch.allclose(loaded._z_mean, warp._z_mean), "loaded _z_mean differs from original"
    assert torch.allclose(loaded._z_std, warp._z_std), "loaded _z_std differs from original"

    with torch.no_grad():
        fwd_after = loaded.forward(z)
    assert torch.allclose(fwd_before, fwd_after, atol=1e-6), \
        f"standardized forward changed across save/load: max diff={(fwd_before - fwd_after).abs().max():.2e}"


def test_state_dict_contains_buffers():
    """After a standardized fit the three previously-dropped tensors are now
    registered buffers and therefore appear in state_dict()."""
    z = _fixed_latents() * torch.tensor([3.0, 0.5, 2.0, 1.0])
    warp = _fit(z, standardize=True)
    keys = warp.state_dict().keys()
    for name in ("_z_mean", "_z_std", "_target_samples"):
        assert name in keys, f"{name} missing from state_dict(): {sorted(keys)}"


# --- ensemble ---------------------------------------------------------------

def test_ensemble_distinct_and_loadable(tmp_path):
    """fit_flow_ensemble writes one checkpoint per seed; each reloads to a
    working warp, and the members are not all identical."""
    z = _fixed_latents()
    paths = fit_flow_ensemble(z, n=3, seeds=[0, 1, 2], out_dir=tmp_path, n_epochs=30)
    assert len(paths) == 3 and all(p.exists() for p in paths), \
        f"expected 3 checkpoints on disk, got {paths}"

    warps = [FlowWarp.load(p) for p in paths]
    for w in warps:
        out = w.forward(z)
        assert torch.isfinite(out).all(), "reloaded ensemble member produced non-finite forward"

    # At least two members must differ on some _flow_z weight (distinct seeds
    # → distinct init/training trajectory).
    p0 = dict(warps[0]._flow_z.named_parameters())
    p1 = dict(warps[1]._flow_z.named_parameters())
    differs = any(not torch.allclose(p0[k], p1[k]) for k in p0)
    assert differs, "seed 0 and seed 1 produced identical flow weights — seeds had no effect"


# --- determinism ------------------------------------------------------------

def test_set_global_seed_reproducible():
    """set_global_seed makes python/numpy/torch RNG draws reproducible."""
    set_global_seed(123)
    t1 = torch.rand(5)
    n1 = np.random.rand(5)
    p1 = [random.random() for _ in range(5)]

    set_global_seed(123)
    t2 = torch.rand(5)
    n2 = np.random.rand(5)
    p2 = [random.random() for _ in range(5)]

    assert torch.equal(t1, t2), "torch RNG not reproducible after set_global_seed"
    assert np.array_equal(n1, n2), "numpy RNG not reproducible after set_global_seed"
    assert p1 == p2, "python random not reproducible after set_global_seed"
