"""The uniform Warp interface across all warps.

Covers conformance (attributes/methods/shape), the straight-through gradient for
the quantized warps, save/load fidelity, and device-preservation. Flow fits use
few epochs — these tests check interface mechanics, not warp quality.
"""
import pytest
import torch

from latent_smoothness.warp import build_warp
from latent_smoothness.warp.adapters import VQWarp
from tests.conftest import SEED

N, DZ = 128, 4
CODEBOOK = 16


def _fit_data():
    g = torch.Generator().manual_seed(SEED)
    return torch.randn(N, DZ, generator=g).float()


# fixed labels aligned to the (deterministic) _fit_data() — for the label warps.
_LABELS = torch.arange(N) % 3

# name -> kwargs for build_warp (kept small/fast)
WARP_KW = {
    "vq": dict(codebook_size=CODEBOOK, seed=0),
    "scramble": dict(codebook_size=CODEBOOK, seed=0),
    "flow_warp": dict(n_epochs=30, seed=0),
    "piecewise_affine": dict(bins_per_dim=4, seed=0),
    "class_scramble": dict(codebook_size=CODEBOOK, labels=_LABELS, class_mode="within", seed=0),
    "class_collapse": dict(labels=_LABELS, n_directions=1),
}
EXPECTED = {
    # name: (is_invertible, is_differentiable)
    "vq": (False, False),
    "scramble": (False, False),
    "flow_warp": (True, True),
    "piecewise_affine": (True, True),
    "class_scramble": (False, False),
    "class_collapse": (False, True),
}


def _build(name):
    return build_warp(name, _fit_data(), **WARP_KW[name])


@pytest.mark.parametrize("name", list(WARP_KW))
def test_all_warps_conform(name):
    """Every warp exposes the interface attributes/methods and forward(z)
    returns a finite tensor of the right shape."""
    warp = _build(name)
    assert warp.name == name
    inv, diff = EXPECTED[name]
    assert warp.is_invertible is inv
    assert warp.is_differentiable is diff
    assert warp.needs_ste is (not diff)

    z = _fit_data()
    out = warp.forward(z)
    assert out.shape == z.shape, f"{name}: forward changed shape {out.shape} != {z.shape}"
    assert torch.isfinite(out).all(), f"{name}: forward produced non-finite values"

    # backproject and forward_ste are callable and shape-preserving.
    assert warp.backproject(out).shape == z.shape
    assert warp.forward_ste(z).shape == z.shape


def test_ste_identity_gradient_for_quantized():
    """For a non-differentiable warp, forward_ste is the identity surrogate:
    d/dz sum(forward_ste(z)) == 1 everywhere."""
    warp = VQWarp(_fit_data(), codebook_size=CODEBOOK)
    z = _fit_data().clone().requires_grad_(True)
    warp.forward_ste(z).sum().backward()
    assert z.grad is not None
    assert torch.allclose(z.grad, torch.ones_like(z.grad)), \
        "forward_ste must pass an identity gradient (all ones) for quantized warps"


@pytest.mark.parametrize("name", list(WARP_KW))
def test_save_load_each(name, tmp_path):
    """Each warp's forward is identical before save and after load."""
    warp = _build(name)
    z = _fit_data()
    with torch.no_grad():
        before = warp.forward(z)

    path = tmp_path / f"{name}.pt"
    warp.save(path)
    loaded = type(warp).load(path)

    with torch.no_grad():
        after = loaded.forward(z)
    assert torch.allclose(before, after, atol=1e-6), \
        f"{name}: forward changed across save/load (max diff {(before - after).abs().max():.2e})"


@pytest.mark.parametrize("name", list(WARP_KW))
def test_warp_device_preserving(name):
    """forward / backproject return on the input's device (CPU case; GPU is
    exercised by downstream GPU usage)."""
    warp = _build(name)
    z = _fit_data()
    out = warp.forward(z)
    assert out.device == z.device, f"{name}: forward moved off input device"
    back = warp.backproject(out)
    assert back.device == z.device, f"{name}: backproject moved off input device"


def test_build_warp_names_and_errors():
    from latent_smoothness.warp import WARP_NAMES
    assert set(WARP_NAMES) == set(WARP_KW)
    with pytest.raises(ValueError, match="Unknown warp name"):
        build_warp("nope", _fit_data())
    with pytest.raises(TypeError):
        build_warp("vq", _fit_data(), codebook_size=CODEBOOK, not_an_arg=1)


def test_scramble_warp_matches_functional_scramble():
    """ScrambleWarp and bijective_scramble share one construction."""
    from latent_smoothness.warp import bijective_scramble
    z = _fit_data()
    _, encode_fn, decode_fn = bijective_scramble(z, CODEBOOK, seed=3)
    warp = build_warp("scramble", z, codebook_size=CODEBOOK, seed=3)
    assert torch.equal(warp.forward(z), encode_fn(z))
    assert torch.equal(warp.backproject(z), decode_fn(z))
