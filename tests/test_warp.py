import numpy as np
import torch

from latent_smoothness.warp.quantize import posthoc_vq
from latent_smoothness.warp.scramble import bijective_scramble
from latent_smoothness.warp.flow_warp import FlowWarp
from tests.conftest import SEED


# --- quantize ---

def test_quantization_error_positive(unit_gaussian_latents):
    q, _ = posthoc_vq(unit_gaussian_latents, codebook_size=64)
    assert torch.norm(unit_gaussian_latents - q) > 0, \
        "Expected nonzero quantization error"


def test_quantization_error_decreases_with_K():
    # Use N=512 here: the conftest fixture has N=128, which is < K=256 and would
    # make sklearn raise ValueError. A local tensor with N > 256 is required.
    torch.manual_seed(SEED)
    z = torch.randn(512, 4)
    errors = []
    for K in [16, 64, 256]:
        q, _ = posthoc_vq(z, codebook_size=K)
        errors.append(torch.norm(z - q).item())
    assert errors[0] > errors[1] > errors[2], \
        f"Quantization error should decrease with larger K: {errors}"


# --- scramble ---

def test_scramble_round_trip_encode_decode(unit_gaussian_latents):
    q, _ = posthoc_vq(unit_gaussian_latents, codebook_size=32)
    _, encode_fn, decode_fn = bijective_scramble(unit_gaussian_latents, codebook_size=32)
    reconstructed = decode_fn(encode_fn(q))
    assert torch.allclose(reconstructed, q, atol=1e-6), \
        f"decode_fn(encode_fn(q)) != q, max diff={( reconstructed - q).abs().max()}"


def test_scramble_round_trip_decode_encode(unit_gaussian_latents):
    q, _ = posthoc_vq(unit_gaussian_latents, codebook_size=32)
    _, encode_fn, decode_fn = bijective_scramble(unit_gaussian_latents, codebook_size=32)
    reconstructed = encode_fn(decode_fn(q))
    assert torch.allclose(reconstructed, q, atol=1e-6), \
        f"encode_fn(decode_fn(q)) != q, max diff={(reconstructed - q).abs().max()}"


def test_scramble_quantization_error_present(unit_gaussian_latents):
    q, _ = posthoc_vq(unit_gaussian_latents, codebook_size=32)
    assert torch.norm(unit_gaussian_latents - q) > 0, \
        "Quantization error should be present before round-trip"


# --- flow_warp ---

def test_flow_warp_round_trip(unit_gaussian_latents):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    warp = FlowWarp()
    warp.fit(unit_gaussian_latents, target="mixture_of_gaussians", n_epochs=500)
    z = unit_gaussian_latents.float()
    with torch.no_grad():
        z_warped = warp.forward(z)
        z_reconstructed = warp.inverse(z_warped)
    # Relative tolerance: the round trip is exact in exact arithmetic, but 16
    # float32 coupling layers with exp-scaling amplify rounding for tail samples.
    rel = torch.norm(z_reconstructed - z, dim=-1) / (torch.norm(z, dim=-1) + 1e-6)
    assert (rel < 1e-3).all(), \
        f"FlowWarp round-trip error too large: max rel err={rel.max():.2e}"


def test_flow_warp_warps_to_target(unit_gaussian_latents):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    warp = FlowWarp()
    warp.fit(unit_gaussian_latents, target="mixture_of_gaussians", n_epochs=500)
    result = warp.verify(unit_gaussian_latents)
    # Warped samples must resemble the target more than the original z.
    # The old Gaussian-normalising implementation inverts this: kl_to_target >> kl_to_original.
    assert result.kl_to_target < result.kl_to_original, (
        f"Warped z should be closer to target than to original: "
        f"kl_to_target={result.kl_to_target:.3f}, kl_to_original={result.kl_to_original:.3f}"
    )
