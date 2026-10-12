from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from latent_smoothness.utils import eval_mode, linear_interpolate, get_lpips, resolve_device


@dataclass
class PPLResult:
    ppl_values: Tensor
    mean_ppl: float
    ppl_spikes: Tensor
    spike_z_locations: Tensor
    spike_path_indices: Tensor


def _prep_for_lpips(x: Tensor, global_scale: float) -> Tensor:
    """Normalise to [-1,1] using a shared scale and upsample to ≥64x64 for LPIPS."""
    x = (x / global_scale).clamp(-1.0, 1.0)
    H, W = x.shape[2], x.shape[3]
    if H < 64 or W < 64:
        new_H = max(int(H * max(64 / H, 64 / W)), 64)
        new_W = max(int(W * max(64 / H, 64 / W)), 64)
        x = F.interpolate(x, size=(new_H, new_W), mode="bilinear", align_corners=False)
    if x.shape[1] == 1:
        x = x.expand(-1, 3, -1, -1)
    return x


def _compute_ppl_raw(
    decoder: nn.Module,
    z_start: Tensor,
    z_end: Tensor,
    n_steps: int = 100,
    lpips_net: str = "vgg",
    device: Optional[str] = None,
) -> PPLResult:
    device = resolve_device(device)
    lpips_fn = get_lpips(lpips_net, device)
    N, dz = z_start.shape

    z_path = linear_interpolate(z_start, z_end, n_steps)  # (N, n_steps, dz)

    # Pre-decode all frames; compute a single global scale for consistent normalisation
    with torch.no_grad():
        all_frames = []
        for step in range(n_steps):
            all_frames.append(decoder(z_path[:, step, :]))
        global_scale = float(
            torch.stack(all_frames, dim=0).abs().max().clamp(min=1.0).item()
        )

    ppl_values = []
    for i in range(n_steps - 1):
        x_a = _prep_for_lpips(all_frames[i].to(device), global_scale)
        x_b = _prep_for_lpips(all_frames[i + 1].to(device), global_scale)
        with torch.no_grad():
            d = lpips_fn(x_a, x_b).reshape(N)
        ppl_values.append(d.cpu())

    ppl_values = torch.stack(ppl_values, dim=1)  # (N, n_steps-1)

    all_vals = ppl_values.reshape(-1)
    mean_val = all_vals.mean().item()

    # Spike detection: for each path, find the interior global-maximum LPIPS segment.
    # Mark it as a spike iff it exceeds the median of its surrounding context windows
    # by ≥ 3×. The wide context window (10% of path) compares the peak against the
    # "quiet" baseline far from the spike, giving high contrast for rough decoders
    # while ignoring smooth monotone variation in linear decoders.
    T = ppl_values.shape[1]
    ppl_spikes = torch.zeros_like(ppl_values, dtype=torch.bool)
    WINDOW = max(T // 10, 5)
    if T >= 2 * WINDOW + 1:
        for i in range(N):
            vals = ppl_values[i]
            interior = vals[WINDOW: T - WINDOW]
            peak_local = int(interior.argmax().item())
            peak_global = peak_local + WINDOW
            left_ctx = vals[max(0, peak_global - WINDOW): peak_global]
            right_ctx = vals[peak_global + 1: min(T, peak_global + WINDOW + 1)]
            if left_ctx.numel() == 0 or right_ctx.numel() == 0:
                continue
            baseline = float(max(
                torch.median(left_ctx).item(),
                torch.median(right_ctx).item(),
                1e-10,
            ))
            if float(vals[peak_global].item()) > baseline * 3.0:
                ppl_spikes[i, peak_global] = True

    spike_indices = ppl_spikes.nonzero(as_tuple=False)

    if spike_indices.shape[0] == 0:
        spike_z_locations = torch.zeros(0, dz, device=z_start.device, dtype=z_start.dtype)
        spike_path_indices = torch.zeros(0, dtype=torch.long, device=z_start.device)
    else:
        path_idx = spike_indices[:, 0]
        step_idx = spike_indices[:, 1]
        spike_z_locations = 0.5 * (
            z_path[path_idx, step_idx, :] + z_path[path_idx, step_idx + 1, :]
        )
        spike_path_indices = path_idx

    return PPLResult(
        ppl_values=ppl_values,
        mean_ppl=mean_val,
        ppl_spikes=ppl_spikes,
        spike_z_locations=spike_z_locations,
        spike_path_indices=spike_path_indices,
    )


def compute_ppl(
    decoder: nn.Module,
    z_start: Tensor,
    z_end: Tensor,
    n_steps: int = 100,
    lpips_net: str = "vgg",
    device: Optional[str] = None,
) -> PPLResult:
    """Per-segment LPIPS distance along straight latent paths, with spike detection.

    The decoder must produce images ``(B, C, H, W)``. ``device`` is where LPIPS
    runs (``None`` = CUDA if available, else CPU). A segment is flagged as a
    spike if it is the interior maximum of its path and exceeds 3x the median
    of a +/-10% context window on both sides.
    """
    with eval_mode(decoder):
        with torch.no_grad():
            test_out = decoder(z_start[0:1])
        if test_out.ndim != 4:
            raise ValueError(
                f"compute_ppl requires a decoder with 4D spatial output (B, C, H, W), "
                f"but got shape {tuple(test_out.shape)}. "
                f"Wrap your decoder to produce spatial output or use a spatial decoder."
            )
        return _compute_ppl_raw(decoder, z_start, z_end, n_steps, lpips_net, device)
