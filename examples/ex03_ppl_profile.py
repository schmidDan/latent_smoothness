"""ex03: Perceptual path length — an image decoder with a narrow rough shell at ||z|| = 1."""
import torch
import torch.nn as nn

from _common import plt, save
from latent_smoothness import compute_ppl

torch.manual_seed(0)
DZ, M, R0, R1 = 4, 8, 0.5, 2.0


class RoughSpatialDecoder(nn.Module):
    """Linear 8x8 image decoder of a radially rescaled z; sharp bump in the scale at ||z|| = 1."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(DZ, 64, bias=False)
        self.unflatten = nn.Unflatten(-1, (1, 8, 8))

    def forward(self, z):
        scale = 1.0 + 10.0 * torch.exp(-1000 * (z.norm(dim=-1, keepdim=True) - 1.0) ** 2)
        return self.unflatten(self.linear(z * scale))


decoder = RoughSpatialDecoder()
directions = torch.randn(M, DZ)
directions = directions / directions.norm(dim=-1, keepdim=True)
z_start, z_end = directions * R0, directions * R1          # radial paths ||z||: 0.5 -> 2.0

result = compute_ppl(decoder, z_start, z_end, n_steps=100, device="cpu")
T = result.ppl_values.shape[1]
radius = R0 + (R1 - R0) * (torch.arange(T) + 0.5) / T        # ||z|| at each segment midpoint

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), width_ratios=[2, 1])
for i in range(M):
    axes[0].plot(radius, result.ppl_values[i], lw=1, alpha=0.8)
spikes = result.ppl_spikes.nonzero()
axes[0].scatter(radius[spikes[:, 1]], result.ppl_values[spikes[:, 0], spikes[:, 1]],
                color="red", zorder=3, s=25, label="detected spike")
axes[0].set_yscale("log")
axes[0].set_xlabel(r"$\|z\|$ along the radial path")
axes[0].set_ylabel("LPIPS between consecutive frames")
axes[0].set_title(f"Per-segment perceptual distance ({M} paths)")
axes[0].legend()

norms = result.spike_z_locations.norm(dim=-1)
axes[1].hist(norms, bins=20, range=(R0, R1), color="red", alpha=0.7)
axes[1].axvline(1.0, color="0.3", ls="--", lw=1, label=r"rough shell $\|z\|=1$")
axes[1].set_xlabel(r"$\|z\|$ at detected spike")
axes[1].set_ylabel("count")
axes[1].set_title("Spike locations")
axes[1].legend()
fig.suptitle("ex03  Perceptual path length: spikes flag where the decoded images change abruptly")
save(fig, "ex03_ppl_profile")

ok = int(result.ppl_spikes.sum()) > 0 and bool(((norms - 1.0).abs() < 0.1).all())
print(f"{int(result.ppl_spikes.sum())} PPL spikes, all within 0.1 of ||z||=1: " + ("PASS" if ok else "FAIL"))
assert ok
