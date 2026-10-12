"""ex03: PPL profile — rough_spatial inline decoder with known spike at ||z||=1."""
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from latent_smoothness import compute_ppl

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

torch.manual_seed(0)

class RoughSpatialDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 64, bias=False)
        self.unflatten = nn.Unflatten(-1, (1, 8, 8))
    def forward(self, z):
        scale = 1.0 + 10.0 * torch.exp(-1000 * (z.norm(dim=-1, keepdim=True) - 1.0) ** 2)
        return self.unflatten(self.linear(z * scale))

decoder = RoughSpatialDecoder()
DZ, M = 4, 8
z = torch.randn(32, DZ)
z_dir = z[:M] / z[:M].norm(dim=-1, keepdim=True)
z_start = z_dir * 0.5
z_end = z_dir * 2.0

result = compute_ppl(decoder, z_start, z_end, n_steps=100, device="cpu")

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i in range(min(4, M)):
    t = torch.linspace(0, 1, result.ppl_values.shape[1]).numpy()
    axes[0].plot(t, result.ppl_values[i].detach().numpy(), alpha=0.7, label=f"path {i}")
    spikes = result.ppl_spikes[i].nonzero().squeeze(-1)
    for s in spikes:
        axes[0].axvline(t[s], color="red", lw=0.8, alpha=0.6)
axes[0].set_title("PPL profile (red=spike)")
axes[0].set_xlabel("t"); axes[0].set_ylabel("LPIPS")
axes[0].legend(fontsize=7)

if result.spike_z_locations.numel() > 0:
    norms = result.spike_z_locations.norm(dim=-1).detach().numpy()
    axes[1].scatter(range(len(norms)), norms, s=40)
    axes[1].axhline(1.0, color="red", lw=2, label="||z||=1.0")
    axes[1].set_title("Spike z-location norms")
    axes[1].set_xlabel("Spike index"); axes[1].set_ylabel("||z||"); axes[1].legend()

fig.tight_layout()
fig.savefig(OUT / "ex03_ppl_profile.png", dpi=80)
plt.close()

n_spikes = int(result.ppl_spikes.sum())
print(f"PPL spike detected at expected location "
      + ("PASS" if n_spikes > 0 else "FAIL (no spikes found)"))
assert n_spikes > 0
