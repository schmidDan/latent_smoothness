"""ex01: Jacobian spectrum comparison — smooth vs. rough inline decoders."""
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from latent_smoothness import compute_jacobian_spectrum

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

torch.manual_seed(0)
DZ, DX = 4, 64
N = 64

class SmoothDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DZ, 32), nn.Tanh(), nn.Linear(32, DX), nn.Tanh())
    def forward(self, z): return self.net(z)

class RoughDecoder(nn.Module):
    def forward(self, z):
        scale = 1.0 + 10.0 * torch.exp(-100 * (z.norm(dim=-1, keepdim=True) - 1.0) ** 2)
        return z * scale

z_pts = torch.randn(N, DZ)
smooth = SmoothDecoder()
rough = RoughDecoder()

res_smooth = compute_jacobian_spectrum(smooth, z_pts)
res_rough = compute_jacobian_spectrum(rough, z_pts)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(res_smooth.mean_spectrum.detach().numpy(), label="smooth", marker="o")
axes[0].plot(res_rough.mean_spectrum.detach().numpy(), label="rough", marker="s")
axes[0].set_title("Mean Singular Value Spectrum")
axes[0].legend(); axes[0].set_xlabel("Index"); axes[0].set_ylabel("Singular value")

axes[1].bar(["smooth", "rough"],
            [res_smooth.lipschitz_estimate, res_rough.lipschitz_estimate])
axes[1].set_title("Lipschitz Estimate (max SV)")
fig.tight_layout()
fig.savefig(OUT / "ex01_jacobian_spectrum.png", dpi=80)
plt.close()

spike_detected = res_rough.lipschitz_estimate > res_smooth.lipschitz_estimate * 2
print(f"Jacobian spike detected: rough Lipschitz ({res_rough.lipschitz_estimate:.2f}) "
      f"> 2x smooth ({res_smooth.lipschitz_estimate:.2f}) "
      + ("PASS" if spike_detected else "FAIL"))
assert spike_detected
