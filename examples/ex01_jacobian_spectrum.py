"""ex01: Jacobian spectrum — a smooth decoder vs. one with a sharp radial bump at ||z|| = 1."""
import torch
import torch.nn as nn

from _common import plt, save
from latent_smoothness import compute_jacobian_spectrum

torch.manual_seed(0)
DZ, DX, N = 4, 64, 256


class SmoothDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(DZ, 32), nn.Tanh(), nn.Linear(32, DX), nn.Tanh())

    def forward(self, z):
        return self.net(z)


class RoughDecoder(nn.Module):
    """Linear map of a radially rescaled z; the rescaling has a narrow bump at ||z|| = 1."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(DZ, DX, bias=False)

    def forward(self, z):
        scale = 1.0 + 10.0 * torch.exp(-100 * (z.norm(dim=-1, keepdim=True) - 1.0) ** 2)
        return self.linear(z * scale)


z = torch.randn(N, DZ) * 0.7
smooth, rough = SmoothDecoder(), RoughDecoder()
res_s = compute_jacobian_spectrum(smooth, z)
res_r = compute_jacobian_spectrum(rough, z)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
idx = torch.arange(1, DZ + 1)
for res, name, c in [(res_s, "smooth", "C0"), (res_r, "rough", "C3")]:
    axes[0].errorbar(idx, res.mean_spectrum, yerr=res.spectral_variation, marker="o",
                     capsize=3, color=c, label=name)
axes[0].set_yscale("log")
axes[0].set_xticks(idx.tolist())
axes[0].set_xlabel("singular value index")
axes[0].set_ylabel("singular value (mean ± std over points)")
axes[0].set_title("Jacobian spectrum")
axes[0].legend()

bars = axes[1].bar(["smooth", "rough"], [res_s.lipschitz_estimate, res_r.lipschitz_estimate],
                   color=["C0", "C3"])
axes[1].bar_label(bars, fmt="%.1f")
axes[1].set_yscale("log")
axes[1].set_title("Lipschitz estimate (max singular value)")

norms = z.norm(dim=-1)
axes[2].scatter(norms, res_s.singular_values[:, 0], s=6, color="C0", label="smooth")
axes[2].scatter(norms, res_r.singular_values[:, 0], s=6, color="C3", label="rough")
axes[2].axvline(1.0, color="0.5", ls="--", lw=1)
axes[2].set_yscale("log")
axes[2].set_xlabel("$\\|z\\|$")
axes[2].set_ylabel("largest singular value")
axes[2].set_title("Where the decoder is rough")
axes[2].legend()
fig.suptitle("ex01  Decoder Jacobian spectrum: the rough decoder's Jacobian blows up near $\\|z\\|=1$")
save(fig, "ex01_jacobian_spectrum")

ok = res_r.lipschitz_estimate > 2 * res_s.lipschitz_estimate
near = (norms - 1).abs() < 0.05
ok &= bool(res_r.singular_values[near, 0].mean() > 2 * res_r.singular_values[~near, 0].median())
print(f"Rough Lipschitz ({res_r.lipschitz_estimate:.2f}) > 2x smooth ({res_s.lipschitz_estimate:.2f}), "
      f"peak located at ||z||=1: " + ("PASS" if ok else "FAIL"))
assert ok
