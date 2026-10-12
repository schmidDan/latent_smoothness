"""ex04: LID along trajectory — 1D submanifold vs. 2D Gaussian (dz=2 inline)."""
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from latent_smoothness import compute_lid

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

torch.manual_seed(0)
N = 128
DZ = 2

# 1D submanifold: points on x-axis in R^2
coords = torch.randn(N, 1)
z_1d = torch.zeros(N, DZ)
z_1d[:, 0] = coords[:, 0]

# Full 2D Gaussian
z_2d = torch.randn(N, DZ)

res_1d = compute_lid(z_1d)
res_2d = compute_lid(z_2d)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

t = torch.linspace(-3, 3, N)
axes[0].scatter(t.numpy(), res_1d.lid_twonn.numpy(), s=10, alpha=0.7, label="TwoNN LID")
axes[0].axhline(1.0, color="red", lw=2, label="True LID=1")
axes[0].set_title("TwoNN LID along 1D submanifold")
axes[0].set_xlabel("Position on 1D line"); axes[0].set_ylabel("LID")
axes[0].legend()

cats = ["1D submanifold\n(LID≈1)", "2D Gaussian\n(LID≈2)"]
means_twonn = [res_1d.mean_lid_twonn, res_2d.mean_lid_twonn]
means_mle = [res_1d.mean_lid_mle, res_2d.mean_lid_mle]
x = range(2)
axes[1].bar([i-0.2 for i in x], means_twonn, 0.35, label="TwoNN")
axes[1].bar([i+0.2 for i in x], means_mle, 0.35, label="MLE")
axes[1].set_xticks(list(x)); axes[1].set_xticklabels(cats)
axes[1].set_title("Mean LID comparison"); axes[1].legend()
axes[1].axhline(1.0, color="orange", lw=1, ls="--")
axes[1].axhline(2.0, color="blue", lw=1, ls="--")

fig.tight_layout()
fig.savefig(OUT / "ex04_lid_along_trajectory.png", dpi=80)
plt.close()

ok = abs(res_1d.mean_lid_twonn - 1.0) < 1.0
print(f"LID on-manifold ~= 1.0: {res_1d.mean_lid_twonn:.2f} "
      + ("PASS" if ok else "FAIL"))
assert ok
