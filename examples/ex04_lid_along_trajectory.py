"""ex04: Local intrinsic dimensionality — a 1-D curve, a 2-D Gaussian, and the toy VAE latents."""
import math

import torch

from _common import plt, save
from latent_smoothness import compute_lid
from toy_models import get_toy_models

torch.manual_seed(0)
N = 512

# A 1-D curve (a spiral arc) embedded in 3-D, a full 3-D Gaussian, and the
# 2-D toy latents.
s = torch.rand(N) * 3 * math.pi
curve = torch.stack([s.cos() * (1 + 0.1 * s), s.sin() * (1 + 0.1 * s), 0.2 * s], dim=1)
gauss3 = torch.randn(N, 3)
_, _, z_toy, _ = get_toy_models()
z_toy = z_toy[torch.randperm(z_toy.shape[0])[:N]]

cases = [("1-D spiral in 3-D", curve, 1), ("3-D Gaussian", gauss3, 3), ("toy VAE latents", z_toy, 2)]
results = {name: compute_lid(x) for name, x, _ in cases}

fig = plt.figure(figsize=(13, 3.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.4, 1])
ax = fig.add_subplot(gs[0], projection="3d")
axes = [ax, fig.add_subplot(gs[1]), fig.add_subplot(gs[2])]
lid_curve = results["1-D spiral in 3-D"].lid_mle
sc = ax.scatter(*curve.T, c=lid_curve, cmap="viridis", s=6, vmin=0, vmax=3)
ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
ax.set_title("1-D spiral, coloured by per-point\nMLE LID (dark 0 → yellow 3)")

order = s.argsort()
mle_sorted = results["1-D spiral in 3-D"].lid_mle[order]
axes[1].plot(s[order], mle_sorted, lw=0.6, color="C0", alpha=0.5, label="MLE (k=10)")
rolling = mle_sorted.unfold(0, 25, 1).median(dim=1).values
axes[1].plot(s[order][12:12 + rolling.shape[0]], rolling, lw=2, color="C0", label="MLE, rolling median")
axes[1].scatter(s[order], results["1-D spiral in 3-D"].lid_twonn[order], s=4, color="0.6",
                label="TwoNN (per point, noisy)")
axes[1].axhline(1.0, color="C3", ls="--", lw=1, label="true dimension")
axes[1].set_ylim(0, 4)
axes[1].set_xlabel("arc parameter $s$ along the spiral")
axes[1].set_ylabel("local intrinsic dimension")
axes[1].set_title("Per-point estimates along the curve")
axes[1].legend(loc="upper right")

x = torch.arange(len(cases)).float()
axes[2].bar(x - 0.2, [results[n].mean_lid_twonn for n, _, _ in cases], 0.38, label="TwoNN (global)")
axes[2].bar(x + 0.2, [results[n].mean_lid_mle for n, _, _ in cases], 0.38, label="MLE (mean)")
axes[2].scatter(x, [d for _, _, d in cases], marker="_", s=900, color="C3", zorder=3,
                label="true dimension")
axes[2].set_xticks(x.tolist(), [n.replace(" in 3-D", "\nin 3-D").replace(" latents", "\nlatents")
                                for n, _, _ in cases])
axes[2].set_ylabel("intrinsic dimension")
axes[2].set_title("Global estimates")
axes[2].set_ylim(0, 4.5)
axes[2].legend(fontsize=8, loc="upper left", markerscale=0.4)
fig.suptitle("ex04  Local intrinsic dimensionality (TwoNN and Levina–Bickel MLE)")
save(fig, "ex04_lid_along_trajectory")

ok = all(abs(results[n].mean_lid_twonn - d) < 0.5 for n, _, d in cases)
print("TwoNN LID: " + ", ".join(f"{n} {results[n].mean_lid_twonn:.2f} (true {d})" for n, _, d in cases)
      + "; all within 0.5: " + ("PASS" if ok else "FAIL"))
assert ok
