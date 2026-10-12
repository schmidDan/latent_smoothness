"""ex07: VQ vs. bijective scramble — information is preserved, the class arrangement is not."""
import torch

from _common import plt, save, latent_grid, class_regions, class_scatter
from latent_smoothness.warp.quantize import posthoc_vq
from latent_smoothness.warp.scramble import bijective_scramble
from toy_models import get_toy_models

torch.manual_seed(0)
decoder, classifier, z_train, y_train = get_toy_models()
K = 32

z_vq, codebook = posthoc_vq(z_train, codebook_size=K)
z_scrambled, encode_fn, decode_fn = bijective_scramble(z_train, codebook_size=K)
z_back = decode_fn(z_scrambled)                       # exact inverse on codebook points
residual = (encode_fn(z_back) - z_scrambled).norm(dim=-1).max().item()


def accuracy(z):
    with torch.no_grad():
        return (classifier(z).argmax(dim=-1) == y_train).float().mean().item()


acc = {"original": accuracy(z_train), "VQ": accuracy(z_vq),
       "scrambled": accuracy(z_scrambled), "backprojected": accuracy(z_back)}

# Voronoi cells of the codebook, for the background of the VQ panel.
gx, gy, pts = latent_grid(z_train, n=300, pad=0.2)
cells = torch.cdist(pts, codebook).argmin(dim=1).reshape(gx.shape)

fig, axes = plt.subplots(1, 5, figsize=(19, 4), width_ratios=[1, 1, 1, 1, 0.8])
panels = [
    (axes[0], z_train, "Original latents"),
    (axes[1], z_vq, f"VQ: snapped to {K} K-means cells"),
    (axes[2], z_scrambled, "Scramble: cells permuted"),
    (axes[3], z_back, "Backprojected: decode(scramble)"),
]
for ax, z, title in panels:
    class_regions(ax, classifier, z_train, alpha=0.12)
    if ax is axes[1]:
        ax.contour(gx, gy, cells, levels=torch.arange(K).float() + 0.5, colors="0.6", linewidths=0.4)
    jitter = 0.0 if ax is axes[0] else 0.06 * torch.randn_like(z)   # so stacked points stay visible
    class_scatter(ax, z + jitter, y_train, s=3, alpha=0.5)
    key = {0: "original", 1: "VQ", 2: "scrambled", 3: "backprojected"}[list(axes).index(ax)]
    ax.set_title(f"{title}\nclassifier accuracy {acc[key]:.2f}")

bars = axes[4].bar(list(acc), list(acc.values()), color=["0.4", "C0", "C3", "C2"])
axes[4].bar_label(bars, fmt="%.2f")
axes[4].axhline(1 / 3, color="0.5", ls="--", lw=1, label="chance (3 classes)")
axes[4].set_ylim(0, 1.1)
axes[4].tick_params(axis="x", rotation=30)
axes[4].set_title(f"Accuracy  (round-trip residual {residual:.0e})")
axes[4].legend(loc="upper center", fontsize=8, bbox_to_anchor=(0.5, -0.25))
fig.suptitle("ex07  The scramble is a bijection on VQ cells: classes get mixed across cells, so accuracy falls to "
             "chance level or below,\nyet backprojection restores it exactly (points jittered for visibility)")
save(fig, "ex07_vq_scramble_comparison")

ok = abs(acc["backprojected"] - acc["VQ"]) < 1e-6 and residual == 0.0 and acc["scrambled"] < 0.6
print("Classifier accuracy -- " + ", ".join(f"{k}: {v:.3f}" for k, v in acc.items())
      + f"  round-trip residual {residual:.1e}  " + ("PASS" if ok else "FAIL"))
assert ok
