"""ex00: The toy model used by most examples — data, latent space, reconstructions, classifier."""
import torch
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor

from _common import plt, save, class_regions, class_scatter, CLASS_COLORS
from toy_models import get_toy_models, _make_dataset, CLASS_NAMES

decoder, classifier, z_train, y_train = get_toy_models()
x, _, uv = _make_dataset()                 # same images and bump positions the models were trained on

with torch.no_grad():
    recon = decoder(z_train)
    acc = (classifier(z_train).argmax(dim=-1) == y_train).float().mean().item()
recon_mse = ((recon - x) ** 2).mean().item()
# How well do the 2-D latents determine the bump position? (cross-validated k-NN R^2)
r2 = cross_val_score(KNeighborsRegressor(n_neighbors=10), z_train.numpy(), uv.numpy(), cv=3).mean()


def mosaic(images, ncol):
    """Tile (N, 1, H, W) images into one (rows*H, ncol*W) array."""
    n, _, h, w = images.shape
    nrow = -(-n // ncol)
    pad = torch.zeros(nrow * ncol - n, 1, h, w)
    tiles = torch.cat([images, pad]).reshape(nrow, ncol, h, w)
    return tiles.permute(0, 2, 1, 3).reshape(nrow * h, ncol * w)


fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1])

# (a) training images, framed by class colour
ax = fig.add_subplot(gs[0, 0])
idx = torch.cat([torch.nonzero(y_train == c).squeeze(1)[:6] for c in range(3)])
ax.imshow(mosaic(x[idx], 6), cmap="gray")
for k, c in enumerate(y_train[idx].tolist()):
    r, col = divmod(k, 6)
    ax.add_patch(plt.Rectangle((col * 16 - 0.5, r * 16 - 0.5), 16, 16, fill=False,
                               edgecolor=CLASS_COLORS[c], lw=2))
ax.set_axis_off()
ax.set_title("Training images (frame = class:\nangle sector of the bump position)")

# (b, c) latent sheet coloured by the true generative factors
for j, (name, ax) in enumerate([("horizontal", fig.add_subplot(gs[0, 1])),
                                ("vertical", fig.add_subplot(gs[0, 2]))]):
    sc = ax.scatter(z_train[:, 0], z_train[:, 1], c=uv[:, j], cmap="coolwarm", s=3, lw=0)
    fig.colorbar(sc, ax=ax, shrink=0.8, label=f"true {name} bump position")
    ax.set_aspect("equal")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title(f"Latents coloured by {name} bump position")

# (d) reconstructions
ax = fig.add_subplot(gs[1, 0])
sel = torch.randperm(len(x), generator=torch.Generator().manual_seed(1))[:8]
ax.imshow(torch.cat([mosaic(x[sel], 8), mosaic(recon[sel], 8)], dim=0), cmap="gray")
ax.set_yticks([8, 24], ["input", "reconstruction"])
ax.set_xticks([])
ax.set_title(f"Reconstructions (pixel MSE {recon_mse:.4f})")

# (e) decoded latent grid, laid out at the latent coordinates
ax = fig.add_subplot(gs[1, 1])
n = 12
lo, hi = z_train.min(dim=0).values, z_train.max(dim=0).values
g1, g2 = torch.linspace(lo[0], hi[0], n), torch.linspace(hi[1], lo[1], n)
grid = torch.stack(torch.meshgrid(g2, g1, indexing="ij")[::-1], dim=-1).reshape(-1, 2)
with torch.no_grad():
    ax.imshow(mosaic(decoder(grid), n), cmap="gray",
              extent=[lo[0].item(), hi[0].item(), lo[1].item(), hi[1].item()])
ax.set_xlabel("$z_1$")
ax.set_ylabel("$z_2$")
ax.set_title("Decoder output across the latent space")

# (f) classifier
ax = fig.add_subplot(gs[1, 2])
class_regions(ax, classifier, z_train, alpha=0.2)
class_scatter(ax, z_train, y_train, s=3, alpha=0.5)
for c, name in enumerate(CLASS_NAMES):
    ax.scatter([], [], color=CLASS_COLORS[c], label=f"class {c} ({name})")
ax.legend(loc="lower left", fontsize=8, markerscale=2)
ax.set_title(f"Latent classifier (accuracy {acc:.3f})")

fig.suptitle("ex00  Toy model: a VAE with a 2-D latent space for 16×16 images of a bump at a continuous "
             "position,\nand a 3-class classifier on the latents")
save(fig, "ex00_toy_model")

ok = r2 > 0.95 and recon_mse < 0.01 and acc > 0.98
print(f"Latents determine the bump position (k-NN R^2 {r2:.3f}), reconstruction MSE {recon_mse:.4f}, "
      f"classifier accuracy {acc:.3f}: " + ("PASS" if ok else "FAIL"))
assert ok
