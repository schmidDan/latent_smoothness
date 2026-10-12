"""ex02: Pullback metric and geodesic discrepancy on the toy VAE decoder (dz=2)."""
import torch

from _common import plt, save, latent_grid
from latent_smoothness import compute_geodesic_discrepancy, compute_pullback_metric
from toy_models import get_toy_models

torch.manual_seed(0)
decoder, classifier, z_train, y_train = get_toy_models()

# Pullback volume element sqrt(det G) over the latent plane: how much decoded
# image space one unit of latent area covers.
gx, gy, pts = latent_grid(z_train, n=60, pad=0.2)
log_vol = 0.5 * compute_pullback_metric(decoder, pts).log_det_G.reshape(gx.shape)

M = 24
perm = torch.randperm(z_train.shape[0])
z_start, z_end = z_train[perm[:M]], z_train[perm[M:2 * M]]
res = compute_geodesic_discrepancy(decoder, z_start, z_end, solver="torchdiffeq", n_steps=30)
order = res.discrepancy.argsort(descending=True)
worst = int(order[0])

fig = plt.figure(figsize=(13, 4.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1.1, 1, 1.5])
ax0 = fig.add_subplot(gs[:, 0])
im = ax0.pcolormesh(gx, gy, log_vol, cmap="magma", shading="auto")
fig.colorbar(im, ax=ax0, label=r"log volume element  $\frac{1}{2}\log\det G$")
for i in range(M):
    ax0.plot([z_start[i, 0], z_end[i, 0]], [z_start[i, 1], z_end[i, 1]],
             color="cyan" if i == worst else "white",
             lw=2 if i == worst else 0.7, alpha=1 if i == worst else 0.6)
ax0.set_aspect("equal")
ax0.set_xlabel("$z_1$")
ax0.set_ylabel("$z_2$")
ax0.set_title("Pullback volume and test paths")

ax1 = fig.add_subplot(gs[:, 1])
ax1.bar(range(M), res.discrepancy[order], color=["cyan"] + ["0.4"] * (M - 1))
ax1.set_xlabel("path (sorted)")
ax1.set_ylabel("(arc − chord) / chord")
ax1.set_title("Geodesic discrepancy per path")

# Decoded images along the most curved path.
t = torch.linspace(0, 1, 8)
with torch.no_grad():
    frames = decoder(z_start[worst] + t[:, None] * (z_end[worst] - z_start[worst]))[:, 0]
ax2 = fig.add_subplot(gs[0, 2])
ax2.imshow(torch.cat(list(frames), dim=1), cmap="gray")
ax2.set_axis_off()
ax2.set_title(f"Decoded along the cyan path (discrepancy {res.discrepancy[worst]:.2f})")
ax3 = fig.add_subplot(gs[1, 2])
ax3.plot(torch.linspace(0, 1, res.pullback_along_path.shape[1]),
         res.pullback_along_path[worst].sqrt(), color="darkcyan")
ax3.set_xlabel("t along path")
ax3.set_ylabel(r"$\sqrt{\det G}$")
fig.suptitle("ex02  Pullback metric: output-space length of straight latent paths vs. their chords")
save(fig, "ex02_geodesic_discrepancy")

ok = (bool(res.discrepancy.isfinite().all()) and bool((res.discrepancy > -1e-3).all())
      and not res.solver_failed)
print(f"Geodesic discrepancy: mean {res.discrepancy.mean():.3f}, max {res.discrepancy.max():.3f}; "
      "finite and arc >= chord: " + ("PASS" if ok else "FAIL"))
assert ok
