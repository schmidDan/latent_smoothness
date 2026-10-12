"""ex05: Decision-boundary geometry of the toy latent classifier (3 classes, dz=2)."""
import torch

from _common import plt, save, latent_grid, class_regions, class_scatter
from latent_smoothness import compute_boundary_geometry, compute_boundary_trajectory
from latent_smoothness.utils import linear_interpolate
from toy_models import get_toy_models

torch.manual_seed(0)
decoder, classifier, z_train, y_train = get_toy_models()
CLASS = 0

gx, gy, pts = latent_grid(z_train, n=45, pad=0.2)
geo = compute_boundary_geometry(classifier, pts, class_idx=CLASS)
margin = geo.margins.reshape(gx.shape)

M = 12
perm = torch.randperm(z_train.shape[0])
z_start, z_end = z_train[perm[:M]], z_train[perm[M:2 * M]]
paths = linear_interpolate(z_start, z_end, n_steps=60)
traj = compute_boundary_trajectory(classifier, paths, class_idx=CLASS)

fig, axes = plt.subplots(2, 2, figsize=(10, 9))
lim = float(margin.abs().max())
panels = [
    (axes[0, 0], margin, "RdBu_r", (-lim, lim), f"Decision margin of class {CLASS}  $m(z)$"),
    (axes[0, 1], geo.gradient_norms.reshape(gx.shape), "viridis", None, r"Margin gradient norm  $\|\nabla m\|$"),
    (axes[1, 0], geo.curvature.reshape(gx.shape).clamp_min(1e-6).log10(), "magma", None,
     r"Curvature  $\log_{10}(\|H_m\|_F^2 / d_z^2)$"),
]
for ax, field, cmap, vlim, title in panels:
    kw = dict(vmin=vlim[0], vmax=vlim[1]) if vlim else {}
    im = ax.pcolormesh(gx, gy, field, cmap=cmap, shading="auto", **kw)
    ax.contour(gx, gy, margin, levels=[0.0], colors="k", linewidths=1.2)
    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_aspect("equal")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_title(title)

ax = axes[1, 1]
class_regions(ax, classifier, z_train)
class_scatter(ax, z_train[::8], y_train[::8], s=2, alpha=0.3)
for i in range(M):
    ax.plot(paths[i, :, 0], paths[i, :, 1], color="0.2", lw=0.8)
    hits = traj.sign_change_locations[i].nonzero().squeeze(-1)
    ax.scatter(paths[i, hits, 0], paths[i, hits, 1], color="red", s=22, zorder=3,
               label="class-0 boundary crossing" if i == 0 else None)
ax.set_title(f"Paths and boundary crossings (total {traj.n_crossings})")
ax.legend(loc="lower left")
fig.suptitle("ex05  Decision-boundary geometry (black line: class-0 boundary, $m = 0$)")
save(fig, "ex05_boundary_geometry")

# Checks: margin > 0 exactly where class 0 is predicted; crossings are argmax changes.
with torch.no_grad():
    pred0 = classifier(pts).argmax(dim=-1) == CLASS
    path_pred0 = classifier(paths.reshape(-1, 2)).argmax(dim=-1).reshape(M, -1) == CLASS
expected = int((path_pred0[:, 1:] != path_pred0[:, :-1]).sum())
ok = bool(((geo.margins > 0) == pred0).all()) and traj.n_crossings == expected
print(f"Margin sign matches argmax on the grid; {traj.n_crossings} crossings match argmax changes: "
      + ("PASS" if ok else "FAIL"))
assert ok
