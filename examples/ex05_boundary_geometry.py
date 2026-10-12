"""ex05: Boundary geometry — toy_models.py linear classifier in 2D latent space."""
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from latent_smoothness import compute_boundary_geometry, compute_boundary_trajectory
from latent_smoothness.utils import linear_interpolate
from toy_models import get_toy_models

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

torch.manual_seed(0)
decoder, classifier, z_train, y_train = get_toy_models()

res = compute_boundary_geometry(classifier, z_train[:64])

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

sc = axes[0].scatter(z_train[:64, 0].numpy(), z_train[:64, 1].numpy(),
                     c=res.margins.detach().numpy(), cmap="RdBu", s=20)
plt.colorbar(sc, ax=axes[0])
axes[0].set_title("Decision margins (class 0)"); axes[0].set_xlabel("z1"); axes[0].set_ylabel("z2")

sc2 = axes[1].scatter(z_train[:64, 0].numpy(), z_train[:64, 1].numpy(),
                      c=res.gradient_norms.detach().numpy(), s=20, cmap="viridis")
plt.colorbar(sc2, ax=axes[1])
axes[1].set_title("Margin gradient norms ||∇m||")

# Trajectory boundary test
z_start = z_train[:8]
z_end = z_train[8:16]
trajs = linear_interpolate(z_start, z_end, 30)
traj_res = compute_boundary_trajectory(classifier, trajs)
axes[2].bar(range(8), traj_res.sign_change_locations.sum(dim=-1).numpy())
axes[2].set_title(f"Boundary crossings per trajectory\n(total: {traj_res.n_crossings})")
axes[2].set_xlabel("Trajectory"); axes[2].set_ylabel("# crossings")

fig.tight_layout()
fig.savefig(OUT / "ex05_boundary_geometry.png", dpi=80)
plt.close()

# Margins must be positive exactly where the classifier predicts class 0.
with torch.no_grad():
    pred0 = classifier(z_train[:64]).argmax(dim=-1) == 0
ok = bool(((res.margins > 0) == pred0).all()) and res.gradient_norms.mean().item() > 0
print(f"Margin sign matches argmax at all points; boundary crossings: {traj_res.n_crossings} "
      + ("PASS" if ok else "FAIL"))
assert ok
