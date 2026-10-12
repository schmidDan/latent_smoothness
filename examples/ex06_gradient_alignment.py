"""ex06: Alignment of latent displacements with the decision-margin gradient (toy VAE + classifier)."""
import torch

from _common import plt, save, class_regions, class_scatter
from latent_smoothness import compute_gradient_alignment
from latent_smoothness.utils import decision_margin
from toy_models import get_toy_models

torch.manual_seed(0)
decoder, classifier, z_train, y_train = get_toy_models()
TARGET, STEP, N = 0, 0.4, 200

# Start from latents of the other classes and move them toward class TARGET in three ways.
others = z_train[y_train != TARGET]
z0 = others[torch.randperm(others.shape[0])[:N]]

z_req = z0.clone().requires_grad_(True)
grad = torch.autograd.grad(decision_margin(classifier(z_req), TARGET).sum(), z_req)[0]
centroid = z_train[y_train == TARGET].mean(dim=0)
random_dir = torch.randn(N, 2)


def unit(v):
    return v / v.norm(dim=-1, keepdim=True)


strategies = {
    "along margin gradient": unit(grad),
    "toward class centroid": unit(centroid - z0),
    "random direction": unit(random_dir),
}
colors = {"along margin gradient": "C2", "toward class centroid": "C1", "random direction": "0.5"}
results = {name: compute_gradient_alignment(classifier, z0, z0 + STEP * d, class_idx=TARGET)
           for name, d in strategies.items()}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), width_ratios=[1, 1.15])
ax = axes[0]
class_regions(ax, classifier, z_train)
class_scatter(ax, z_train[::6], y_train[::6], s=2, alpha=0.25)
show = torch.arange(0, N, 6)
for name in ["along margin gradient", "toward class centroid"]:
    d = STEP * strategies[name][show]
    ax.quiver(z0[show, 0], z0[show, 1], d[:, 0], d[:, 1], angles="xy", scale_units="xy", scale=1,
              color=colors[name], width=0.004, label=name)
ax.scatter(*centroid, marker="*", s=200, color="k", zorder=4, label=f"class {TARGET} centroid")
ax.set_title(f"Moving latents toward class {TARGET}")
ax.legend(loc="lower left", fontsize=8)

for name, res in results.items():
    axes[1].hist(res.cosine_similarities, bins=40, range=(-1, 1), alpha=0.6, color=colors[name],
                 label=f"{name}: mean {res.mean_alignment:+.2f}, "
                       f"{100 * res.fraction_aligned:.0f}% increase the margin")
axes[1].set_xlabel(r"cosine$(\Delta z,\ \nabla m_{%d})$" % TARGET)
axes[1].set_ylabel("count")
axes[1].set_title("Alignment with the decision-margin gradient")
axes[1].legend(loc="upper left", fontsize=8)
axes[1].set_ylim(top=axes[1].get_ylim()[1] * 1.35)
fig.suptitle("ex06  Gradient alignment: how directly does a latent displacement increase the target-class margin?")
save(fig, "ex06_gradient_alignment")

m = {name: r.mean_alignment for name, r in results.items()}
ok = (m["along margin gradient"] > 0.99 and m["toward class centroid"] > m["random direction"]
      and abs(m["random direction"]) < 0.2)
print("Mean alignment: " + ", ".join(f"{k} {v:+.3f}" for k, v in m.items()) + "  " + ("PASS" if ok else "FAIL"))
assert ok
