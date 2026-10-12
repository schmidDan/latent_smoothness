"""ex11: Warp gallery — every warp applied to the toy VAE latents, placed by smoothness and readability.

Each warp re-maps the same latents. Two numbers characterise the result:

* **max local stretch** — the largest secant ratio ||W(z+Δ) − W(z)|| / ||Δ|| at the
  smallest step along random paths (``compute_secant_roughness``). It stays small for
  continuous maps and explodes for maps that jump across cell boundaries.
* **readability** — accuracy of a k-NN classifier *retrained* on the warped latents,
  i.e. whether the class information is still there, in whatever arrangement.
* **arrangement kept** — accuracy of the *original* latent classifier on the warped
  latents, i.e. whether classes still sit where they used to.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import KNeighborsClassifier

from _common import plt, save, class_scatter
from latent_smoothness import compute_secant_roughness
from latent_smoothness.warp import build_warp
from toy_models import get_toy_models

torch.manual_seed(0)
np.random.seed(0)
decoder, classifier, z_train, y_train = get_toy_models()

SPECS = {
    "vq": dict(codebook_size=32),
    "scramble": dict(codebook_size=32),
    "class_scramble": dict(codebook_size=32, labels=y_train, class_mode="within"),
    "flow_warp": dict(n_epochs=300),
    "piecewise_affine": dict(bins_per_dim=4),
    "class_collapse": dict(labels=y_train),
}


class AsModule(nn.Module):
    """Expose a warp's forward as an nn.Module for the decoder-style diagnostics."""
    def __init__(self, warp):
        super().__init__()
        self.warp = warp

    def forward(self, z):
        return self.warp.forward(z)


perm = torch.randperm(z_train.shape[0])
path_a, path_b = z_train[perm[:200]], z_train[perm[200:400]]
split = 3000


def readability(z):
    knn = KNeighborsClassifier(n_neighbors=10).fit(z[:split].numpy(), y_train[:split].numpy())
    return knn.score(z[split:].numpy(), y_train[split:].numpy())


def arrangement(z):
    with torch.no_grad():
        return (classifier(z).argmax(dim=-1) == y_train).float().mean().item()


def stretch(module):
    return float(compute_secant_roughness(module, path_a, path_b, n_steps=64).max_roughness[0])


summary = {"original": (stretch(nn.Identity()), readability(z_train), None, arrangement(z_train))}
warped = {}
for name, kw in SPECS.items():
    warp = build_warp(name, z_train, **kw)
    with torch.no_grad():
        warped[name] = warp.forward(z_train)
    summary[name] = (stretch(AsModule(warp)), readability(warped[name]), warp, arrangement(warped[name]))

fig, axes = plt.subplots(2, 4, figsize=(16, 8.4))
panels = [("original", z_train)] + list(warped.items())
quantizing = {"vq", "scramble", "class_scramble"}
lims = (z_train.min().item() - 0.3, z_train.max().item() + 0.3)
for ax, (name, z) in zip(axes.flat, panels):
    if name in quantizing:      # all points sit on 32 codebook entries: jitter to show them
        z = z + 0.07 * torch.randn_like(z)
    class_scatter(ax, z, y_train, s=2, alpha=0.5)
    if name != "flow_warp":     # the flow moves points outside the original range
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
    s, r, warp, a = summary[name]
    flags = "" if warp is None else (
        f"\n{'invertible' if warp.is_invertible else 'not invertible'}, "
        f"{'differentiable' if warp.is_differentiable else 'not differentiable'}")
    ax.set_title(f"{name}{flags}\nstretch {s:.0f}\nreadability {r:.2f}  ·  arrangement {a:.2f}",
                 fontsize=9)

ax = axes.flat[-1]
for name, (s, r, _, a) in summary.items():
    sc = ax.scatter(s, r, s=60, c=[a], cmap="viridis", vmin=0, vmax=1, edgecolors="k", zorder=3)
    offset = {"scramble": (-10, -13), "class_scramble": (-25, 8), "vq": (-12, 8)}.get(name, (5, 4))
    ax.annotate(name, (s, r), textcoords="offset points", xytext=offset, fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("max local stretch (log scale)  →  rougher")
ax.set_ylabel("readability (retrained k-NN accuracy)")
ax.set_ylim(0.3, 1.05)
ax.axhline(1 / 3, color="0.6", ls="--", lw=1)
fig.colorbar(sc, ax=ax, shrink=0.8, label="arrangement kept\n(original classifier accuracy)")
ax.set_title("Smoothness vs. readability\n(colour: arrangement kept)", fontsize=9)
fig.suptitle("ex11  Warp gallery: the same latents under each warp (colour = original class; quantized "
             "warps jittered for visibility).\n"
             "Warps change smoothness and class arrangement independently.")
save(fig, "ex11_warp_gallery")

rough = [summary[n][0] for n in ("vq", "scramble", "class_scramble", "piecewise_affine")]
arr = {n: v[3] for n, v in summary.items()}
ok = (min(rough) > 3 * summary["flow_warp"][0]                      # jumps >> smooth flow
      and summary["class_collapse"][0] < 2                            # linear projection
      and summary["class_collapse"][1] < min(summary[n][1] for n in SPECS if n != "class_collapse")
      and summary["scramble"][1] > 0.8                                # scrambled but still readable
      and arr["class_scramble"] > arr["scramble"] + 0.3)             # within-class swaps keep the arrangement
print("Stretch: " + ", ".join(f"{n} {v[0]:.0f}" for n, v in summary.items()))
print("Readability: " + ", ".join(f"{n} {v[1]:.2f}" for n, v in summary.items()))
print("Arrangement: " + ", ".join(f"{n} {v:.2f}" for n, v in arr.items()))
print("Discontinuous warps stretch more than the flow; only class_collapse loses readability; "
      "within-class scramble keeps the arrangement: "
      + ("PASS" if ok else "FAIL"))
assert ok
