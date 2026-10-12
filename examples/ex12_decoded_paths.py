"""ex12: Roughness made visible — decoding one straight path through each warped latent space.

A warp W defines a new latent space z' = W(z). Generating from it means decoding
``decoder(W.backproject(z'))``. We take the same two latents, map them into each
warped space, walk the straight line between them there, and decode every step.
Smooth warps give a bump gliding across the image; discontinuous warps make it
jump wherever the path crosses a cell boundary.
"""
import numpy as np
import torch

from _common import plt, save
from latent_smoothness.warp import build_warp
from toy_models import get_toy_models

torch.manual_seed(0)
np.random.seed(0)
decoder, classifier, z_train, y_train = get_toy_models()

SPECS = {
    "original": None,
    "vq": dict(codebook_size=32),
    "scramble": dict(codebook_size=32),
    "class_scramble": dict(codebook_size=32, labels=y_train, class_mode="within"),
    "flow_warp": dict(n_epochs=300),
    "piecewise_affine": dict(bins_per_dim=4),
    "class_collapse": dict(labels=y_train),
}
SMOOTH = ("original", "flow_warp", "class_collapse")
N_STEPS, N_FRAMES = 200, 12

# Two latents on opposite sides of the sheet.
z_a = z_train[(z_train - torch.tensor([-1.5, -1.5])).norm(dim=-1).argmin()]
z_b = z_train[(z_train - torch.tensor([1.5, 1.5])).norm(dim=-1).argmin()]
t = torch.linspace(0, 1, N_STEPS)[:, None]

frames, steps = {}, {}
for name, kw in SPECS.items():
    if kw is None:
        path_z = z_a + t * (z_b - z_a)
    else:
        warp = build_warp(name, z_train, **kw)
        with torch.no_grad():
            w_a, w_b = warp.forward(torch.stack([z_a, z_b]))
            path_z = warp.backproject(w_a + t * (w_b - w_a))   # straight line in the warped space
    with torch.no_grad():
        images = decoder(path_z)[:, 0]
    frames[name] = images
    steps[name] = (images[1:] - images[:-1]).flatten(1).norm(dim=-1)    # pixel change per step

FLOOR = 1e-3   # quantized warps change nothing between jumps; show exact zeros at this floor
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(len(SPECS), 2, width_ratios=[1.5, 1])
show = torch.linspace(0, N_STEPS - 1, N_FRAMES).round().long()
ymax = max(float(s.max()) for s in steps.values()) * 2
for i, name in enumerate(SPECS):
    ax = fig.add_subplot(gs[i, 0])
    strip = torch.cat([torch.nn.functional.pad(f, (1, 1, 1, 1), value=0.35) for f in frames[name][show]], dim=1)
    ax.imshow(strip, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)

    pax = fig.add_subplot(gs[i, 1])
    pax.fill_between(torch.linspace(0, 1, N_STEPS - 1), FLOOR, steps[name].clamp_min(FLOOR),
                     color="C0" if name in SMOOTH else "C3", lw=0.8, alpha=0.8)
    pax.set_yscale("log")
    pax.set_ylim(FLOOR, ymax)
    pax.set_xlim(0, 1)
    pax.set_yticks([1e-2, 1e0])
    if i < len(SPECS) - 1:
        pax.set_xticklabels([])
    if i == 0:
        ax.set_title("Decoded frames along the path (start → end)")
        pax.set_title(r"Image change per step  $\|x_{t+1} - x_t\|$  (log scale; floor = no change)")
pax.set_xlabel("position t along the straight path in the warped space")
fig.suptitle("ex12  One straight path through each warped latent space, decoded via backproject:\n"
             "continuous warps glide (blue), discontinuous warps jump (red)")
save(fig, "ex12_decoded_paths")

# Jumpiness: largest single step relative to the average step.
jumpiness = {n: float(s.max() / s.mean().clamp_min(1e-12)) for n, s in steps.items()}
print("Jumpiness (max step / mean step): " + ", ".join(f"{n} {v:.1f}" for n, v in jumpiness.items()))
ok = min(jumpiness[n] for n in ("scramble", "class_scramble", "piecewise_affine")) > \
     3 * max(jumpiness[n] for n in ("original", "flow_warp"))
print("Discontinuous warps jump far more than smooth ones: " + ("PASS" if ok else "FAIL"))
assert ok
