"""ex07: VQ vs. bijective scramble — backprojection and exact round-trip."""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from latent_smoothness.warp.quantize import posthoc_vq
from latent_smoothness.warp.scramble import bijective_scramble
from toy_models import get_toy_models

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

torch.manual_seed(0)
decoder, classifier, z_train, y_train = get_toy_models()
colors = ["C0" if yi == 0 else "C1" for yi in y_train.numpy()]

K = 32

# --- forward path ---
z_vq, _ = posthoc_vq(z_train, codebook_size=K)
z_scrambled, encode_fn, decode_fn = bijective_scramble(z_train, codebook_size=K)

# --- bijective inverse ---
# decode_fn exactly inverts the permutation; backprojection recovers z_vq (from scramble's VQ)
z_backprojected = decode_fn(z_scrambled)

# round-trip residual: encode(decode(z_scrambled)) must equal z_scrambled exactly
z_roundtrip = encode_fn(z_backprojected)
residuals = torch.norm(z_roundtrip - z_scrambled, dim=-1)


def accuracy(z, y):
    with torch.no_grad():
        return (classifier(z).argmax(dim=-1) == y).float().mean().item()


acc_orig = accuracy(z_train, y_train)
acc_vq = accuracy(z_vq, y_train)
acc_scrambled = accuracy(z_scrambled, y_train)
acc_back = accuracy(z_backprojected, y_train)


def scatter(ax, z, title, subtitle=""):
    ax.scatter(z[:, 0].detach().numpy(), z[:, 1].detach().numpy(),
               c=colors, s=6, alpha=0.5)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    ax.set_xlabel("z[0]"); ax.set_ylabel("z[1]")


fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# --- row 1: forward path ---
scatter(axes[0, 0], z_train,     "Original z",             f"acc={acc_orig:.3f}")
scatter(axes[0, 1], z_vq,        f"VQ quantized (K={K})",  f"acc={acc_vq:.3f}")
scatter(axes[0, 2], z_scrambled, f"Bijective scramble (K={K})",
        f"acc={acc_scrambled:.3f}  — clusters permuted, information preserved")

# --- row 2: bijective property ---

# panel 4: backprojection recovers original cluster positions
scatter(axes[1, 0], z_backprojected,
        "Backprojected: decode_fn(z_scrambled)",
        f"acc={acc_back:.3f}  — cluster positions restored exactly")

# panel 5: displacement quiver — one arrow per unique codebook cell
# shows how far and in which direction the permutation moved each cluster
src_np = z_scrambled[:, :2].detach().numpy()
dst_np = z_backprojected[:, :2].detach().numpy()
# deduplicate: only K unique source positions (codebook cells)
seen = {}
for s, d in zip(src_np.round(4), dst_np.round(4)):
    seen[tuple(s)] = tuple(d)
srcs = np.array(list(seen.keys()))
dsts = np.array(list(seen.values()))
axes[1, 1].quiver(
    srcs[:, 0], srcs[:, 1],
    dsts[:, 0] - srcs[:, 0], dsts[:, 1] - srcs[:, 1],
    angles="xy", scale_units="xy", scale=1,
    color="navy", alpha=0.6, width=0.005, zorder=3,
)
axes[1, 1].scatter(srcs[:, 0], srcs[:, 1], s=25, c="steelblue",
                   zorder=4, label=f"scrambled ({len(srcs)} cells)")
axes[1, 1].scatter(dsts[:, 0], dsts[:, 1], s=25, c="tomato",
                   marker="x", zorder=5, label="backprojected")
axes[1, 1].set_title(
    f"Displacement: scrambled -> backprojected\n(one arrow per codebook cell, K={K})",
    fontsize=10,
)
axes[1, 1].set_xlabel("z[0]"); axes[1, 1].set_ylabel("z[1]")
axes[1, 1].legend(fontsize=8)

# panel 6: round-trip residual histogram — should spike at 0
axes[1, 2].hist(residuals.detach().numpy(), bins=30, color="steelblue", alpha=0.8)
axes[1, 2].axvline(
    residuals.max().item(), color="red", linestyle="--",
    label=f"max = {residuals.max():.2e}",
)
axes[1, 2].set_title(
    "Round-trip residual\n||encode_fn(decode_fn(z_scrambled)) - z_scrambled||",
    fontsize=10,
)
axes[1, 2].set_xlabel("residual norm")
axes[1, 2].legend(fontsize=8)

fig.suptitle(
    f"Bijective scramble: accuracy drops on scrambled latents, "
    f"restored exactly by backprojection  "
    f"(orig={acc_orig:.3f}  vq={acc_vq:.3f}  scrambled={acc_scrambled:.3f}  "
    f"backprojected={acc_back:.3f}  round-trip max={residuals.max():.2e})",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(OUT / "ex07_vq_scramble_comparison.png", dpi=80)
plt.close()

# The scramble is a bijection on codebook cells: backprojection must restore the
# VQ accuracy exactly, and the round trip must be lossless.
ok = abs(acc_back - acc_vq) < 1e-6 and residuals.max().item() == 0.0
print(
    f"Classifier accuracy -- Original: {acc_orig:.3f}, VQ: {acc_vq:.3f}, "
    f"Scrambled: {acc_scrambled:.3f}, Backprojected: {acc_back:.3f}  "
    f"round-trip max={residuals.max():.2e}  " + ("PASS" if ok else "FAIL")
)
assert ok
