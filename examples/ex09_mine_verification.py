"""ex09: MINE mutual information — what VQ and the bijective scramble do to information, and where MINE fails.

For z ~ N(0, I_4), quantising to K cells keeps at most log K nats of information,
and the bijective scramble (a permutation of the K cells) must keep exactly as
much as VQ itself. Three estimates per K:

    MI(z, VQ_K(z))               direct quantisation
    MI(z, scramble(VQ_K(z)))     must equal the VQ value (bijection on cells)
    MI(z, roundtrip(VQ_K(z)))    decode(encode(VQ_K)) == VQ_K exactly (sanity check)

For K=4, MINE recovers this. For K=64 it underestimates the scrambled case: the
critic would have to learn 64 arbitrary cell-to-cell associations and does not
within 200 epochs, even with ~256 samples per cell. The round trip equalling
VQ shows that this is an estimation limit, not information loss.
"""
import math

import numpy as np
import torch

from _common import plt, save
from latent_smoothness.mine import estimate_mi
from latent_smoothness.warp.scramble import bijective_scramble

SEED = 0


def build_conditions(n):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    z = torch.randn(n, 4)
    conds = {"identity": z}
    for k in (64, 4):
        z_sc, enc, dec = bijective_scramble(z, codebook_size=k)
        z_vq = dec(z_sc)                       # the scramble's own VQ
        conds[f"vq{k}"], conds[f"sc{k}"], conds[f"rt{k}"] = z_vq, z_sc, dec(enc(z_vq))
    conds["shuffle"] = z[torch.randperm(n)]
    return z, conds


def run_mine(z, conds):
    return {key: estimate_mi(z, z_b, n_epochs=200, seed=SEED) for key, z_b in conds.items()}


SIZES = (1024, 16 * 1024)
results = {}
for n in SIZES:
    z, conds = build_conditions(n)
    results[n] = run_mine(z, conds)
    print(f"N={n}: " + ", ".join(f"{k}={v:.3f}" for k, v in results[n].items()))

groups = [("identity", ["identity"]), ("K = 64", ["vq64", "sc64", "rt64"]),
          ("K = 4", ["vq4", "sc4", "rt4"]), ("shuffled", ["shuffle"])]
labels = {"identity": "MI(z, z)", "shuffle": "MI(z, shuffled z)"}
shades = {"vq": "C0", "sc": "C3", "rt": "C2"}
names = {"vq": "VQ", "sc": "scramble", "rt": "round trip"}

fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
for ax, n in zip(axes, SIZES):
    res, x, ticks = results[n], 0.0, []
    for g_label, keys in groups:
        xs = []
        for key in keys:
            color = shades.get(key[:2], "0.4")
            bar = ax.bar(x, res[key], width=0.8, color=color,
                         label=names.get(key[:2]) if (n == SIZES[0] and key.endswith("64")) else None)
            ax.bar_label(bar, fmt="%.2f", fontsize=8, zorder=4,
                         bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
            xs.append(x)
            x += 1
        ticks.append((sum(xs) / len(xs), g_label))
        if g_label.startswith("K"):
            k = int(g_label.split("=")[1])
            ax.hlines(math.log(k), xs[0] - 0.5, xs[-1] + 0.5, colors="k", linestyles="--", lw=1)
            ax.text(xs[0] - 0.5, math.log(k), f"log {k} ", va="center", ha="right", fontsize=8)
        x += 0.8
    ax.set_xticks([t for t, _ in ticks], [lbl for _, lbl in ticks])
    ax.set_title(f"N = {n:,}  ({n // 64} samples per cell at K = 64)")
    ax.set_ylabel("MINE estimate (nats)")
axes[0].legend(title="for each K", loc="upper right")
fig.suptitle("ex09  MINE: the scramble preserves VQ's information (round trip = VQ exactly), "
             "but MINE underestimates it at K = 64.  Dashed: upper bound log K")
save(fig, "ex09_mine_verification")

checks = []
for n, res in results.items():
    spread4 = max(res["vq4"], res["sc4"], res["rt4"]) - min(res["vq4"], res["sc4"], res["rt4"])
    checks += [res["vq64"] > res["vq4"],                 # more cells keep more information
               spread4 < 0.1,                            # K=4 scramble == VQ
               abs(res["rt64"] - res["vq64"]) < 0.5 and abs(res["rt4"] - res["vq4"]) < 0.1,
               res["vq4"] < math.log(4) + 0.3]           # bounded by log K (+ MINE bias)
ok = all(checks)
print("K=64 > K=4, K=4 triplet tight, round trip == VQ, VQ_4 <= log 4 (both N): " + ("PASS" if ok else "FAIL"))
assert ok
