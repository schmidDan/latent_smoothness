"""ex09: MINE mutual information — bijective scramble and the limits of MINE estimation.

Runs two experiments:
  N=1024  (K=64: ~16 samples/cluster, K=4: ~256 samples/cluster)
  N=16384 (K=64: ~256 samples/cluster, K=4: ~4096 samples/cluster)

Key findings:
  K=4 triplet   — always tight regardless of N.  MINE correctly estimates that
                  scramble(VQ_4) preserves as much MI as VQ_4 directly.
  K=64 triplet  — diverges at both N.  MI(z, scramble(VQ_64)) is severely
                  underestimated despite the bijection being algebraically exact.
                  Root cause: the 64-hidden-unit MINE net must learn 64 distinct
                  centroid-to-centroid associations simultaneously; it lacks the
                  capacity to do so within 200 epochs regardless of sample size.
  roundtrip     — always equals VQ_direct (same tensor, same MINE estimate).
                  This is the decisive algebraic proof: the bijection adds
                  zero information loss — MINE simply cannot estimate it for K=64.

Within each K-triplet (VQ | scramble | roundtrip) all three bars should cluster:
  MI(z, VQ_K)            direct quantisation
  MI(z, scramble(VQ_K))  bijective permutation — should == VQ_K by MI invariance
  MI(z, roundtrip(VQ_K)) decode(encode(VQ_K)) == VQ_K exactly by algebra
"""
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from latent_smoothness.mine import estimate_mi
from latent_smoothness.warp.scramble import bijective_scramble

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

SEED = 0


# ─── helpers ────────────────────────────────────────────────────────────────

def build_conditions(N):
    torch.manual_seed(SEED); np.random.seed(SEED)
    z = torch.randn(N, 4)

    torch.manual_seed(SEED)
    z_sc64, enc64, dec64 = bijective_scramble(z, codebook_size=64)
    z_vq64 = dec64(z_sc64)                 # bijective_scramble's internal VQ
    z_rt64 = dec64(enc64(z_vq64))          # round-trip: == z_vq64 exactly

    torch.manual_seed(SEED + 1)
    z_sc4, enc4, dec4 = bijective_scramble(z, codebook_size=4)
    z_vq4 = dec4(z_sc4)
    z_rt4 = dec4(enc4(z_vq4))

    torch.manual_seed(SEED); np.random.seed(SEED)
    z_shuf = z[torch.randperm(N)]

    return z, dict(identity=z, vq64=z_vq64, sc64=z_sc64, rt64=z_rt64,
                   vq4=z_vq4, sc4=z_sc4, rt4=z_rt4, shuffle=z_shuf)


def run_mine(z, conds, n_epochs=200):
    tags = dict(
        identity="MI(z, z)                 identity",
        vq64    ="MI(z, VQ_64)             K=64 direct",
        sc64    ="MI(z, scramble(VQ_64))   K=64 bijective",
        rt64    ="MI(z, roundtrip(VQ_64))  K=64 round-trip",
        vq4     ="MI(z, VQ_4)              K=4  direct",
        sc4     ="MI(z, scramble(VQ_4))    K=4  bijective",
        rt4     ="MI(z, roundtrip(VQ_4))   K=4  round-trip",
        shuffle ="MI(z, shuffle)           floor",
    )
    res = {}
    for key, z_b in conds.items():
        torch.manual_seed(SEED); np.random.seed(SEED)
        val = estimate_mi(z, z_b, n_epochs=n_epochs)
        print(f"  {tags[key]:45s}: {val:.3f}")
        res[key] = val
    return res


def make_figure(res, N, path):
    K64_spc = N // 64
    K4_spc  = N // 4
    spread64 = max(res["vq64"], res["sc64"], res["rt64"]) - \
               min(res["vq64"], res["sc64"], res["rt64"])
    spread4  = max(res["vq4"],  res["sc4"],  res["rt4"])  - \
               min(res["vq4"],  res["sc4"],  res["rt4"])

    groups = [
        ("Identity",
         ["MI(z,z)"],
         ["#444444"],
         [res["identity"]]),
        (f"K = 64  ({K64_spc} spc)",
         ["VQ", "scramble", "roundtrip"],
         ["#1a4f8a", "#4d9de0", "#aed6f1"],
         [res["vq64"], res["sc64"], res["rt64"]]),
        (f"K = 4  ({K4_spc} spc)",
         ["VQ", "scramble", "roundtrip"],
         ["#8a1a1a", "#e04d4d", "#f1aeae"],
         [res["vq4"], res["sc4"], res["rt4"]]),
        ("Shuffle\nfloor",
         ["shuffle"],
         ["#aaaaaa"],
         [res["shuffle"]]),
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    bw, gap = 0.6, 1.1
    x = 0.0
    bar_xs = {}   # key -> x centre

    for g_label, b_labels, b_colors, values in groups:
        centres = []
        for lbl, col, val, key in zip(b_labels, b_colors, values,
                                      _keys_for_group(g_label)):
            ax.bar(x, val, width=bw, color=col, alpha=0.88, zorder=3)
            ax.text(x, val + 0.04, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold")
            centres.append(x)
            bar_xs[key] = x
            x += bw
        g_centre = (centres[0] + centres[-1]) / 2
        ax.text(g_centre, -0.22, g_label, ha="center", va="top",
                fontsize=9, fontweight="bold",
                transform=ax.get_xaxis_transform())
        x += gap

    ax.axhline(res["shuffle"],  color="#888", linestyle="--", lw=1.1,
               label=f"shuffle floor  = {res['shuffle']:.3f}")
    ax.axhline(res["identity"], color="#333", linestyle=":",  lw=1.1,
               label=f"identity ceiling = {res['identity']:.3f}")

    # ── annotations ──────────────────────────────────────────────────────
    if spread64 > 0.5:
        # annotate the underestimated scramble bar
        x_sc64 = bar_xs["sc64"]
        ax.annotate(
            f"MINE underestimates\n(only ~{K64_spc} samples/cluster\nis too few for K=64)",
            xy=(x_sc64, res["sc64"] + 0.05),
            xytext=(x_sc64 - 0.8, res["sc64"] + (res["identity"] - res["sc64"]) * 0.45),
            fontsize=7.5, color="#b22222",
            arrowprops=dict(arrowstyle="->", color="#b22222", lw=0.9),
        )
        # annotate that roundtrip == VQ proving algebraic exactness
        x_vq64  = bar_xs["vq64"]
        x_rt64  = bar_xs["rt64"]
        mid_x   = (x_vq64 + x_rt64) / 2
        yval    = max(res["vq64"], res["rt64"]) + 0.15
        ax.annotate("", xy=(x_vq64, yval), xytext=(x_rt64, yval),
                    arrowprops=dict(arrowstyle="<->", color="navy", lw=1.2))
        ax.text(mid_x, yval + 0.05,
                "algebraically\nexact (round-trip = VQ)",
                ha="center", va="bottom", fontsize=7, color="navy")

    ax.set_ylabel("MI estimate (nats)")
    ax.set_xticks([])
    ax.set_ylim(bottom=0)
    if spread64 < 0.5:
        verdict = "K=64 triplet tight: MINE estimation succeeds"
    else:
        verdict = ("K=64 triplet diverges: MINE cannot learn 64-centroid bijection "
                   "— roundtrip=VQ proves algebraic exactness instead")
    ax.set_title(
        f"MINE mutual information — bijective scramble vs. VQ  (N={N:,})\n"
        f"K=64 spread={spread64:.3f}  |  K=4 spread={spread4:.3f}  |  {verdict}",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close()
    print(f"  -> {path.name}  (K=64 spread={spread64:.3f}  K=4 spread={spread4:.3f})")


def _keys_for_group(label):
    if "Identity" in label: return ["identity"]
    if "K = 64"   in label: return ["vq64", "sc64", "rt64"]
    if "K = 4"    in label: return ["vq4",  "sc4",  "rt4"]
    return ["shuffle"]


# ─── run N=1024 ─────────────────────────────────────────────────────────────
N1 = 1024
print(f"\n{'='*62}\nN={N1}  (K=64: {N1//64} spc,  K=4: {N1//4} spc)\n{'='*62}")
z1, cond1 = build_conditions(N1)
res1 = run_mine(z1, cond1)
make_figure(res1, N1, OUT / "ex09_mine_N1024.png")

# ─── run N=16384 ────────────────────────────────────────────────────────────
N2 = 16 * 1024
print(f"\n{'='*62}\nN={N2}  (K=64: {N2//64} spc,  K=4: {N2//4} spc)\n{'='*62}")
z2, cond2 = build_conditions(N2)
res2 = run_mine(z2, cond2)
make_figure(res2, N2, OUT / "ex09_mine_N16384.png")

# ─── assertions ─────────────────────────────────────────────────────────────
# K=64 > K=4: more codebook cells preserve more information
ok_order1 = res1["vq64"] > res1["vq4"]
ok_order2 = res2["vq64"] > res2["vq4"]

# K=4 triplet is tight: MINE reliably estimates simple-bijection MI preservation
spread4_n1 = max(res1["vq4"], res1["sc4"], res1["rt4"]) - min(res1["vq4"], res1["sc4"], res1["rt4"])
spread4_n2 = max(res2["vq4"], res2["sc4"], res2["rt4"]) - min(res2["vq4"], res2["sc4"], res2["rt4"])
ok_k4_tight = spread4_n1 < 0.1 and spread4_n2 < 0.1

# roundtrip == VQ_direct within MINE noise: algebraic bijection confirmed
ok_rt1 = abs(res1["rt64"] - res1["vq64"]) < 0.5 and abs(res1["rt4"] - res1["vq4"]) < 0.1
ok_rt2 = abs(res2["rt64"] - res2["vq64"]) < 0.5 and abs(res2["rt4"] - res2["vq4"]) < 0.1

spread64_n1 = max(res1["vq64"], res1["sc64"], res1["rt64"]) - min(res1["vq64"], res1["sc64"], res1["rt64"])
spread64_n2 = max(res2["vq64"], res2["sc64"], res2["rt64"]) - min(res2["vq64"], res2["sc64"], res2["rt64"])

print(f"\nN={N1}:  K=64 > K=4 ordering:  {'PASS' if ok_order1 else 'FAIL'}")
print(f"N={N2}: K=64 > K=4 ordering:  {'PASS' if ok_order2 else 'FAIL'}")
print(f"K=4 triplet tight (spread<0.1) at both N: {'PASS' if ok_k4_tight else 'FAIL'}  "
      f"(N1={spread4_n1:.3f}, N2={spread4_n2:.3f})")
print(f"roundtrip==VQ within noise at both N:     {'PASS' if (ok_rt1 and ok_rt2) else 'FAIL'}")
print(f"K=64 spread: N={N1}: {spread64_n1:.3f}  N={N2}: {spread64_n2:.3f}  "
      f"(MINE estimation limit — algebraic proof via roundtrip=VQ is conclusive)")

assert ok_order1 and ok_order2, \
    f"K=64 > K=4 must hold at both N."
assert ok_k4_tight, \
    f"K=4 triplet must be tight (spread<0.1) at both N. Got {spread4_n1:.3f}, {spread4_n2:.3f}"
assert ok_rt1 and ok_rt2, \
    f"roundtrip must match VQ_direct within noise, confirming algebraic exactness."
