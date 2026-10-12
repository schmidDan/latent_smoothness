"""ex10: The full DiagnosticSuite — confident vs. ambiguous latents of the toy VAE."""
import torch

from _common import plt, save, class_regions, class_scatter
from latent_smoothness import DiagnosticSuite, SuiteConfig
from toy_models import get_toy_models

torch.manual_seed(0)
decoder, classifier, z_train, y_train = get_toy_models()

# Reference: the most confidently classified latents. Query: the most ambiguous
# ones (lowest top-class probability), which sit where the classes meet.
with torch.no_grad():
    probs = classifier(z_train).softmax(dim=-1)
confidence, pred = probs.max(dim=-1)
z_query = z_train[confidence.argsort()[:200]]
# Reference: the 70 most confident latents of each predicted class.
per_class = [torch.nonzero(pred == c).squeeze(1) for c in range(probs.shape[1])]
per_class = [idx[confidence[idx].argsort(descending=True)[:70]] for idx in per_class]
z_reference = z_train[torch.cat(per_class)]
# Paths from ambiguous latents to confident class-0 latents.
targets = z_train[per_class[0]]
endpoints = torch.stack([z_query[:16], targets[:16]], dim=1)          # (16, 2, dz)

config = SuiteConfig(geodesic_solver="euler", geodesic_n_steps=20, ppl_n_steps=40, device="cpu")
result = DiagnosticSuite(decoder, classifier, config).run(
    z_reference=z_reference, z_query=z_query, trajectory_endpoints=endpoints)
print(result.summary_table())
scalars = result.to_scalars()

fig = plt.figure(figsize=(15, 7.5))
gs = fig.add_gridspec(2, 6, height_ratios=[1.5, 1])
ax = fig.add_subplot(gs[0, :3])
class_regions(ax, classifier, z_train)
class_scatter(ax, z_train[::5], y_train[::5], s=2, alpha=0.2)
ax.scatter(*z_reference.T, s=6, color="k", label="reference: confident")
ax.scatter(*z_query.T, s=6, color="red", label="query: ambiguous")
for a, b in endpoints:
    ax.annotate("", xy=b.tolist(), xytext=a.tolist(),
                arrowprops=dict(arrowstyle="->", color="0.3", lw=0.7))
ax.set_title("Point sets and trajectories (ambiguous → confident class 0)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

text_ax = fig.add_subplot(gs[0, 3:])
text_ax.set_axis_off()
traj = [
    ("geodesic discrepancy (mean)", f"{scalars['geodesic.mean_discrepancy']:.2f}"),
    ("perceptual path length (mean LPIPS/step)", f"{scalars['ppl.mean_ppl']:.4f}"),
    ("boundary crossings (class 0, all paths)", f"{scalars['boundary_trajectory.n_crossings']}"),
    ("gradient alignment (mean cosine)", f"{scalars['alignment.mean_alignment']:+.2f}"),
    ("secant roughness log-log slope", f"{scalars['secant_roughness.loglog_slope']:+.2f}"),
    ("label flips per path (mean)", f"{scalars['label_coherence.mean_label_flips']:.2f}"),
    ("monotonic gain of P(target) (mean)", f"{scalars['label_coherence.mean_monotonic_gain']:+.2f}"),
]
text_ax.text(0.0, 1.0, "Trajectory diagnostics", fontweight="bold", fontsize=11, va="top")
for i, (name, val) in enumerate(traj):
    y = 0.86 - i * 0.115
    text_ax.text(0.02, y, name, va="top")
    text_ax.text(0.80, y, val, va="top", ha="right", family="monospace")

point_metrics = [
    ("Jacobian\nLipschitz", "jacobian_{}.lipschitz_estimate"),
    ("intrinsic dim.\n(TwoNN)", "lid_{}.mean_lid_twonn"),
    ("|class-0 margin|\n(mean)", None),
    ("boundary\ncurvature (mean)", "boundary_point_{}.mean_curvature"),
    ("within-class\neffective rank", "label_dimensionality_{}.within_class_effrank"),
    ("Fisher\nratio", "label_dimensionality_{}.fisher_ratio"),
]
for i, (title, key) in enumerate(point_metrics):
    ax = fig.add_subplot(gs[1, i])
    if key is None:
        vals = [getattr(result, f"boundary_point_{t}").margins.abs().mean().item() for t in ("reference", "query")]
    else:
        vals = [scalars[key.format(t)] for t in ("reference", "query")]
    bars = ax.bar(["reference", "query"], vals, color=["k", "red"])
    ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.margins(y=0.2)
fig.suptitle("ex10  DiagnosticSuite: point-set diagnostics on confident (reference) vs. ambiguous (query) "
             "latents, and trajectory diagnostics on paths between them")
save(fig, "ex10_full_suite")

margin_ref, margin_query = (getattr(result, f"boundary_point_{t}").margins.abs().mean().item()
                            for t in ("reference", "query"))
ok = (all(getattr(result, name) is not None for name in result.__dataclass_fields__)
      and margin_query < margin_ref)
print("Suite complete; ambiguous latents have smaller margins than confident ones: "
      + ("PASS" if ok else "FAIL"))
assert ok
