"""ex10: Full DiagnosticSuite on toy_models.py VAE."""
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from latent_smoothness import DiagnosticSuite, SuiteConfig
from toy_models import get_toy_models

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

torch.manual_seed(0)
decoder, classifier, z_train, y_train = get_toy_models()

N = min(64, z_train.shape[0])
z_reference = z_train[y_train == 0][:N//2]   # class-0 latents
z_query = z_train[y_train == 1][:N//2]       # class-1 latents
endpoints = torch.stack([z_reference[:16], z_query[:16]], dim=1)  # (16, 2, dz)

config = SuiteConfig(geodesic_solver="euler", device="cpu")  # all diagnostics enabled

suite = DiagnosticSuite(decoder=decoder, classifier=classifier, config=config)
result = suite.run(z_reference=z_reference, z_query=z_query, trajectory_endpoints=endpoints)

print(result.summary_table())

scalars = result.to_scalars()

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

if result.jacobian_reference is not None:
    svs = result.jacobian_reference.singular_values.detach()
    axes[0, 0].boxplot([svs[:, i].numpy() for i in range(svs.shape[1])])
    axes[0, 0].set_title("Jacobian SVs (reference)")

if result.lid_reference is not None:
    axes[0, 1].hist(result.lid_reference.lid_twonn.numpy(), bins=20)
    axes[0, 1].set_title(f"LID TwoNN (reference)\nmean={result.lid_reference.mean_lid_twonn:.2f}")

if result.ppl is not None:
    axes[0, 2].hist(result.ppl.ppl_values.detach().numpy().ravel(), bins=30)
    axes[0, 2].set_title(f"PPL values\nmean={result.ppl.mean_ppl:.4f}")

if result.geodesic is not None:
    disc = result.geodesic.discrepancy.detach().numpy()
    axes[1, 0].bar(range(len(disc)), disc)
    axes[1, 0].set_title("Geodesic discrepancy")

if result.alignment is not None:
    axes[1, 1].hist(result.alignment.cosine_similarities.detach().numpy(), bins=20, range=(-1.0, 1.0))
    axes[1, 1].set_title(f"Alignment cos sim\nmean={result.alignment.mean_alignment:.3f}")

if result.boundary_trajectory is not None:
    sc = result.boundary_trajectory.sign_change_locations.sum(dim=-1).numpy()
    axes[1, 2].bar(range(len(sc)), sc)
    axes[1, 2].set_title(f"Boundary crossings\ntotal={result.boundary_trajectory.n_crossings}")

fig.tight_layout()
fig.savefig(OUT / "ex10_full_suite.png", dpi=80)
plt.close()

ok = all(getattr(result, name) is not None for name in result.__dataclass_fields__) and len(scalars) > 0
print("Suite completed successfully " + ("PASS" if ok else "FAIL"))
assert ok
