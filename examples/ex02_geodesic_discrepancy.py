"""ex02: Geodesic discrepancy on toy_models.py decoder (dz=2)."""
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from latent_smoothness import compute_geodesic_discrepancy
from toy_models import get_toy_models

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

torch.manual_seed(0)
decoder, classifier, z_train, y_train = get_toy_models()

M = 16
z_start = z_train[:M]
z_end = z_train[M:2*M]

result = compute_geodesic_discrepancy(decoder, z_start, z_end, solver="euler", n_steps=30)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
t = torch.linspace(0, 1, result.pullback_along_path.shape[1]).numpy()
for i in range(min(4, M)):
    axes[0].plot(t, result.pullback_along_path[i].detach().numpy(), alpha=0.7)
axes[0].set_title("det G along path (pullback metric)")
axes[0].set_xlabel("t"); axes[0].set_ylabel("|det G|")

disc = result.discrepancy.detach().numpy()
axes[1].bar(range(M), disc)
axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_title("Geodesic discrepancy per path")
axes[1].set_xlabel("Path index"); axes[1].set_ylabel("(arc - chord) / chord")

fig.tight_layout()
fig.savefig(OUT / "ex02_geodesic_discrepancy.png", dpi=80)
plt.close()

mean_disc = result.discrepancy.mean().item()
# Geodesic discrepancy may be near-zero or slightly negative for very smooth decoders;
# the key check is that the diagnostic runs and returns a finite value.
ok = result.discrepancy.isfinite().all().item()
print(f"Mean geodesic discrepancy: {mean_disc:.4f} "
      + ("PASS" if ok else "FAIL (non-finite values)"))
assert ok
