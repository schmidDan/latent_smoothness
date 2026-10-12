"""ex06: Displacement / decision-margin-gradient alignment — toy_models.py VAE + classifier."""
from pathlib import Path
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from latent_smoothness import compute_gradient_alignment
from toy_models import get_toy_models

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

torch.manual_seed(0)
decoder, classifier, z_train, y_train = get_toy_models()

N = 64
z_before = z_train[:N]
# Move along the gradient of the class-0 decision margin (l_0 - l_1 for 2 classes)
W = classifier.weight
delta = (W[0] - W[1]) / (W[0] - W[1]).norm()
z_after = z_before + delta.unsqueeze(0).expand(N, -1)

result = compute_gradient_alignment(classifier, z_before, z_after)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Quiver plot
ax = axes[0]
z2 = z_before.detach().numpy()
d2 = (z_after - z_before).detach().numpy()
ax.quiver(z2[:, 0], z2[:, 1], d2[:, 0], d2[:, 1], alpha=0.5,
          color="blue", label="Δμ", scale=20)
ax.set_title("Latent displacements Δμ")
ax.set_xlabel("z1"); ax.set_ylabel("z2"); ax.legend()

axes[1].hist(result.cosine_similarities.detach().numpy(), bins=20, range=(-1.0, 1.0))
axes[1].axvline(result.mean_alignment, color="red", lw=2,
                label=f"mean={result.mean_alignment:.3f}")
axes[1].set_title("Cosine similarity distribution")
axes[1].set_xlabel("cos(Δμ, ∇m)"); axes[1].legend()

fig.tight_layout()
fig.savefig(OUT / "ex06_gradient_alignment.png", dpi=80)
plt.close()

ok = result.mean_alignment > 0
print(f"Mean alignment: {result.mean_alignment:.3f}, fraction aligned: {result.fraction_aligned:.3f} "
      + ("PASS" if ok else "FAIL"))
assert ok
