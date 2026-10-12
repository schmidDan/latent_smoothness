"""ex08: Flow warp — toy_models.py latents warped to mixture_of_gaussians."""
from pathlib import Path
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from latent_smoothness import compute_jacobian_spectrum
from latent_smoothness.warp.flow_warp import FlowWarp
from toy_models import get_toy_models

OUT = Path(__file__).parent / "outputs"
OUT.mkdir(exist_ok=True)

torch.manual_seed(0)
np.random.seed(0)

decoder, classifier, z_train, y_train = get_toy_models()

warp = FlowWarp()
warp.fit(z_train, target="mixture_of_gaussians", n_epochs=300)
diag = warp.verify(z_train)

with torch.no_grad():
    z_warped = warp.forward(z_train.float()).float()
    z_recovered = warp.inverse(z_warped).float()
    residuals = torch.norm(z_recovered - z_train.float(), dim=-1)


class WarpedDecoder(torch.nn.Module):
    def __init__(self, dec, warp):
        super().__init__()
        self.dec = dec
        self.warp = warp

    def forward(self, z):
        return self.dec(self.warp.forward(z))


warped_decoder = WarpedDecoder(decoder, warp)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# --- row 1: scatter plots (original → warped → round-trip) ---
axes[0, 0].scatter(z_train[:, 0].numpy(), z_train[:, 1].numpy(), s=5, alpha=0.5)
axes[0, 0].set_title("Original latents")

axes[0, 1].scatter(
    z_warped[:, 0].detach().numpy(), z_warped[:, 1].detach().numpy(),
    s=5, alpha=0.5, color="orange",
)
axes[0, 1].set_title("Warped latents (forward)")

axes[0, 2].scatter(
    z_train[:, 0].numpy(), z_train[:, 1].numpy(),
    s=8, alpha=0.4, label="original", color="steelblue",
)
axes[0, 2].scatter(
    z_recovered[:, 0].detach().numpy(), z_recovered[:, 1].detach().numpy(),
    s=5, alpha=0.5, label="recovered", color="tomato", marker="x",
)
axes[0, 2].set_title("Round-trip: original vs recovered")
axes[0, 2].legend(fontsize=8)

# --- row 2: density grid | Jacobian spectrum | residual histogram ---
axes[1, 0].imshow(diag.density_grid.numpy(), origin="lower")
axes[1, 0].set_title(
    f"Warped density (KDE)\nKL→target={diag.kl_to_target:.3f}  KL→orig={diag.kl_to_original:.3f}"
)

js_orig = compute_jacobian_spectrum(decoder, z_train[:32])
js_warp = compute_jacobian_spectrum(warped_decoder, z_train[:32])
axes[1, 1].plot(js_orig.mean_spectrum.detach().numpy(), label="original", marker="o")
axes[1, 1].plot(js_warp.mean_spectrum.detach().numpy(), label="warped", marker="s")
axes[1, 1].set_title("Jacobian spectrum before/after warp")
axes[1, 1].legend()

axes[1, 2].hist(residuals.detach().numpy(), bins=30, color="purple", alpha=0.7)
axes[1, 2].axvline(
    residuals.max().item(), color="red", linestyle="--",
    label=f"max = {residuals.max():.2e}",
)
axes[1, 2].set_xlabel("‖inverse(forward(z)) − z‖")
axes[1, 2].set_title("Round-trip residuals per sample")
axes[1, 2].legend(fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "ex08_flow_warp.png", dpi=80)
plt.close()

ok = diag.kl_to_target < diag.kl_to_original
print(
    f"FlowWarp KL-to-target={diag.kl_to_target:.3f}  KL-to-orig={diag.kl_to_original:.3f}  "
    f"round-trip max={residuals.max():.2e}  " + ("PASS" if ok else "FAIL")
)
assert ok, (
    f"Expected kl_to_target < kl_to_original, "
    f"got {diag.kl_to_target:.3f} vs {diag.kl_to_original:.3f}"
)
