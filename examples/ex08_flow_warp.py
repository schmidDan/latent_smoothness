"""ex08: Flow warp — the toy latents mapped onto a mixture of four Gaussians by a learned bijection."""
import numpy as np
import torch

from _common import plt, save, class_scatter
from latent_smoothness.warp.flow_warp import FlowWarp
from toy_models import get_toy_models

torch.manual_seed(0)
np.random.seed(0)
decoder, classifier, z_train, y_train = get_toy_models()

warp = FlowWarp()
warp.fit(z_train, target="mixture_of_gaussians", n_epochs=300)
diag = warp.verify(z_train)

with torch.no_grad():
    z_warped = warp.forward(z_train)
    z_recovered = warp.inverse(z_warped)
rel_err = (z_recovered - z_train).norm(dim=-1) / z_train.norm(dim=-1).clamp_min(1e-6)

fig, axes = plt.subplots(1, 4, figsize=(17, 4), width_ratios=[1, 1, 1, 0.9])
class_scatter(axes[0], z_train, y_train)
axes[0].set_title("Original latents")
target = warp._target_samples
axes[1].scatter(target[:, 0], target[:, 1], s=3, alpha=0.4, color="0.4", lw=0)
axes[1].set_aspect("equal")
axes[1].set_xlabel("$w_1$")
axes[1].set_ylabel("$w_2$")
axes[1].set_title("Target: mixture of 4 Gaussians")
class_scatter(axes[2], z_warped, y_train)
axes[2].set_xlabel("$w_1$")
axes[2].set_ylabel("$w_2$")
axes[2].set_title(f"Warped latents  W(z)\nKL to target {diag.kl_to_target:.2f}, "
                  f"KL to original {diag.kl_to_original:.2f}")
axes[3].hist(rel_err.clamp_min(1e-9).log10(), bins=40, color="C4", alpha=0.8)
axes[3].set_xlabel(r"$\log_{10}$ relative error  $\|W^{-1}(W(z)) - z\| / \|z\|$  (exact: $-9$)")
axes[3].set_ylabel("count")
axes[3].set_title(f"Round trip (max rel. error {rel_err.max():.1e})")
fig.suptitle("ex08  FlowWarp: a smooth bijection reshapes the latent distribution; "
             "the class arrangement moves with it and is recoverable up to float precision")
save(fig, "ex08_flow_warp")

ok = diag.kl_to_target < diag.kl_to_original and rel_err.max().item() < 1e-3
print(f"FlowWarp KL-to-target={diag.kl_to_target:.3f}  KL-to-orig={diag.kl_to_original:.3f}  "
      f"max round-trip rel. error={rel_err.max():.1e}  " + ("PASS" if ok else "FAIL"))
assert ok
