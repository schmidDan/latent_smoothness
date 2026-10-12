"""Shared plotting setup for the examples.

Figures go to ``examples/outputs/`` (git-ignored) unless the environment
variable ``LS_EXAMPLES_OUT`` points elsewhere; ``run_all.py --update-figures``
uses that to refresh the tracked figures in ``examples/figures/``.
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

OUT = Path(os.environ.get("LS_EXAMPLES_OUT", Path(__file__).parent / "outputs"))
OUT.mkdir(parents=True, exist_ok=True)

CLASS_COLORS = ["#4C72B0", "#DD8452", "#55A868"]   # one colour per toy class

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
    "figure.constrained_layout.use": True,
})


def save(fig, name: str) -> None:
    """Save ``fig`` as ``<OUT>/<name>.png`` and close it."""
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  saved {OUT / (name + '.png')}")


def latent_grid(z: torch.Tensor, n: int = 200, pad: float = 0.3):
    """A regular ``n x n`` grid covering the 2-D latents ``z`` (plus a margin).

    Returns ``(gx, gy, points)`` with ``points`` of shape ``(n*n, 2)``.
    """
    lo = z.min(dim=0).values - pad
    hi = z.max(dim=0).values + pad
    gx, gy = torch.meshgrid(torch.linspace(float(lo[0]), float(hi[0]), n),
                            torch.linspace(float(lo[1]), float(hi[1]), n), indexing="xy")
    return gx, gy, torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def class_regions(ax, classifier, z, alpha: float = 0.15) -> None:
    """Shade the classifier's argmax regions over the latent extent of ``z``."""
    gx, gy, pts = latent_grid(z)
    with torch.no_grad():
        pred = classifier(pts).argmax(dim=-1).reshape(gx.shape)
    k = len(CLASS_COLORS)
    ax.contourf(gx, gy, pred, levels=[i - 0.5 for i in range(k + 1)],
                colors=CLASS_COLORS, alpha=alpha)
    ax.contour(gx, gy, pred, levels=[i + 0.5 for i in range(k - 1)],
               colors="0.35", linewidths=0.8)


def class_scatter(ax, z, y, s: float = 3, alpha: float = 0.6, **kw) -> None:
    """Scatter 2-D latents coloured by class label."""
    from matplotlib.colors import ListedColormap
    ax.scatter(z[:, 0], z[:, 1], c=y, cmap=ListedColormap(CLASS_COLORS),
               vmin=-0.5, vmax=len(CLASS_COLORS) - 0.5, s=s, alpha=alpha, lw=0, **kw)
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_aspect("equal")
