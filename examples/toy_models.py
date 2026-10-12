"""Shared toy VAE + latent classifier used by the examples.

Data: 16x16 images of a single Gaussian bump at a continuous position
``(u, v) ∈ [-1, 1]^2``. The two generative factors span a 2-D sheet, so a VAE
with a 2-D latent learns a smooth, spread-out latent map of bump positions.

Labels: three classes by the angle sector of ``(u, v)`` (0-120°, 120-240°,
240-360°). The class regions meet at the image centre, which gives curved
decision boundaries in latent space and a genuinely ambiguous region where all
three classes meet.

Usage:
    from toy_models import get_toy_models
    decoder, classifier, z_train, y_train = get_toy_models()

Models are trained once (about a minute on CPU, single-threaded for reproducibility) and cached in
``examples/outputs/``.
"""
import math
from pathlib import Path

import torch
import torch.nn as nn

CACHE_PATH = Path(__file__).parent / "outputs" / "toy_model_v2.pt"
DZ = 2
SIDE = 16
IMG_SHAPE = (1, SIDE, SIDE)
IMG_DIM = SIDE * SIDE
N_CLASSES = 3
CLASS_NAMES = ["sector 0", "sector 1", "sector 2"]


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IMG_DIM, 128), nn.Tanh(),
            nn.Linear(128, 64), nn.Tanh(),
            nn.Linear(64, DZ * 2),
        )

    def forward(self, x):
        h = self.net(x.reshape(x.shape[0], -1))
        mu, log_var = h.chunk(2, dim=-1)
        return mu, log_var


class ToyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DZ, 64), nn.Tanh(),
            nn.Linear(64, 128), nn.Tanh(),
            nn.Linear(128, IMG_DIM), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z).reshape(-1, *IMG_SHAPE)


class ToyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ToyEncoder()
        self.decoder = ToyDecoder()

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = mu + torch.randn_like(mu) * (0.5 * log_var).exp()
        return self.decoder(z), mu, log_var


def make_classifier() -> nn.Module:
    return nn.Sequential(nn.Linear(DZ, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                         nn.Linear(32, N_CLASSES))


def render(uv: torch.Tensor, width: float = 0.18) -> torch.Tensor:
    """Images of a Gaussian bump centred at ``uv`` (N, 2) in [-1, 1]^2."""
    grid = torch.linspace(-1.0, 1.0, SIDE)
    gy, gx = torch.meshgrid(grid, grid, indexing="ij")
    d2 = (gx[None] - uv[:, 0, None, None]) ** 2 + (gy[None] - uv[:, 1, None, None]) ** 2
    return torch.exp(-d2 / (2 * width ** 2)).unsqueeze(1)          # (N, 1, SIDE, SIDE)


def sector_labels(uv: torch.Tensor) -> torch.Tensor:
    angle = torch.atan2(uv[:, 1], uv[:, 0]) % (2 * math.pi)
    return (angle // (2 * math.pi / N_CLASSES)).long().clamp(max=N_CLASSES - 1)


def _make_dataset(n: int = 4096):
    g = torch.Generator().manual_seed(42)
    uv = torch.rand(n, 2, generator=g) * 2 - 1
    x = (render(uv) + 0.02 * torch.randn(n, *IMG_SHAPE, generator=g)).clamp(0, 1)
    return x, sector_labels(uv), uv


def _train_vae(x, n_epochs: int = 150, beta: float = 1.0):
    torch.manual_seed(42)
    vae = ToyVAE()
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    n = x.shape[0]
    for _ in range(n_epochs):
        perm = torch.randperm(n)
        for start in range(0, n, 128):
            xb = x[perm[start:start + 128]]
            x_hat, mu, log_var = vae(xb)
            recon = nn.functional.binary_cross_entropy(x_hat, xb, reduction="sum") / xb.shape[0]
            kld = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(dim=1).mean()
            loss = recon + beta * kld
            opt.zero_grad()
            loss.backward()
            opt.step()
    return vae


def _train_classifier(z, y, n_steps: int = 1000):
    # Strong weight decay keeps the classifier smooth (gradual margins rather
    # than a cliff at the boundary), which makes the boundary plots readable.
    torch.manual_seed(42)
    clf = make_classifier()
    opt = torch.optim.AdamW(clf.parameters(), lr=1e-2, weight_decay=3.0)
    for _ in range(n_steps):
        loss = nn.functional.cross_entropy(clf(z), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return clf


def get_toy_models():
    """Returns ``(decoder, classifier, z_train, y_train)``.

    ``z_train`` are the encoder means of the training images (N, 2) and
    ``y_train`` their true sector labels (N,). Both models are in eval mode.
    """
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    vae, clf = ToyVAE(), make_classifier()
    if CACHE_PATH.exists():
        ckpt = torch.load(CACHE_PATH, weights_only=True)
        vae.load_state_dict(ckpt["vae"])
        clf.load_state_dict(ckpt["classifier"])
        z_train, y_train = ckpt["z_train"], ckpt["y_train"]
    else:
        # Train with one thread (bit-reproducible across machines with the same
        # torch build) and in an isolated RNG context, so that the caller's random
        # state is the same whether or not the models came from the cache.
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with torch.random.fork_rng(devices=[]):
                x, y_train, _ = _make_dataset()
                vae = _train_vae(x)
                vae.eval()
                with torch.no_grad():
                    z_train, _ = vae.encoder(x)
                clf = _train_classifier(z_train, y_train)
        finally:
            torch.set_num_threads(n_threads)
        torch.save({"vae": vae.state_dict(), "classifier": clf.state_dict(),
                    "z_train": z_train, "y_train": y_train}, CACHE_PATH)
    vae.eval()
    clf.eval()
    return vae.decoder, clf, z_train, y_train
