"""Shared toy VAE + linear classifier used across all examples.

Architecture:
  - VAE with dz=2 latent space, decoder output (1, 8, 8)
  - Trained on two Gaussian blobs (±2 along first axis)
  - Linear classifier trained on latent codes

Usage:
  from toy_models import get_toy_models
  decoder, classifier, z_train, y_train = get_toy_models()
"""
from pathlib import Path
import torch
import torch.nn as nn

CACHE_PATH = Path(__file__).parent / "outputs" / "toy_model_cache.pt"
DZ = 2
IMG_SHAPE = (1, 8, 8)
IMG_DIM = 64  # 1*8*8


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(IMG_DIM, 64),
            nn.Tanh(),
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
            nn.Linear(DZ, 64),
            nn.Tanh(),
            nn.Linear(64, IMG_DIM),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.net(z)
        return out.reshape(-1, *IMG_SHAPE)


class ToyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ToyEncoder()
        self.decoder = ToyDecoder()

    def reparametrize(self, mu, log_var):
        std = (0.5 * log_var).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparametrize(mu, log_var)
        x_hat = self.decoder(z)
        return x_hat, mu, log_var


def _make_dataset():
    torch.manual_seed(42)
    n_per_class = 512
    # Class 0: mean = (-2, 0, ...), Class 1: mean = (2, 0, ...)
    x0 = torch.randn(n_per_class, *IMG_SHAPE) * 0.3 - 2.0
    x1 = torch.randn(n_per_class, *IMG_SHAPE) * 0.3 + 2.0
    x = torch.cat([x0, x1], dim=0)
    y = torch.cat([torch.zeros(n_per_class, dtype=torch.long),
                   torch.ones(n_per_class, dtype=torch.long)], dim=0)
    idx = torch.randperm(len(x))
    return x[idx], y[idx]


def _train_vae(x):
    torch.manual_seed(42)
    vae = ToyVAE()
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    n_epochs = 200
    batch_size = 64
    N = x.shape[0]
    for epoch in range(n_epochs):
        perm = torch.randperm(N)
        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            xb = x[idx]
            x_hat, mu, log_var = vae(xb)
            recon = ((x_hat - xb) ** 2).sum(dim=(1, 2, 3)).mean()
            kld = -0.5 * (1 + log_var - mu ** 2 - log_var.exp()).sum(dim=1).mean()
            loss = recon + 0.1 * kld
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return vae


def _encode_latents(vae, x):
    vae.eval()
    with torch.no_grad():
        mu, _ = vae.encoder(x)
    return mu


def _train_classifier(z, y):
    torch.manual_seed(42)
    clf = nn.Linear(DZ, 2, bias=False)
    optimizer = torch.optim.Adam(clf.parameters(), lr=0.01)
    for _ in range(500):
        loss = nn.functional.cross_entropy(clf(z), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return clf


def get_toy_models():
    """Returns (decoder, classifier, z_train, y_train).

    Loads from cache on subsequent calls; trains from scratch on first call.
    Training completes in under 60 seconds on CPU.
    """
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if CACHE_PATH.exists():
        ckpt = torch.load(CACHE_PATH, weights_only=False)
        vae = ToyVAE()
        vae.load_state_dict(ckpt["vae"])
        clf = nn.Linear(DZ, 2, bias=False)
        clf.load_state_dict(ckpt["classifier"])
        z_train = ckpt["z_train"]
        y_train = ckpt["y_train"]
    else:
        x, y = _make_dataset()
        vae = _train_vae(x)
        z_train = _encode_latents(vae, x)
        clf = _train_classifier(z_train, y)
        torch.save({
            "vae": vae.state_dict(),
            "classifier": clf.state_dict(),
            "z_train": z_train,
            "y_train": y,
        }, CACHE_PATH)
        y_train = y

    vae.eval()
    clf.eval()
    return vae.decoder, clf, z_train, y_train
