"""MINE (Belghazi et al., 2018) mutual-information estimate between latents and warped latents."""
from typing import Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

# Seed used only for the evaluation permutation generator, making the MI
# estimate reproducible regardless of global torch random state (critical for
# cache-hit vs cache-miss consistency).
_EVAL_PERM_SEED = 42
_EVAL_K = 20  # permutations to average over for variance reduction


class _MINENet(nn.Module):
    def __init__(self, dz: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dz, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)


def _eval_mi(net: nn.Module, z_original: Tensor, z_warped: Tensor) -> float:
    N = z_original.shape[0]
    # Use a fixed-seed generator so the permutation sequence is identical
    # regardless of the global random state (training path vs cache-hit path).
    gen = torch.Generator()
    gen.manual_seed(_EVAL_PERM_SEED)
    with torch.no_grad():
        joint = torch.cat([z_original, z_warped], dim=1)
        t_joint = net(joint)
        mi_estimates = []
        for _ in range(_EVAL_K):
            perm = torch.randperm(N, generator=gen).to(z_original.device)
            marginal = torch.cat([z_original, z_warped[perm]], dim=1)
            t_marginal = net(marginal)
            mi_k = (t_joint.mean() - torch.logsumexp(t_marginal, dim=0) + torch.log(torch.tensor(float(N)))).item()
            mi_estimates.append(mi_k)
    return sum(mi_estimates) / len(mi_estimates)


def _train_mine(
    z_original: Tensor,
    z_warped: Tensor,
    n_epochs: int,
    hidden_dim: int,
    seed: int,
) -> tuple:
    dz = z_original.shape[1]
    # Seed network init and the training permutations locally, without touching
    # the caller's global RNG state.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = _MINENet(dz, hidden_dim).to(z_original.device)
    gen = torch.Generator()
    gen.manual_seed(seed)
    # weight_decay regularises T toward 0, suppressing overfitting-driven bias
    # for independent variables where the true optimum is T*=0.
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    N = z_original.shape[0]
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        joint = torch.cat([z_original, z_warped], dim=1)
        t_joint = net(joint)

        perm = torch.randperm(N, generator=gen).to(z_original.device)
        marginal = torch.cat([z_original, z_warped[perm]], dim=1)
        t_marginal = net(marginal)

        loss = -(t_joint.mean() - torch.logsumexp(t_marginal, dim=0) + torch.log(torch.tensor(float(N))))
        loss.backward()
        optimizer.step()

    mi = _eval_mi(net, z_original, z_warped)
    return net, mi


def estimate_mi(
    z_original: Tensor,
    z_warped: Tensor,
    n_epochs: int = 200,
    hidden_dim: int = 64,
    save_path: Optional[str] = None,
    seed: int = 0,
) -> float:
    """Estimate I(z_original; z_warped) in nats with a MINE critic network.

    The critic is trained for ``n_epochs`` full-batch steps; the final estimate
    averages the Donsker-Varadhan bound over 20 fixed permutations. The
    estimator has a known finite-sample positive bias (~0.2 nats for
    independent 4-d Gaussians at N=1024). If ``save_path`` exists, the cached
    critic is reloaded instead of retrained (hyperparameters must match).
    ``seed`` fixes network init and training permutations.
    """
    dz = z_original.shape[1]

    if save_path is not None and Path(save_path).exists():
        checkpoint = torch.load(save_path, map_location=z_original.device, weights_only=True)
        if checkpoint["n_epochs"] != n_epochs or checkpoint["hidden_dim"] != hidden_dim:
            raise ValueError(
                f"Cached hyperparams (n_epochs={checkpoint['n_epochs']}, "
                f"hidden_dim={checkpoint['hidden_dim']}) do not match "
                f"current call (n_epochs={n_epochs}, hidden_dim={hidden_dim})."
            )
        net = _MINENet(dz, hidden_dim).to(z_original.device)
        net.load_state_dict(checkpoint["state_dict"])
        net.eval()
        return _eval_mi(net, z_original.float(), z_warped.float())

    net, mi = _train_mine(z_original.float(), z_warped.float(), n_epochs, hidden_dim, seed)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": net.state_dict(),
            "n_epochs": n_epochs,
            "hidden_dim": hidden_dim,
        }, save_path)

    return mi
