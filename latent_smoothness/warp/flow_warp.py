"""Normalizing-flow warp: a learned bijection from the latent distribution onto a target."""
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from latent_smoothness.utils import eval_mode, set_global_seed


@dataclass
class WarpDiagnosticResult:
    kl_to_target: float
    kl_to_original: float
    density_grid: Tensor
    sample_plot_data: Tensor
    n_input: int = 0       # rows passed to verify()
    n_dropped: int = 0     # non-finite warped rows excluded from KL/KDE


def _knn_kl(p_samples: np.ndarray, q_samples: np.ndarray, k: int = 5) -> float:
    from sklearn.neighbors import NearestNeighbors
    n, d = p_samples.shape
    m = q_samples.shape[0]

    nbrs_p = NearestNeighbors(n_neighbors=k + 1).fit(p_samples)
    dists_p, _ = nbrs_p.kneighbors(p_samples)
    rho = dists_p[:, k]

    nbrs_q = NearestNeighbors(n_neighbors=k).fit(q_samples)
    dists_q, _ = nbrs_q.kneighbors(p_samples)
    nu = dists_q[:, k - 1]

    kl = d * np.mean(np.log(nu / np.maximum(rho, 1e-12))) + np.log(m / (n - 1))
    return float(np.maximum(kl, 0.0))


def _make_target_samples(target: str, n: int, dz: int, target_samples: Optional[Tensor]) -> np.ndarray:
    if target == "mixture_of_gaussians":
        n_comp = 4
        means = np.array([[2, 2], [-2, 2], [-2, -2], [2, -2]])
        if dz > 2:
            means = np.pad(means, ((0, 0), (0, dz - 2)))
        samples = []
        per = n // n_comp
        for i in range(n_comp):
            s = np.random.randn(per, dz) * 0.5 + means[i]
            samples.append(s)
        return np.concatenate(samples, axis=0).astype(np.float32)
    elif target == "banana":
        t = np.random.randn(n)
        x = t
        y = t ** 2 / 4 + np.random.randn(n) * 0.3
        base = np.stack([x, y], axis=1)
        if dz > 2:
            extra = np.random.randn(n, dz - 2) * 0.1
            base = np.concatenate([base, extra], axis=1)
        return base.astype(np.float32)
    elif target == "uniform":
        return (np.random.rand(n, dz) * 6 - 3).astype(np.float32)
    elif target == "random_affine_cells":
        # Mixture of 8 randomly placed, randomly sheared Gaussian blobs.
        n_cells = 8
        centroids = np.random.randn(n_cells, dz)
        assignments = np.random.randint(0, n_cells, size=n)
        affines = [np.random.randn(dz, dz) * 0.3 + np.eye(dz) for _ in range(n_cells)]
        base = np.random.randn(n, dz)
        result = np.zeros_like(base)
        for i, a in enumerate(centroids):
            mask = assignments == i
            result[mask] = base[mask] @ affines[i] + a
        return result.astype(np.float32)
    elif target == "custom":
        if target_samples is None:
            raise ValueError("target='custom' requires target_samples to be provided.")
        return target_samples.detach().cpu().numpy().astype(np.float32)
    else:
        raise ValueError(f"Unknown target: {target!r}")


def _build_flow(dz: int):
    try:
        import normflows as nf
    except ImportError as e:
        raise ImportError(
            "FlowWarp needs the optional 'normflows' package: "
            "pip install 'latent_smoothness[flow]'"
        ) from e
    K = 8
    hidden_units = max(32, dz * 4)
    mlp_in = max(dz // 2, 1)
    flows = []
    for _ in range(K):
        flows.append(nf.flows.AffineCouplingBlock(
            nf.nets.MLP([mlp_in, hidden_units, dz], init_zeros=True)
        ))
        flows.append(nf.flows.Permute(dz, mode="swap"))
    q0 = nf.distributions.DiagGaussian(dz, trainable=False)
    return nf.NormalizingFlow(q0=q0, flows=flows)


def _train_flow(flow, z_data: Tensor, n_epochs: int) -> None:
    optimizer = torch.optim.Adam(flow.parameters(), lr=5e-4)
    best_state = None
    best_loss = float("inf")
    n_skipped = 0
    last_error = None
    flow.train()
    for _ in range(n_epochs):
        optimizer.zero_grad()
        try:
            loss = flow.forward_kld(z_data)
        except (RuntimeError, ValueError) as e:  # numerical failure in a coupling layer
            n_skipped += 1
            last_error = e
            continue
        if not torch.isfinite(loss):
            n_skipped += 1
            continue
        loss_val = loss.item()
        loss.backward()
        any_nan_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all()
            for p in flow.parameters()
        )
        if any_nan_grad:
            optimizer.zero_grad()
            n_skipped += 1
            continue
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 0.5)
        optimizer.step()
        if loss_val < best_loss:
            best_loss = loss_val
            best_state = {k: v.clone() for k, v in flow.state_dict().items()}
    if best_state is None:
        raise RuntimeError(
            f"Flow training failed: all {n_epochs} steps were non-finite or raised "
            f"(last error: {last_error!r})."
        )
    if n_skipped:
        warnings.warn(f"Flow training skipped {n_skipped}/{n_epochs} non-finite steps.")
    flow.load_state_dict(best_state)


class FlowWarp(nn.Module):
    """Bijective warp mapping z → target distribution via two composed flows.

    Targets: ``"mixture_of_gaussians"`` (4 blobs), ``"banana"``, ``"uniform"``,
    ``"random_affine_cells"`` (8 random sheared blobs), or ``"custom"`` with
    ``target_samples``. Requires the optional ``normflows`` package.

    fit() trains:
      _flow_z     : forward_kld on z       → z.inverse(z) ≈ N(0,I)
      _flow_target: forward_kld on target  → _flow_target.forward(u) ≈ target

    forward(z)  = _flow_target.forward(_flow_z.inverse(z))   # z → target
    inverse(t)  = _flow_z.forward(_flow_target.inverse(t))   # target → z
    Round-trip: inverse(forward(z)) ≈ z  (exact up to float precision)
    """

    def __init__(self):
        super().__init__()
        self._flow_z = None       # normalises z → Gaussian base
        self._flow_target = None  # models target; generative dir → target
        self._dz = None
        # Registered as buffers (not plain attrs) so they appear in
        # state_dict() and survive save/load. PyTorch routes a later tensor (or
        # None) assignment to one of these names into self._buffers, so fit()
        # below keeps working unchanged. _dz stays a plain int (not a tensor).
        self.register_buffer("_target_samples", None)
        self.register_buffer("_z_mean", None)  # set only when fit(standardize=True)
        self.register_buffer("_z_std", None)

    def fit(
        self,
        z: Tensor,
        target: str = "mixture_of_gaussians",
        target_samples: Optional[Tensor] = None,
        n_epochs: int = 500,
        standardize: bool = False,
    ) -> None:
        """Fit the z→target warp.

        standardize: if True, z-score the z-flow input (per-dim mean/std from
        the training batch) before training and undo it in inverse(). The base
        distribution is N(0, I), so standardizing heavy-tailed latents keeps the
        flow inverse in a well-conditioned range and avoids exp-overflow at the
        tails. Default False (unchanged behaviour).
        """
        if target == "custom" and target_samples is None:
            raise ValueError("target='custom' requires target_samples.")

        N, dz = z.shape
        self._dz = dz

        np_target = _make_target_samples(target, N, dz, target_samples)
        self._target_samples = torch.from_numpy(np_target).float()

        z_train = z.detach().float()

        if standardize:
            self._z_mean = z_train.mean(dim=0, keepdim=True)
            self._z_std = z_train.std(dim=0, keepdim=True).clamp_min(1e-6)
            z_train = (z_train - self._z_mean) / self._z_std
        else:
            self._z_mean = None
            self._z_std = None

        # Flow 1: learn to normalise z; inverse(z) pushes z toward N(0,I)
        self._flow_z = _build_flow(dz)
        _train_flow(self._flow_z, z_train, n_epochs)

        # Flow 2: learn to model target; forward(u) samples from target
        self._flow_target = _build_flow(dz)
        _train_flow(self._flow_target, self._target_samples, n_epochs)

    def forward(self, z: Tensor) -> Tensor:
        if self._flow_z is None:
            raise RuntimeError("FlowWarp.fit() must be called before forward() or inverse().")
        # z → target: normalise z to base, then generate from target flow
        with eval_mode(self):
            z_in = z.float()
            if self._z_mean is not None:
                z_in = (z_in - self._z_mean.to(z_in)) / self._z_std.to(z_in)
            z_base = self._flow_z.inverse(z_in)
            z_target = self._flow_target.forward(z_base)
        return z_target.to(dtype=z.dtype)

    def inverse(self, z_warped: Tensor) -> Tensor:
        if self._flow_z is None:
            raise RuntimeError("FlowWarp.fit() must be called before forward() or inverse().")
        # target → z: normalise from target space, then generate via z flow
        with eval_mode(self):
            z_base = self._flow_target.inverse(z_warped.float())
            z_orig = self._flow_z.forward(z_base)
            if self._z_mean is not None:
                z_orig = z_orig * self._z_std.to(z_orig) + self._z_mean.to(z_orig)
        return z_orig.to(dtype=z_warped.dtype)

    def save(self, path: Union[str, Path]) -> None:
        """Persist the fitted warp to ``path`` (a torch ``.pt`` blob).

        Saves the full state_dict (both flows + the registered buffers
        ``_target_samples`` / ``_z_mean`` / ``_z_std``), the latent dimension
        ``_dz`` (a plain int, not in the state_dict), and a ``standardized``
        flag. Call only after fit(); reload with :meth:`load`.
        """
        torch.save(
            {
                "state_dict": self.state_dict(),
                "_dz": self._dz,
                "standardized": self._z_mean is not None,
            },
            path,
        )

    @classmethod
    def load(cls, path: Union[str, Path], map_location="cpu") -> "FlowWarp":
        """Reconstruct a fitted ``FlowWarp`` from a :meth:`save` blob.

        Rebuilds both flow architectures for ``_dz`` so ``load_state_dict``
        matches, then restores parameters and buffers. ``map_location`` is
        forwarded to ``torch.load`` so the warp can be materialised directly on
        e.g. ``"cuda"``. Returns a ready-to-use warp (call before any
        forward/inverse).
        """
        blob = torch.load(path, map_location=map_location, weights_only=True)
        w = cls()
        w._dz = blob["_dz"]
        w._flow_z = _build_flow(w._dz)
        w._flow_target = _build_flow(w._dz)
        # Buffers were registered as None in __init__. PyTorch EXCLUDES
        # None-valued buffers from a module's local_state during
        # load_state_dict, so a buffer key present in the checkpoint (always
        # _target_samples; _z_mean/_z_std only when standardized) would be
        # flagged UNEXPECTED and the strict load would RAISE. Pre-assign the
        # buffers that exist in the checkpoint (now real tensors -> they enter
        # local_state); buffers absent from it stay None and are skipped on
        # both sides.
        sd = blob["state_dict"]
        for b in ("_z_mean", "_z_std", "_target_samples"):
            if b in sd:
                setattr(w, b, sd[b].clone())  # routes into _buffers as a tensor
        w.load_state_dict(sd)                 # strict OK: flows + present buffers match
        return w

    def verify(self, z: Tensor) -> WarpDiagnosticResult:
        with torch.no_grad():
            z_warped = self.forward(z)

        z_np = z.detach().cpu().numpy().astype(np.float32)
        z_warped_np = z_warped.detach().cpu().numpy().astype(np.float32)
        target_np = self._target_samples.numpy().astype(np.float32)

        # Flow forward/inverse can overflow to inf/NaN in the distribution
        # tails (e.g. for heavy-tailed latents). Drop those rows so the
        # kNN-KL estimator and KDE get finite input; if almost everything blew
        # up, the warp genuinely diverged — report NaN rather than a fake number.
        finite = np.isfinite(z_warped_np).all(axis=1)
        n_input = int(finite.size)
        n_bad = int((~finite).sum())
        if n_bad:
            warnings.warn(
                f"FlowWarp.verify: dropping {n_bad}/{n_input} non-finite "
                f"warped samples before KL/KDE."
            )
        z_warped_np = z_warped_np[finite]
        z_np = z_np[finite]
        if z_warped_np.shape[0] < 10:
            return WarpDiagnosticResult(
                kl_to_target=float("nan"),
                kl_to_original=float("nan"),
                density_grid=torch.zeros(50, 50),
                sample_plot_data=torch.zeros(0, 2),
                n_input=n_input,
                n_dropped=n_bad,
            )

        kl_to_target = _knn_kl(z_warped_np, target_np)
        kl_to_original = _knn_kl(z_warped_np, z_np)

        dz = z.shape[1]
        if dz == 2:
            proj = z_warped_np
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            proj = pca.fit_transform(z_warped_np)

        grid_size = 50
        x_min, x_max = proj[:, 0].min() - 1, proj[:, 0].max() + 1
        y_min, y_max = proj[:, 1].min() - 1, proj[:, 1].max() + 1
        xs = np.linspace(x_min, x_max, grid_size)
        ys = np.linspace(y_min, y_max, grid_size)
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(bandwidth=0.5).fit(proj)
        xx, yy = np.meshgrid(xs, ys)
        grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
        log_dens = kde.score_samples(grid_pts)
        density_grid = torch.from_numpy(np.exp(log_dens).reshape(grid_size, grid_size).astype(np.float32))

        sample_plot_data = torch.from_numpy(proj[:, :2].astype(np.float32))

        return WarpDiagnosticResult(
            kl_to_target=kl_to_target,
            kl_to_original=kl_to_original,
            density_grid=density_grid,
            sample_plot_data=sample_plot_data,
            n_input=n_input,
            n_dropped=n_bad,
        )


def fit_flow_ensemble(
    z: Tensor,
    n: int,
    seeds: Sequence[int],
    out_dir: Union[str, Path],
    target: str = "mixture_of_gaussians",
    n_epochs: int = 300,
    standardize: bool = False,
) -> List[Path]:
    """Fit and persist an ensemble of ``n`` independently-seeded FlowWarps.

    Each member is fit on the same training latents ``z`` (shape ``(N, dz)``)
    but with a distinct global seed, so the ensemble samples the fit's
    stochasticity (flow initialisation and target samples). Every fit is preceded by
    :func:`latent_smoothness.utils.set_global_seed` for reproducibility, and the
    fitted warp is saved to ``out_dir/flow_seed{seed}.pt`` via
    :meth:`FlowWarp.save`.

    Args:
        z: training latents, ``(N, dz)``.
        n: number of ensemble members (must equal ``len(seeds)``).
        seeds: the ``n`` integer seeds, one per member.
        out_dir: directory to write the checkpoints into (created if absent).
        target, n_epochs, standardize: forwarded to :meth:`FlowWarp.fit`.

    Returns:
        The list of written checkpoint paths, in ``seeds`` order.
    """
    seeds = list(seeds)
    if len(seeds) != n:
        raise ValueError(f"len(seeds)={len(seeds)} must equal n={n}.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    for seed in seeds:
        set_global_seed(seed)
        warp = FlowWarp()
        warp.fit(z, target=target, n_epochs=n_epochs, standardize=standardize)
        path = out_dir / f"flow_seed{seed}.pt"
        warp.save(path)
        paths.append(path)
    return paths
