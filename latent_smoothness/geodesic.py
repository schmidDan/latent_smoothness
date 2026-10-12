import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from latent_smoothness.utils import eval_mode


@dataclass
class PullbackResult:
    G: Tensor
    det_G: Tensor
    log_det_G: Tensor


@dataclass
class GeodesicResult:
    chord_length: Tensor          # (M,) ‖f(z_end) − f(z_start)‖ in decoder OUTPUT space
    geodesic_path_length: Tensor  # (M,) output-space arc length of the straight z-path
    discrepancy: Tensor           # (M,) (arc − chord) / chord; 0 for a linear decoder
    pullback_along_path: Tensor   # (M, n_steps) |det G(z_t)| along the path
    solver_failed: bool = False  # adaptive ODE could not integrate (e.g. dt underflow)


def _pullback_metric_at_points(decoder: nn.Module, z_points: Tensor) -> Tensor:
    """Returns G(z) = J^T J, shape (N, dz, dz). Assumes eval mode is set by caller.

    Uses torch.enable_grad() to ensure Jacobians can be computed even inside
    torchdiffeq's internal no-grad context.
    """
    N, dz = z_points.shape
    G_list = []

    with torch.enable_grad():
        for i in range(N):
            z = z_points[i].detach().clone().requires_grad_(True)  # (dz,)
            out = decoder(z.unsqueeze(0)).reshape(-1)  # (dx,)
            dx = out.shape[0]
            eye = torch.eye(dx, device=out.device, dtype=out.dtype)
            J = torch.autograd.grad(out, z, grad_outputs=eye, is_grads_batched=True, allow_unused=True)[0]  # (dx, dz)
            if J is None:  # non-differentiable variant: zero Jacobian (flat cell)
                J = torch.zeros(dx, dz, device=out.device, dtype=out.dtype)
            G = J.T @ J  # (dz, dz)
            G_list.append(G.detach())

    return torch.stack(G_list, dim=0)  # (N, dz, dz)


def _compute_pullback_metric_raw(decoder: nn.Module, z_points: Tensor) -> PullbackResult:
    G = _pullback_metric_at_points(decoder, z_points)
    det_G = torch.linalg.det(G)
    log_det_G = torch.log(det_G.abs().clamp(min=1e-40))
    return PullbackResult(G=G, det_G=det_G, log_det_G=log_det_G)


def compute_pullback_metric(decoder: nn.Module, z_points: Tensor) -> PullbackResult:
    with eval_mode(decoder):
        return _compute_pullback_metric_raw(decoder, z_points)


def _pullback_speed(G: Tensor, dz_dt: Tensor) -> Tensor:
    """sqrt(dz_dt^T G dz_dt) for batch. G: (N,dz,dz), dz_dt: (N,dz) -> (N,)"""
    sq = torch.bmm(
        dz_dt.unsqueeze(1),
        torch.bmm(G, dz_dt.unsqueeze(2))
    ).squeeze(-1).squeeze(-1).clamp(min=0.0)
    return sq.sqrt()


def _arc_length_euler(decoder, z_start, z_end, n_steps):
    M, dz = z_start.shape
    dz_dt = z_end - z_start
    dt = 1.0 / (n_steps - 1)
    waypoints = torch.linspace(0, 1, n_steps, device=z_start.device, dtype=z_start.dtype)
    pullback_list = []
    length = torch.zeros(M, device=z_start.device, dtype=z_start.dtype)
    for i, t in enumerate(waypoints):
        z_t = z_start + t * dz_dt
        G = _pullback_metric_at_points(decoder, z_t)
        speed = _pullback_speed(G, dz_dt)
        pullback_list.append(torch.linalg.det(G).abs())
        if i > 0:
            length = length + speed * dt
    return length, torch.stack(pullback_list, dim=1)


def _arc_length_rk4(decoder, z_start, z_end, n_steps):
    M, dz = z_start.shape
    dz_dt = z_end - z_start
    dt = 1.0 / (n_steps - 1)
    waypoints = torch.linspace(0, 1, n_steps, device=z_start.device, dtype=z_start.dtype)
    pullback_list = []
    length = torch.zeros(M, device=z_start.device, dtype=z_start.dtype)

    def speed_at(t_val):
        z_t = z_start + t_val * dz_dt
        G = _pullback_metric_at_points(decoder, z_t)
        return _pullback_speed(G, dz_dt)

    for i, t in enumerate(waypoints):
        z_t = z_start + t * dz_dt
        G = _pullback_metric_at_points(decoder, z_t)
        pullback_list.append(torch.linalg.det(G).abs())
        if i > 0:
            t_prev = waypoints[i - 1]
            k1 = speed_at(t_prev)
            k2 = speed_at(t_prev + dt / 2)
            k3 = speed_at(t_prev + dt / 2)
            k4 = speed_at(t)
            length = length + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return length, torch.stack(pullback_list, dim=1)


def _arc_length_torchdiffeq(decoder, z_start, z_end, n_steps, rtol, atol, adjoint=False):
    import torchdiffeq
    M, dz = z_start.shape
    dz_dt = z_end - z_start

    def odefunc(t, state):
        z_t = z_start + t * dz_dt
        G = _pullback_metric_at_points(decoder, z_t)
        speed = _pullback_speed(G, dz_dt)
        return speed.unsqueeze(1)

    t_eval = torch.linspace(0.0, 1.0, n_steps, device=z_start.device, dtype=z_start.dtype)
    state0 = torch.zeros(M, 1, device=z_start.device, dtype=z_start.dtype)

    kwargs = dict(rtol=rtol, atol=atol, method="dopri5")
    if adjoint:
        kwargs["adjoint_params"] = ()
        sol = torchdiffeq.odeint_adjoint(odefunc, state0, t_eval, **kwargs)
    else:
        sol = torchdiffeq.odeint(odefunc, state0, t_eval, **kwargs)

    length = sol[-1, :, 0]

    pullback_list = []
    for t in t_eval:
        z_t = z_start + t * dz_dt
        G = _pullback_metric_at_points(decoder, z_t)
        pullback_list.append(torch.linalg.det(G).abs())

    return length, torch.stack(pullback_list, dim=1)


def _compute_geodesic_discrepancy_raw(
    decoder: nn.Module,
    z_start: Tensor,
    z_end: Tensor,
    n_steps: int = 20,
    solver: str = "torchdiffeq",
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> GeodesicResult:
    # chord_length = chord in output (x) space = ||f(z_end) - f(z_start)||
    # This equals the arc length for any LINEAR decoder (constant Jacobian → straight path in x)
    with torch.no_grad():
        x_start = decoder(z_start).reshape(z_start.shape[0], -1)  # (M, dx)
        x_end = decoder(z_end).reshape(z_end.shape[0], -1)
    chord_length = torch.norm(x_end - x_start, dim=-1)  # (M,)

    # geodesic_path_length = arc length in x-space along linear z-interpolation
    # = integral ||J(z_t)*Δz|| dt = integral sqrt(Δz^T G(z_t) Δz) dt
    if solver not in ("euler", "rk4", "torchdiffeq", "torchdiffeq_adjoint"):
        raise ValueError(f"Unknown solver: {solver!r}")

    solver_failed = False
    try:
        if solver == "euler":
            geo_length, pullback = _arc_length_euler(decoder, z_start, z_end, n_steps)
        elif solver == "rk4":
            geo_length, pullback = _arc_length_rk4(decoder, z_start, z_end, n_steps)
        elif solver == "torchdiffeq":
            geo_length, pullback = _arc_length_torchdiffeq(decoder, z_start, z_end, n_steps, rtol, atol, adjoint=False)
        else:  # torchdiffeq_adjoint
            geo_length, pullback = _arc_length_torchdiffeq(decoder, z_start, z_end, n_steps, rtol, atol, adjoint=True)
    except AssertionError as e:
        # ONLY the documented adaptive-ODE 'underflow in dt' on a stiff pullback
        # metric. Deliberately narrow: real RuntimeErrors (device/OOM/cuBLAS)
        # must surface, not be masked as a NaN geodesic.
        # Report NaN rather than crash the suite or silently swap solvers
        # (which would make this trajectory incomparable to the others).
        warnings.warn(
            f"compute_geodesic_discrepancy: solver '{solver}' failed "
            f"({type(e).__name__}: {e}); returning NaN geodesic for this set."
        )
        solver_failed = True
        M = z_start.shape[0]
        nan = float("nan")
        geo_length = torch.full((M,), nan, device=z_start.device, dtype=z_start.dtype)
        pullback = torch.full((M, n_steps), nan, device=z_start.device, dtype=z_start.dtype)

    discrepancy = (geo_length - chord_length) / chord_length.clamp(min=1e-12)
    return GeodesicResult(
        chord_length=chord_length,
        geodesic_path_length=geo_length,
        discrepancy=discrepancy,
        pullback_along_path=pullback,
        solver_failed=solver_failed,
    )


def compute_geodesic_discrepancy(
    decoder: nn.Module,
    z_start: Tensor,
    z_end: Tensor,
    n_steps: int = 20,
    solver: str = "torchdiffeq",
    rtol: float = 1e-5,
    atol: float = 1e-7,
) -> GeodesicResult:
    """Output-space arc length of the straight latent path z_start -> z_end vs. its chord.

    ``solver`` is one of ``"euler"``, ``"rk4"``, ``"torchdiffeq"`` (adaptive
    dopri5) or ``"torchdiffeq_adjoint"``. If the adaptive solver fails (dt
    underflow on a stiff metric) the result is NaN with ``solver_failed=True``.
    """
    with eval_mode(decoder):
        return _compute_geodesic_discrepancy_raw(decoder, z_start, z_end, n_steps, solver, rtol, atol)
