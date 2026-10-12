"""Run all diagnostics on a decoder/classifier pair in one call.

Point-set diagnostics (Jacobian, LID, label dimensionality, boundary geometry)
are computed on a *reference* point set and, optionally, a *query* point set,
e.g. latents of typical vs. atypical inputs. Trajectory diagnostics (geodesic,
PPL, boundary crossings, gradient alignment, secant roughness, label coherence)
are computed along the straight paths between ``trajectory_endpoints``.
"""
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from latent_smoothness.jacobian import compute_jacobian_spectrum, JacobianResult
from latent_smoothness.geodesic import compute_geodesic_discrepancy, GeodesicResult
from latent_smoothness.ppl import compute_ppl, PPLResult
from latent_smoothness.lid import compute_lid, LIDResult
from latent_smoothness.boundary import (
    compute_boundary_geometry, BoundaryPointResult,
    compute_boundary_trajectory, BoundaryTrajectoryResult,
)
from latent_smoothness.alignment import compute_gradient_alignment, AlignmentResult
from latent_smoothness.roughness import compute_secant_roughness, SecantRoughnessResult
from latent_smoothness.label_coherence import compute_label_coherence, LabelCoherenceResult
from latent_smoothness.label_dimensionality import (
    compute_label_dimensionality, LabelDimensionalityResult,
)
from latent_smoothness.utils import linear_interpolate, resolve_device, timed_step


@dataclass
class SuiteConfig:
    """Which diagnostics :class:`DiagnosticSuite` runs, and their settings.

    ``n_samples`` subsamples each point set (seeded by ``seed``) before the
    point-set diagnostics; ``None`` uses all points. ``device`` is where LPIPS
    runs (``None`` = CUDA if available, else CPU). ``class_idx`` is the class
    whose decision margin the boundary and alignment diagnostics use.
    ``verbose`` prints per-step progress and timings to stderr.
    """
    run_jacobian: bool = True
    run_geodesic: bool = True
    run_ppl: bool = True
    run_lid: bool = True
    run_boundary_point: bool = True
    run_boundary_trajectory: bool = True
    run_alignment: bool = True
    run_secant_roughness: bool = True
    run_label_coherence: bool = True
    run_label_dimensionality: bool = True
    class_idx: int = 0
    geodesic_solver: str = "torchdiffeq"
    geodesic_n_steps: int = 20
    ppl_n_steps: int = 100
    secant_roughness_n_steps: int = 32
    boundary_trajectory_n_steps: int = 50
    label_coherence_n_steps: int = 64
    label_dim_min_class_count: int = 5
    n_samples: Optional[int] = None
    seed: int = 0
    device: Optional[str] = None
    verbose: bool = False


_TRAJECTORY_FLAGS = (
    "run_geodesic", "run_ppl", "run_boundary_trajectory",
    "run_alignment", "run_secant_roughness", "run_label_coherence",
)


@dataclass
class SuiteResult:
    jacobian_reference: Optional[JacobianResult] = None
    lid_reference: Optional[LIDResult] = None
    boundary_point_reference: Optional[BoundaryPointResult] = None
    label_dimensionality_reference: Optional[LabelDimensionalityResult] = None
    jacobian_query: Optional[JacobianResult] = None
    lid_query: Optional[LIDResult] = None
    boundary_point_query: Optional[BoundaryPointResult] = None
    label_dimensionality_query: Optional[LabelDimensionalityResult] = None
    geodesic: Optional[GeodesicResult] = None
    ppl: Optional[PPLResult] = None
    boundary_trajectory: Optional[BoundaryTrajectoryResult] = None
    alignment: Optional[AlignmentResult] = None
    secant_roughness: Optional[SecantRoughnessResult] = None
    label_coherence: Optional[LabelCoherenceResult] = None

    def to_dict(self) -> dict:
        """Flatten every result field to ``{"<diagnostic>.<attr>": value}`` (tensors as lists)."""
        result = {}
        for fname, fval in self.__dict__.items():
            if fval is None:
                result[fname] = None
            elif hasattr(fval, "__dict__"):
                for attr, val in fval.__dict__.items():
                    key = f"{fname}.{attr}"
                    if isinstance(val, Tensor):
                        result[key] = val.tolist()
                    else:
                        result[key] = val
            else:
                result[fname] = fval
        return result

    def summary_table(self) -> str:
        """Human-readable summary of the headline scalar of each diagnostic."""
        lines = ["=" * 60, "DiagnosticSuite Summary", "=" * 60]

        def _add(name, obj, attrs):
            if obj is None:
                lines.append(f"  {name}: not computed")
                return
            lines.append(f"  {name}:")
            for a in attrs:
                val = getattr(obj, a, None)
                if val is None:
                    continue
                if isinstance(val, Tensor):
                    lines.append(f"    {a}: shape={tuple(val.shape)}")
                elif isinstance(val, float):
                    lines.append(f"    {a}: {val:.4f}")
                else:
                    lines.append(f"    {a}: {val}")

        for tag in ("reference", "query"):
            _add(f"jacobian_{tag}", getattr(self, f"jacobian_{tag}"),
                 ["lipschitz_estimate", "gradient_degenerate", "n_zero_fill"])
            _add(f"lid_{tag}", getattr(self, f"lid_{tag}"), ["mean_lid_twonn", "mean_lid_mle"])
            _add(f"boundary_point_{tag}", getattr(self, f"boundary_point_{tag}"), ["margins"])
            _add(f"label_dimensionality_{tag}", getattr(self, f"label_dimensionality_{tag}"),
                 ["within_class_effrank", "class_diff_effrank"])
        _add("geodesic", self.geodesic, ["discrepancy", "solver_failed"])
        _add("ppl", self.ppl, ["mean_ppl"])
        _add("boundary_trajectory", self.boundary_trajectory, ["n_crossings"])
        _add("alignment", self.alignment, ["mean_alignment", "fraction_aligned"])
        _add("secant_roughness", self.secant_roughness, ["loglog_slope"])
        _add("label_coherence", self.label_coherence, ["mean_label_flips", "mean_monotonic_gain"])
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_scalars(self) -> dict:
        """Flat ``{name: int | float}`` dict of scalar summaries, e.g. for experiment loggers."""
        result = {}

        def _scalar(prefix, obj, attrs):
            if obj is None:
                return
            for a in attrs:
                val = getattr(obj, a, None)
                if isinstance(val, (int, float)):
                    result[f"{prefix}.{a}"] = val
                elif isinstance(val, Tensor) and val.numel() == 1:
                    result[f"{prefix}.{a}"] = val.item()

        for tag in ("reference", "query"):
            _scalar(f"jacobian_{tag}", getattr(self, f"jacobian_{tag}"),
                    ["lipschitz_estimate", "gradient_degenerate"])
            _scalar(f"lid_{tag}", getattr(self, f"lid_{tag}"), ["mean_lid_twonn", "mean_lid_mle"])
            bp = getattr(self, f"boundary_point_{tag}")
            if bp is not None:
                result[f"boundary_point_{tag}.mean_margin"] = bp.margins.mean().item()
                result[f"boundary_point_{tag}.mean_curvature"] = bp.curvature.mean().item()
            _scalar(f"label_dimensionality_{tag}", getattr(self, f"label_dimensionality_{tag}"),
                    ["within_class_effrank", "within_class_spread",
                     "class_diff_effrank", "class_diff_consistency", "fisher_ratio"])
        if self.geodesic is not None:
            result["geodesic.mean_discrepancy"] = self.geodesic.discrepancy.mean().item()
            result["geodesic.max_discrepancy"] = self.geodesic.discrepancy.max().item()
        _scalar("ppl", self.ppl, ["mean_ppl"])
        _scalar("boundary_trajectory", self.boundary_trajectory, ["n_crossings"])
        _scalar("alignment", self.alignment, ["mean_alignment", "fraction_aligned"])
        _scalar("secant_roughness", self.secant_roughness, ["loglog_slope"])
        _scalar("label_coherence", self.label_coherence,
                ["mean_label_flips", "mean_monotonic_gain", "mean_target_prob_gain"])
        return result


class DiagnosticSuite:
    """Runs the diagnostics selected in a :class:`SuiteConfig`.

    Args:
        decoder: maps latents ``(B, dz)`` to outputs (images ``(B, C, H, W)``
            if PPL is enabled).
        classifier: maps latents ``(B, dz)`` to logits ``(B, C)``.
        config: which diagnostics to run; defaults to all.
    """

    def __init__(
        self,
        decoder: nn.Module,
        classifier: nn.Module,
        config: Optional[SuiteConfig] = None,
    ):
        self.decoder = decoder
        self.classifier = classifier
        self.config = config if config is not None else SuiteConfig()

    def _subsample(self, z: Tensor, generator: torch.Generator) -> Tensor:
        n = self.config.n_samples
        if n is None:
            return z
        perm = torch.randperm(z.shape[0], generator=generator).to(z.device)
        return z[perm[:n]]

    def _run_point_set(self, result: SuiteResult, tag: str, z: Tensor) -> None:
        cfg = self.config
        if cfg.run_jacobian:
            with timed_step(f"jacobian ({tag})", cfg.verbose):
                setattr(result, f"jacobian_{tag}", compute_jacobian_spectrum(self.decoder, z))
        if cfg.run_lid:
            with timed_step(f"lid ({tag})", cfg.verbose):
                setattr(result, f"lid_{tag}", compute_lid(z))
        if cfg.run_label_dimensionality:
            with timed_step(f"label_dimensionality ({tag})", cfg.verbose):
                setattr(result, f"label_dimensionality_{tag}", compute_label_dimensionality(
                    self.classifier, z, min_class_count=cfg.label_dim_min_class_count
                ))
        if cfg.run_boundary_point:
            with timed_step(f"boundary_point ({tag})", cfg.verbose):
                setattr(result, f"boundary_point_{tag}", compute_boundary_geometry(
                    self.classifier, z, class_idx=cfg.class_idx
                ))

    def run(
        self,
        z_reference: Tensor,
        z_query: Optional[Tensor] = None,
        trajectory_endpoints: Optional[Tensor] = None,
    ) -> SuiteResult:
        """Run the configured diagnostics.

        Args:
            z_reference: ``(N, dz)`` reference latents.
            z_query: optional ``(N', dz)`` latents to compare against the reference.
            trajectory_endpoints: ``(M, 2, dz)`` start/end latents of ``M`` straight
                paths; required if any trajectory diagnostic is enabled.
        """
        cfg = self.config
        enabled = [f for f in _TRAJECTORY_FLAGS if getattr(cfg, f)]
        if trajectory_endpoints is None and enabled:
            raise ValueError(
                f"trajectory_endpoints is None but trajectory diagnostics are enabled "
                f"({', '.join(enabled)}). Provide trajectory_endpoints or disable them."
            )
        if trajectory_endpoints is not None and (
            trajectory_endpoints.dim() != 3 or trajectory_endpoints.shape[1] != 2
        ):
            raise ValueError(
                f"trajectory_endpoints must have shape (M, 2, dz); "
                f"got {tuple(trajectory_endpoints.shape)}."
            )

        gen = torch.Generator().manual_seed(cfg.seed)
        result = SuiteResult()
        self._run_point_set(result, "reference", self._subsample(z_reference, gen))
        if z_query is not None:
            self._run_point_set(result, "query", self._subsample(z_query, gen))

        if trajectory_endpoints is not None:
            z_start = trajectory_endpoints[:, 0, :]
            z_end = trajectory_endpoints[:, 1, :]

            if cfg.run_geodesic:
                with timed_step("geodesic", cfg.verbose):
                    result.geodesic = compute_geodesic_discrepancy(
                        self.decoder, z_start, z_end,
                        n_steps=cfg.geodesic_n_steps, solver=cfg.geodesic_solver,
                    )

            if cfg.run_ppl:
                with timed_step("ppl", cfg.verbose):
                    result.ppl = compute_ppl(
                        self.decoder, z_start, z_end,
                        n_steps=cfg.ppl_n_steps, device=resolve_device(cfg.device),
                    )

            if cfg.run_boundary_trajectory:
                with timed_step("boundary_trajectory", cfg.verbose):
                    trajs = linear_interpolate(z_start, z_end, cfg.boundary_trajectory_n_steps)
                    result.boundary_trajectory = compute_boundary_trajectory(
                        self.classifier, trajs, class_idx=cfg.class_idx
                    )

            if cfg.run_alignment:
                with timed_step("alignment", cfg.verbose):
                    result.alignment = compute_gradient_alignment(
                        self.classifier, z_start, z_end, class_idx=cfg.class_idx
                    )

            if cfg.run_secant_roughness:
                with timed_step("secant_roughness", cfg.verbose):
                    result.secant_roughness = compute_secant_roughness(
                        self.decoder, z_start, z_end,
                        n_steps=cfg.secant_roughness_n_steps,
                    )

            if cfg.run_label_coherence:
                with timed_step("label_coherence", cfg.verbose):
                    result.label_coherence = compute_label_coherence(
                        self.classifier, z_start, z_end,
                        n_steps=cfg.label_coherence_n_steps,
                    )

        return result
