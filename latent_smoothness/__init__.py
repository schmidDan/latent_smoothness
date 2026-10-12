"""Diagnostics for the geometry and smoothness of learned latent spaces."""
__version__ = "0.1.0"

from latent_smoothness.suite import DiagnosticSuite, SuiteConfig, SuiteResult
from latent_smoothness.jacobian import compute_jacobian_spectrum, JacobianResult
from latent_smoothness.geodesic import (
    compute_pullback_metric,
    PullbackResult,
    compute_geodesic_discrepancy,
    GeodesicResult,
)
from latent_smoothness.ppl import compute_ppl, PPLResult
from latent_smoothness.lid import compute_lid, LIDResult
from latent_smoothness.boundary import (
    compute_boundary_geometry,
    BoundaryPointResult,
    compute_boundary_trajectory,
    BoundaryTrajectoryResult,
)
from latent_smoothness.alignment import compute_gradient_alignment, AlignmentResult
from latent_smoothness.roughness import compute_secant_roughness, SecantRoughnessResult
from latent_smoothness.label_coherence import compute_label_coherence, LabelCoherenceResult
from latent_smoothness.label_dimensionality import (
    compute_label_dimensionality, LabelDimensionalityResult,
)
from latent_smoothness.mine import estimate_mi
from latent_smoothness.utils import (
    eval_mode, linear_interpolate, get_lpips, decision_margin, resolve_device, set_global_seed,
)

__all__ = [
    "DiagnosticSuite", "SuiteConfig", "SuiteResult",
    "compute_jacobian_spectrum", "JacobianResult",
    "compute_pullback_metric", "PullbackResult",
    "compute_geodesic_discrepancy", "GeodesicResult",
    "compute_ppl", "PPLResult",
    "compute_lid", "LIDResult",
    "compute_boundary_geometry", "BoundaryPointResult",
    "compute_boundary_trajectory", "BoundaryTrajectoryResult",
    "compute_gradient_alignment", "AlignmentResult",
    "compute_secant_roughness", "SecantRoughnessResult",
    "compute_label_coherence", "LabelCoherenceResult",
    "compute_label_dimensionality", "LabelDimensionalityResult",
    "estimate_mi",
    "eval_mode", "linear_interpolate", "get_lpips",
    "decision_margin", "resolve_device", "set_global_seed",
    "__version__",
]
