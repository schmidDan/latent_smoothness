from latent_smoothness.warp.quantize import posthoc_vq
from latent_smoothness.warp.scramble import bijective_scramble, class_constrained_scramble
from latent_smoothness.warp.flow_warp import FlowWarp, WarpDiagnosticResult, fit_flow_ensemble
from latent_smoothness.warp.base import Warp
from latent_smoothness.warp.piecewise_affine import PiecewiseAffineWarp
from latent_smoothness.warp.class_collapse import ClassCollapseWarp
from latent_smoothness.warp.adapters import (
    VQWarp,
    ScrambleWarp,
    ClassScrambleWarp,
    FlowWarpAdapter,
    WARP_NAMES,
    build_warp,
)

__all__ = [
    # functional primitives
    "posthoc_vq",
    "bijective_scramble",
    "class_constrained_scramble",
    "FlowWarp",
    "WarpDiagnosticResult",
    "fit_flow_ensemble",
    # uniform Warp interface
    "Warp",
    "VQWarp",
    "ScrambleWarp",
    "ClassScrambleWarp",
    "ClassCollapseWarp",
    "FlowWarpAdapter",
    "PiecewiseAffineWarp",
    "WARP_NAMES",
    "build_warp",
]
