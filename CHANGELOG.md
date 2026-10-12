# Changelog

## 0.1.0 — first public release

### Diagnostics
- Decoder Jacobian spectrum (`compute_jacobian_spectrum`) with a degeneracy flag for non-differentiable maps.
- Pullback metric and output-space arc length vs. chord (`compute_pullback_metric`, `compute_geodesic_discrepancy`) with Euler, RK4 and adaptive (`torchdiffeq`) solvers.
- Perceptual path length with local-contrast spike detection (`compute_ppl`, optional `lpips` extra).
- Local intrinsic dimensionality, TwoNN and Levina–Bickel MLE (`compute_lid`).
- Decision-boundary geometry and boundary crossings along paths (`compute_boundary_geometry`, `compute_boundary_trajectory`).
- Alignment of latent displacements with the decision-margin gradient (`compute_gradient_alignment`).
- Gradient-free secant roughness across step scales (`compute_secant_roughness`).
- Label coherence along paths and label-conditioned dimensionality (`compute_label_coherence`, `compute_label_dimensionality`).
- MINE mutual-information estimate (`estimate_mi`).
- `DiagnosticSuite` / `SuiteConfig` / `SuiteResult` to run everything on a reference and an optional query point set.

### Warps
- Uniform `Warp` interface (`forward`, `backproject`, `forward_ste`, `save`/`load`) and the `build_warp` factory.
- `vq`, `scramble`, `class_scramble` (within/across class), `flow_warp` (normalizing flows, optional `normflows` extra), `piecewise_affine`, `class_collapse`.
- Functional primitives `posthoc_vq`, `bijective_scramble`, `class_constrained_scramble`, `FlowWarp`, `fit_flow_ensemble`.

### Examples
- Thirteen runnable examples on a toy 2-D VAE with a three-class latent classifier, each saving a figure and checking its expected outcome; `examples/run_all.py` runs them all. The figures are included in `examples/figures/` and shown in `examples/README.md`.
