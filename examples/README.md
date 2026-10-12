# `latent_smoothness` Examples

Install with the example extras (`pip install -e ".[all,examples]"`) and run any example from the repository root:

```bash
python examples/ex01_jacobian_spectrum.py
```

All examples save figures to `examples/outputs/` (git-ignored) and print a one-line pass/fail assertion.

## Index

| File | Toy problem | Figure saved | Pass assertion |
|------|-------------|-------------|----------------|
| `ex01_jacobian_spectrum.py` | Smooth vs. rough inline decoder (dz=4) — side-by-side SVD spectra | `ex01_jacobian_spectrum.png` | `Jacobian spike detected: rough Lipschitz > 2× smooth PASS` |
| `ex02_geodesic_discrepancy.py` | `toy_models.py` decoder (dz=2) — pullback metric along 16 paths | `ex02_geodesic_discrepancy.png` | `Mean geodesic discrepancy: X PASS` |
| `ex03_ppl_profile.py` | Rough spatial decoder with narrow spike at ‖z‖=1 — per-segment LPIPS | `ex03_ppl_profile.png` | `PPL spike detected at expected location PASS` |
| `ex04_lid_along_trajectory.py` | 1D submanifold (dz=2) vs. full 2D Gaussian — TwoNN LID | `ex04_lid_along_trajectory.png` | `LID on-manifold ~= 1.0: X PASS` |
| `ex05_boundary_geometry.py` | `toy_models.py` linear classifier — decision margins, gradient norms, trajectory crossings | `ex05_boundary_geometry.png` | `Margin sign matches argmax at all points; boundary crossings: N PASS` |
| `ex06_gradient_alignment.py` | `toy_models.py` VAE + classifier — quiver plot of Δμ, cosine to the margin gradient ∇m | `ex06_gradient_alignment.png` | `Mean alignment: X PASS` |
| `ex07_vq_scramble_comparison.py` | `toy_models.py` latents → VQ → bijective scramble → backprojection — classifier accuracy and exact round trip | `ex07_vq_scramble_comparison.png` | `Classifier accuracy -- ... PASS` |
| `ex08_flow_warp.py` | `toy_models.py` latents warped to mixture_of_gaussians target | `ex08_flow_warp.png` | `FlowWarp KL-to-target=X  KL-to-orig=Y ... PASS` |
| `ex09_mine_verification.py` | MINE MI: shuffled baseline vs. VQ / bijective scramble / round trip at K=4 and K=64, two sample sizes | `ex09_mine_N1024.png`, `ex09_mine_N16384.png` | four `PASS`/`FAIL` lines (ordering, K=4 tightness, round trip) |
| `ex10_full_suite.py` | Full `DiagnosticSuite` on `toy_models.py` VAE — all diagnostics | `ex10_full_suite.png` | `Suite completed successfully PASS` |
