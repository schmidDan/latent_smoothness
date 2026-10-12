import torch
from latent_smoothness import compute_boundary_geometry, compute_boundary_trajectory
from latent_smoothness.utils import decision_margin, linear_interpolate
from tests.conftest import DZ, M, SEED


def test_margins_match_linear_formula(linear_classifier, unit_gaussian_latents):
    W = linear_classifier.weight  # (N_CLS, DZ)
    result = compute_boundary_geometry(linear_classifier, unit_gaussian_latents)
    expected = unit_gaussian_latents @ (W[0, :] - W[1, :])
    assert torch.allclose(result.margins, expected, atol=1e-5), \
        f"Margins should equal mu @ (W[0,:] - W[1,:]), max diff={(result.margins - expected).abs().max()}"


def test_gradient_norms_match_linear(linear_classifier, unit_gaussian_latents):
    W = linear_classifier.weight
    expected_norm = (W[0, :] - W[1, :]).norm()
    result = compute_boundary_geometry(linear_classifier, unit_gaussian_latents)
    expected = expected_norm.expand(unit_gaussian_latents.shape[0])
    assert torch.allclose(result.gradient_norms, expected, atol=1e-5), \
        f"Gradient norms should equal ||W[0,:] - W[1,:]||"


def test_curvature_zero_for_linear(linear_classifier, unit_gaussian_latents):
    result = compute_boundary_geometry(linear_classifier, unit_gaussian_latents)
    assert torch.allclose(result.curvature, torch.zeros_like(result.curvature), atol=1e-6), \
        f"Curvature should be zero for linear classifier, max={result.curvature.abs().max()}"


def test_boundary_trajectory_one_crossing_per_path(linear_classifier, device):
    torch.manual_seed(SEED)
    W = linear_classifier.weight  # (N_CLS, DZ)
    w0 = W[0, :] - W[1, :]  # class-0 decision margin is z @ w0

    # Construct z_start with positive margin, z_end with negative margin
    z_start = torch.randn(M, DZ, device=device)
    # Ensure positive margin
    logit_start = z_start @ w0
    z_start = z_start - logit_start.unsqueeze(1) * w0.unsqueeze(0) / (w0 @ w0) + \
              (logit_start.abs() + 0.5).unsqueeze(1) * w0.unsqueeze(0) / (w0 @ w0)

    z_end = torch.randn(M, DZ, device=device)
    logit_end = z_end @ w0
    z_end = z_end - logit_end.unsqueeze(1) * w0.unsqueeze(0) / (w0 @ w0) - \
            (logit_end.abs() + 0.5).unsqueeze(1) * w0.unsqueeze(0) / (w0 @ w0)

    # Verify analytically
    assert (z_start @ w0 > 0).all(), "z_start should have positive margins"
    assert (z_end @ w0 < 0).all(), "z_end should have negative margins"

    trajs = linear_interpolate(z_start, z_end, n_steps=50)
    result = compute_boundary_trajectory(linear_classifier, trajs)

    assert result.n_crossings == M, \
        f"Expected {M} crossings (one per trajectory), got {result.n_crossings}"
    assert (result.sign_change_locations.sum(dim=-1) == 1).all(), \
        "Each trajectory should have exactly one sign change"


def test_decision_margin_definition():
    logits = torch.tensor([[2.0, 0.5, 1.0], [0.0, 3.0, 1.0]])
    assert torch.allclose(decision_margin(logits, 0), torch.tensor([1.0, -3.0]))
    assert torch.allclose(decision_margin(logits, 2), torch.tensor([-1.0, -2.0]))
    # single-output (binary) classifier: the logit itself
    assert torch.allclose(decision_margin(torch.tensor([[0.7], [-0.2]]), 0), torch.tensor([0.7, -0.2]))


def test_crossings_follow_argmax_not_single_logit():
    """Both logits positive along the path, but the argmax flips once: one crossing."""
    clf = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        clf.weight.copy_(torch.eye(2))           # logits = z
    z_start = torch.tensor([[3.0, 1.0]])
    z_end = torch.tensor([[1.0, 3.0]])
    result = compute_boundary_trajectory(clf, linear_interpolate(z_start, z_end, n_steps=21))
    assert result.n_crossings == 1
