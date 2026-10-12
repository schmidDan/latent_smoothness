import torch
from latent_smoothness import compute_jacobian_spectrum
from tests.conftest import compute_jacobian_no_eval_mode


def test_jacobian_svd_matches_weight_matrix(linear_decoder, unit_gaussian_latents):
    W = linear_decoder.weight  # (DX, DZ)
    expected_svs = torch.linalg.svdvals(W)  # (min(DX, DZ),)
    result = compute_jacobian_spectrum(linear_decoder, unit_gaussian_latents)
    # For linear decoder Jacobian is constant = W at all points
    for i in range(result.singular_values.shape[0]):
        assert torch.allclose(result.singular_values[i], expected_svs, atol=1e-5), \
            f"SVD mismatch at point {i}"


def test_eval_mode_restores_train_state(smooth_nonlinear_decoder, unit_gaussian_latents):
    smooth_nonlinear_decoder.train()
    compute_jacobian_spectrum(smooth_nonlinear_decoder, unit_gaussian_latents)
    assert smooth_nonlinear_decoder.training, \
        "eval_mode failed to restore train state after compute_jacobian_spectrum"


def test_jacobian_differs_in_train_mode(smooth_nonlinear_decoder, unit_gaussian_latents):
    J_eval = compute_jacobian_spectrum(smooth_nonlinear_decoder, unit_gaussian_latents)
    smooth_nonlinear_decoder.train()
    J_train = compute_jacobian_no_eval_mode(smooth_nonlinear_decoder, unit_gaussian_latents)
    assert not torch.allclose(J_eval.singular_values, J_train.singular_values), \
        "eval_mode wrapper is not doing real work — train and eval Jacobians are identical"
