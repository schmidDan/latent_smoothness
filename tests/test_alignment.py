import torch
from latent_smoothness import compute_gradient_alignment


def test_alignment_is_one_for_aligned_pairs(linear_classifier, aligned_pairs):
    mu_before, mu_after = aligned_pairs
    result = compute_gradient_alignment(linear_classifier, mu_before, mu_after)
    assert torch.allclose(result.cosine_similarities, torch.ones_like(result.cosine_similarities), atol=1e-5), \
        f"Cosine similarities should be 1.0 for aligned pairs, got min={result.cosine_similarities.min()}"


def test_alignment_is_zero_for_perpendicular_pairs(linear_classifier, perpendicular_pairs):
    mu_before, mu_after = perpendicular_pairs
    result = compute_gradient_alignment(linear_classifier, mu_before, mu_after)
    assert torch.allclose(result.cosine_similarities, torch.zeros_like(result.cosine_similarities), atol=1e-5), \
        f"Cosine similarities should be 0.0 for perpendicular pairs, got max={result.cosine_similarities.abs().max()}"


def test_alignment_is_neg_one_for_anti_aligned_pairs(linear_classifier, anti_aligned_pairs):
    mu_before, mu_after = anti_aligned_pairs
    result = compute_gradient_alignment(linear_classifier, mu_before, mu_after)
    expected = -torch.ones_like(result.cosine_similarities)
    assert torch.allclose(result.cosine_similarities, expected, atol=1e-5), \
        f"Cosine similarities should be -1.0 for anti-aligned pairs, got max={result.cosine_similarities.max()}"
