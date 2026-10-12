import pytest
import torch

from latent_smoothness import DiagnosticSuite, SuiteConfig
from tests.conftest import SEED

# Small step counts keep the suite tests fast; the per-diagnostic test files
# cover accuracy at realistic settings.
FAST = dict(ppl_n_steps=12, geodesic_n_steps=5, geodesic_solver="euler")


def _run(decoder, classifier, z, pairs, device, **cfg):
    torch.manual_seed(SEED)
    mu_before, mu_after = pairs
    endpoints = torch.stack([mu_before, mu_after], dim=1)
    config = SuiteConfig(device=device, **{**FAST, **cfg})
    suite = DiagnosticSuite(decoder, classifier, config)
    return suite.run(z_reference=z, z_query=z, trajectory_endpoints=endpoints)


@pytest.mark.slow
@pytest.mark.usefixtures("ensure_lpips_weights")
def test_full_suite_completes(
    linear_spatial_decoder, linear_classifier, unit_gaussian_latents, aligned_pairs, device
):
    result = _run(linear_spatial_decoder, linear_classifier, unit_gaussian_latents,
                  aligned_pairs, device)

    for name in result.__dataclass_fields__:
        assert getattr(result, name) is not None, f"{name} was not computed"

    keys = result.to_dict()
    assert "label_coherence.mean_monotonic_gain" in keys
    assert "label_coherence.mean_label_flips" in keys
    assert "label_dimensionality_reference.within_class_effrank" in keys
    assert "label_dimensionality_query.class_diff_effrank" in keys
    assert "boundary_trajectory.n_crossings" in keys


@pytest.mark.slow
@pytest.mark.usefixtures("ensure_lpips_weights")
def test_to_scalars_returns_flat_dict(
    linear_spatial_decoder, linear_classifier, unit_gaussian_latents, aligned_pairs, device
):
    result = _run(linear_spatial_decoder, linear_classifier, unit_gaussian_latents,
                  aligned_pairs, device)
    scalars = result.to_scalars()
    assert isinstance(scalars, dict)
    assert len(scalars) > 0
    for k, v in scalars.items():
        assert isinstance(v, (int, float)), f"scalar {k}={v} is not int/float"
    assert "jacobian_query.lipschitz_estimate" in scalars


@pytest.mark.slow
@pytest.mark.usefixtures("ensure_lpips_weights")
def test_summary_table_is_nonempty(
    linear_spatial_decoder, linear_classifier, unit_gaussian_latents, aligned_pairs, device
):
    result = _run(linear_spatial_decoder, linear_classifier, unit_gaussian_latents,
                  aligned_pairs, device)
    table = result.summary_table()
    assert isinstance(table, str)
    assert "label_dimensionality_query" in table
    assert "not computed" not in table


def test_query_is_optional(linear_spatial_decoder, linear_classifier, unit_gaussian_latents, device):
    config = SuiteConfig(device=device, run_ppl=False, run_geodesic=False,
                         run_boundary_trajectory=False, run_alignment=False,
                         run_secant_roughness=False, run_label_coherence=False)
    suite = DiagnosticSuite(linear_spatial_decoder, linear_classifier, config)
    result = suite.run(unit_gaussian_latents)
    assert result.jacobian_reference is not None
    assert result.jacobian_query is None


def test_subsampling_is_seeded(linear_spatial_decoder, linear_classifier, unit_gaussian_latents, device):
    config = SuiteConfig(device=device, n_samples=32, run_ppl=False, run_geodesic=False,
                         run_boundary_trajectory=False, run_alignment=False,
                         run_secant_roughness=False, run_label_coherence=False)
    suite = DiagnosticSuite(linear_spatial_decoder, linear_classifier, config)
    r1 = suite.run(unit_gaussian_latents)
    r2 = suite.run(unit_gaussian_latents)
    assert r1.lid_reference.lid_twonn.shape == (32,)
    assert torch.equal(r1.lid_reference.lid_twonn, r2.lid_reference.lid_twonn)


@pytest.mark.parametrize("flag", [
    "run_ppl", "run_geodesic", "run_boundary_trajectory",
    "run_alignment", "run_secant_roughness", "run_label_coherence",
])
def test_value_error_when_endpoints_missing(
    linear_spatial_decoder, linear_classifier, unit_gaussian_latents, device, flag
):
    flags = dict(run_ppl=False, run_geodesic=False, run_boundary_trajectory=False,
                 run_alignment=False, run_secant_roughness=False, run_label_coherence=False)
    flags[flag] = True
    suite = DiagnosticSuite(linear_spatial_decoder, linear_classifier,
                            SuiteConfig(device=device, **flags))
    with pytest.raises(ValueError, match=flag):
        suite.run(unit_gaussian_latents, trajectory_endpoints=None)


def test_value_error_on_bad_endpoint_shape(
    linear_spatial_decoder, linear_classifier, unit_gaussian_latents, device
):
    suite = DiagnosticSuite(linear_spatial_decoder, linear_classifier, SuiteConfig(device=device))
    with pytest.raises(ValueError, match=r"\(M, 2, dz\)"):
        suite.run(unit_gaussian_latents, trajectory_endpoints=unit_gaussian_latents)


@pytest.mark.parametrize("verbose", [False, True])
def test_progress_output_is_opt_in(
    linear_spatial_decoder, linear_classifier, unit_gaussian_latents, device, verbose, capsys
):
    config = SuiteConfig(device=device, verbose=verbose, run_jacobian=False, run_ppl=False,
                         run_geodesic=False, run_boundary_trajectory=False, run_alignment=False,
                         run_secant_roughness=False, run_label_coherence=False)
    DiagnosticSuite(linear_spatial_decoder, linear_classifier, config).run(unit_gaussian_latents)
    assert ("[suite]" in capsys.readouterr().err) is verbose
