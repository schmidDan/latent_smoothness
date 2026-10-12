import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from latent_smoothness.jacobian import _compute_jacobian_raw

DZ = 4
DX = 64
N_CLS = 2
N = 128
M = 16
SEED = 0


def pytest_addoption(parser):
    default = "cuda" if torch.cuda.is_available() else "cpu"
    parser.addoption(
        "--device", default=default,
        help=f"Device to run tests on (default: {default}). "
             "Use --device=cpu to force CPU even when CUDA is available.",
    )


@pytest.fixture(scope="session")
def device(request):
    d = request.config.getoption("--device")
    if d == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available on this machine")
    return d


@pytest.fixture(scope="session", autouse=True)
def enforce_determinism():
    torch.use_deterministic_algorithms(True)


@pytest.fixture(scope="session")
def ensure_lpips_weights(tmp_path_factory, request):
    """Pre-warm LPIPS/VGG weights for the tests that need them.

    Under xdist the worker with the lowest ID (gw0) triggers the download if
    weights are not already on disk; all other workers wait on a sentinel file
    written by gw0.  No external locking library needed — the sentinel lives in
    the shared tmp-root directory that all xdist workers can see.
    When not running under xdist (worker_id == "master") this runs synchronously
    as a simple warm-up.  If weights are already cached the fixture returns
    immediately and costs nothing.
    """
    _lpips_pkg = pytest.importorskip("lpips")

    weights_path = Path(_lpips_pkg.__file__).parent / "weights" / "v0.1" / "vgg.pth"
    if weights_path.exists():
        return  # Already on disk — all workers proceed immediately (common case)

    # Weights not yet downloaded.  Coordinate via a sentinel in the shared
    # tmp-root (parent of every xdist worker's per-worker basetemp directory).
    root_tmp = tmp_path_factory.getbasetemp().parent
    sentinel = root_tmp / "lpips_vgg_ready"

    # worker_id is "gw0", "gw1", … under xdist, or "master" without it.
    worker_id = getattr(request.config, "workerinput", {}).get("workerid", "master")

    if worker_id in ("master", "gw0"):
        # Lowest-ID process: trigger download, then signal all others.
        _lpips_pkg.LPIPS(net="vgg")
        sentinel.touch()
    else:
        # All other workers: poll until gw0 signals completion (max 60 s).
        for _ in range(120):
            if sentinel.exists():
                return
            time.sleep(0.5)
        raise RuntimeError(
            "Timed out waiting for LPIPS weights — gw0 did not write sentinel within 60 s."
        )


@pytest.fixture
def linear_decoder(device):
    torch.manual_seed(SEED)
    return nn.Linear(DZ, DX, bias=False).to(device)


@pytest.fixture
def linear_spatial_decoder(device):
    torch.manual_seed(SEED)

    class LinearSpatial(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(DZ, DX, bias=False)
            self.unflatten = nn.Unflatten(-1, (1, 8, 8))

        def forward(self, z):
            return self.unflatten(self.linear(z))

    return LinearSpatial().to(device)


@pytest.fixture
def rough_spatial_decoder(device):
    torch.manual_seed(SEED)

    class RoughSpatial(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(DZ, DX, bias=False)
            self.unflatten = nn.Unflatten(-1, (1, 8, 8))

        def forward(self, z):
            scale = 1.0 + 10.0 * torch.exp(-100 * (z.norm(dim=-1, keepdim=True) - 1.0) ** 2)
            h = z * scale
            return self.unflatten(self.linear(h))

    return RoughSpatial().to(device)


@pytest.fixture
def smooth_nonlinear_decoder(device):
    torch.manual_seed(SEED)
    return nn.Sequential(
        nn.Linear(DZ, 32),
        nn.Tanh(),
        nn.BatchNorm1d(32),
        nn.Linear(32, DX),
        nn.Tanh(),
    ).to(device)


@pytest.fixture
def rough_nonlinear_decoder(device):
    torch.manual_seed(SEED)

    class RoughNonlinear(nn.Module):
        def forward(self, z):
            scale = 1.0 + 10.0 * torch.exp(-100 * (z.norm(dim=-1, keepdim=True) - 1.0) ** 2)
            return z * scale

    return RoughNonlinear().to(device)


@pytest.fixture
def linear_classifier(device):
    torch.manual_seed(SEED)
    return nn.Linear(DZ, N_CLS, bias=False).to(device)


@pytest.fixture
def unit_gaussian_latents(device):
    torch.manual_seed(SEED)
    return torch.randn(N, DZ, device=device)


@pytest.fixture
def aligned_pairs(linear_classifier, unit_gaussian_latents):
    torch.manual_seed(SEED)
    W = linear_classifier.weight  # (N_CLS, DZ)
    w0 = W[0, :] - W[1, :]  # gradient of the class-0 decision margin
    delta = w0 / w0.norm()
    mu_before = unit_gaussian_latents
    mu_after = mu_before + delta.unsqueeze(0).expand(N, -1)
    return mu_before, mu_after


@pytest.fixture
def perpendicular_pairs(linear_classifier, unit_gaussian_latents):
    torch.manual_seed(SEED)
    W = linear_classifier.weight
    w0 = W[0, :] - W[1, :]  # gradient of the class-0 decision margin
    e0 = torch.zeros(DZ, device=w0.device)
    e0[0] = 1.0
    proj = (e0 @ w0) / (w0 @ w0) * w0
    perp = e0 - proj
    if perp.norm() < 1e-6:
        e1 = torch.zeros(DZ, device=w0.device)
        e1[1] = 1.0
        proj = (e1 @ w0) / (w0 @ w0) * w0
        perp = e1 - proj
    perp = perp / perp.norm()
    mu_before = unit_gaussian_latents
    mu_after = mu_before + perp.unsqueeze(0).expand(N, -1)
    return mu_before, mu_after


@pytest.fixture
def anti_aligned_pairs(linear_classifier, unit_gaussian_latents):
    torch.manual_seed(SEED)
    W = linear_classifier.weight
    w0 = W[0, :] - W[1, :]  # gradient of the class-0 decision margin
    delta = -w0 / w0.norm()
    mu_before = unit_gaussian_latents
    mu_after = mu_before + delta.unsqueeze(0).expand(N, -1)
    return mu_before, mu_after


def compute_jacobian_no_eval_mode(decoder, z_points):
    return _compute_jacobian_raw(decoder, z_points)
