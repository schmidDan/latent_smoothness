"""Class-collapse (smooth, non-injective, class-merging) warp."""
import torch

from latent_smoothness.warp import build_warp, ClassCollapseWarp


def _labelled_blobs(seed=0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.tensor([[6.0, 0, 0, 0], [-6.0, 0, 0, 0], [0, 3.0, 0, 0]])
    z = torch.cat([c + 0.2 * torch.randn(100, 4, generator=g) for c in centres], dim=0)
    labels = torch.cat([torch.full((100,), k) for k in range(3)])
    return z, labels


def test_collapses_top_class_axis():
    z, labels = _labelled_blobs()
    w = build_warp("class_collapse", fit_data=z, labels=labels, n_directions=1)
    means = torch.stack([z[labels == k].mean(0) for k in range(3)])
    projected = w.forward(means)
    # the class-mean cloud loses its dominant axis -> variance collapses
    assert float(projected.var(0).sum()) < 0.5 * float(means.var(0).sum())
    assert w.is_differentiable and not w.is_invertible
    assert w.n_directions == 1


def test_forward_is_linear_and_differentiable():
    z, labels = _labelled_blobs()
    w = build_warp("class_collapse", fit_data=z, labels=labels, n_directions=1)
    # linearity: W(a*z1 + b*z2) - W(0) == a*(W(z1)-W(0)) + b*(W(z2)-W(0))
    z0 = torch.zeros(1, 4)
    a, b = 2.0, -3.0
    z1, z2 = torch.randn(1, 4), torch.randn(1, 4)
    lhs = w.forward(a * z1 + b * z2) - w.forward(z0)
    rhs = a * (w.forward(z1) - w.forward(z0)) + b * (w.forward(z2) - w.forward(z0))
    assert torch.allclose(lhs, rhs, atol=1e-5)
    zr = z.clone().requires_grad_(True)
    w.forward(zr).sum().backward()
    assert zr.grad is not None and zr.grad.abs().sum() > 0


def test_dtype_device_preserving_and_save_load(tmp_path):
    z, labels = _labelled_blobs()
    w = build_warp("class_collapse", fit_data=z, labels=labels, n_directions=2)
    assert w.forward(z.double()).dtype == torch.float64
    assert w.forward(z).device == z.device
    p = tmp_path / "cc.pt"
    w.save(p)
    assert torch.allclose(ClassCollapseWarp.load(p).forward(z), w.forward(z))


def test_n_directions_capped_by_class_count():
    # only 2 classes -> at most 1 between-class direction can be removed.
    g = torch.Generator().manual_seed(0)
    z = torch.cat([torch.tensor([3.0, 0, 0, 0]) + 0.2 * torch.randn(50, 4, generator=g),
                   torch.tensor([-3.0, 0, 0, 0]) + 0.2 * torch.randn(50, 4, generator=g)], dim=0)
    labels = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    w = build_warp("class_collapse", fit_data=z, labels=labels, n_directions=3)
    assert w.n_directions == 1
