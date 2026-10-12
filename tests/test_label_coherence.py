import torch
import torch.nn as nn

from latent_smoothness import compute_label_coherence


class Coord(nn.Module):
    """2-class linear classifier: logits = first two coords of z."""
    def forward(self, z):
        return z[:, :2]


class Wiggly(nn.Module):
    """Oscillating decision boundary -> many argmax flips along a straight path."""
    def forward(self, z):
        a = torch.cos(6.0 * z[:, 0:1])
        return torch.cat([a, -a], dim=1)


def _pair(z0, z1, M=8):
    zs = torch.tensor(z0, dtype=torch.float32).repeat(M, 1)
    ze = torch.tensor(z1, dtype=torch.float32).repeat(M, 1)
    return zs, ze


def test_monotone_path_is_coherent():
    # straight path from class-0 region to class-1 region: one crossing, P(target) climbs.
    zs, ze = _pair([1.0, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0])
    r = compute_label_coherence(Coord(), zs, ze, n_steps=64)
    assert r.mean_monotonic_gain == 1.0          # P(target=class1) increases every step
    assert r.mean_label_flips == 1.0             # exactly one argmax crossing
    assert r.mean_target_prob_gain > 0.5


def test_scrambled_path_thrashes():
    zs, ze = _pair([-2.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0])
    coherent = compute_label_coherence(Coord(), zs, ze, n_steps=64)
    wiggly = compute_label_coherence(Wiggly(), zs, ze, n_steps=64)
    assert wiggly.mean_label_flips > coherent.mean_label_flips        # many crossings
    assert wiggly.mean_monotonic_gain < coherent.mean_monotonic_gain  # not a clean climb


def test_shape_and_determinism():
    zs, ze = _pair([1.0, -1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 0.0], M=5)
    r1 = compute_label_coherence(Coord(), zs, ze)
    r2 = compute_label_coherence(Coord(), zs, ze)
    assert r1.label_flips.shape == (5,)
    assert r1.label_flips.dtype == torch.int64
    assert torch.equal(r1.label_flips, r2.label_flips)
    assert torch.allclose(r1.monotonic_gain, r2.monotonic_gain)
