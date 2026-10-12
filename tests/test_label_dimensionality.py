import torch
import torch.nn as nn

from latent_smoothness import compute_label_dimensionality


class NearestAnchor(nn.Module):
    """logits = -distance to each anchor -> argmax = nearest anchor index (= class).

    Lets a test place class means at chosen anchor positions and control which
    latent gets which label, independently of the point coordinates.
    """
    def __init__(self, anchors):
        super().__init__()
        self.register_buffer("anchors", anchors)

    def forward(self, z):
        return -torch.cdist(z, self.anchors)


def _clusters(anchors, spread, n_per=100, seed=0):
    """Isotropic gaussian clusters around each anchor."""
    g = torch.Generator().manual_seed(seed)
    dz = anchors.shape[1]
    return torch.cat(
        [a + spread * torch.randn(n_per, dz, generator=g) for a in anchors], dim=0
    )


def _needle_clusters(anchors, spread, axis, n_per=100, seed=0):
    """Clusters with variance only along one within-class axis (rank-1 shape)."""
    g = torch.Generator().manual_seed(seed)
    dz = anchors.shape[1]
    out = []
    for a in anchors:
        noise = torch.zeros(n_per, dz)
        noise[:, axis] = spread * torch.randn(n_per, generator=g)
        out.append(a + noise)
    return torch.cat(out, dim=0)


def test_within_class_spread_tracks_scale():
    # spread (magnitude) grows with cluster scale; effrank (shape) is scale-invariant.
    anchors = torch.tensor([[3.0, 0, 0, 0], [-3.0, 0, 0, 0]])
    clf = NearestAnchor(anchors)
    tight = compute_label_dimensionality(clf, _clusters(anchors, 0.1), min_class_count=5)
    wide = compute_label_dimensionality(clf, _clusters(anchors, 0.8), min_class_count=5)
    assert tight.n_classes_used == 2 and wide.n_classes_used == 2
    assert wide.within_class_spread > tight.within_class_spread


def test_within_class_effrank_tracks_shape():
    # isotropic (fills dz dims) vs needle (one within-class dim) -> effrank drops.
    anchors = torch.tensor([[3.0, 0, 0, 0], [-3.0, 0, 0, 0]])
    clf = NearestAnchor(anchors)
    iso = compute_label_dimensionality(clf, _clusters(anchors, 0.3), min_class_count=5)
    needle = compute_label_dimensionality(
        clf, _needle_clusters(anchors, 0.4, axis=1), min_class_count=5
    )
    assert iso.within_class_effrank > needle.within_class_effrank
    assert needle.within_class_effrank < 1.5     # ~rank-1 shape


def test_class_mean_arrangement_rank():
    line = torch.tensor([[-4.0, 0, 0, 0], [0.0, 0, 0, 0], [4.0, 0, 0, 0]])
    r_line = compute_label_dimensionality(NearestAnchor(line), _clusters(line, 0.2))
    spread = torch.tensor([[4.0, 0, 0, 0], [0, 4.0, 0, 0], [0, 0, 4.0, 0]])
    r_spread = compute_label_dimensionality(NearestAnchor(spread), _clusters(spread, 0.2))
    assert r_line.n_classes_used == 3 and r_spread.n_classes_used == 3
    assert r_line.class_diff_consistency > 0.9                       # one dominant axis
    assert r_spread.class_diff_effrank > r_line.class_diff_effrank   # spread over more dims


def test_single_class_guard():
    anchors = torch.tensor([[0.0, 0, 0, 0]])
    r = compute_label_dimensionality(NearestAnchor(anchors), _clusters(anchors, 0.2))
    assert r.n_classes_used == 1
    assert r.class_diff_effrank != r.class_diff_effrank             # nan (single class)
    assert r.within_class_effrank == r.within_class_effrank         # still defined
