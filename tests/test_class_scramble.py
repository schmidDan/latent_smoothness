"""Class-constrained scramble (within/across) warp."""
import torch

from latent_smoothness.warp import build_warp, ClassScrambleWarp
from latent_smoothness.warp.scramble import _fit_class_scramble

K = 24


def _labelled_blobs(seed=0):
    g = torch.Generator().manual_seed(seed)
    centres = torch.tensor([[4.0, 0, 0, 0], [-4.0, 0, 0, 0], [0, 4.0, 0, 0]])
    z = torch.cat([c + 0.3 * torch.randn(100, 4, generator=g) for c in centres], dim=0)
    labels = torch.cat([torch.full((100,), k) for k in range(3)])
    return z, labels


def _moved(perm, cell_cls):
    same = diff = 0
    for i in range(len(perm)):
        j = int(perm[i])
        if j == i:
            continue
        if cell_cls[i] < 0 or cell_cls[j] < 0:
            continue
        same += int(cell_cls[i] == cell_cls[j])
        diff += int(cell_cls[i] != cell_cls[j])
    return same, diff


def test_within_moves_only_same_class():
    z, labels = _labelled_blobs()
    _, perm, cell_cls = _fit_class_scramble(z, K, labels, "within", seed=0)
    assert sorted(perm.tolist()) == list(range(K))       # valid bijection
    assert torch.equal(perm[perm], torch.arange(K))       # involution
    same, diff = _moved(perm, cell_cls)
    assert diff == 0 and same > 0                          # every swap is within-class


def test_across_moves_only_different_class():
    z, labels = _labelled_blobs()
    _, perm, cell_cls = _fit_class_scramble(z, K, labels, "across", seed=0)
    assert sorted(perm.tolist()) == list(range(K))
    assert torch.equal(perm[perm], torch.arange(K))
    same, diff = _moved(perm, cell_cls)
    assert same == 0 and diff > 0                          # every swap crosses a class


def test_backproject_undoes_forward():
    z, labels = _labelled_blobs()
    w = build_warp("class_scramble", fit_data=z, codebook_size=K,
                   labels=labels, class_mode="within", seed=0)
    # involution: backproject(forward(z)) == the plain quantisation cb[cell(z)]
    rt = w.backproject(w.forward(z))
    assert torch.equal(w.forward(rt), w.forward(z))        # rt lands in the same cells


def test_within_and_across_differ():
    z, labels = _labelled_blobs()
    wi = build_warp("class_scramble", fit_data=z, codebook_size=K,
                    labels=labels, class_mode="within", seed=0)
    wa = build_warp("class_scramble", fit_data=z, codebook_size=K,
                    labels=labels, class_mode="across", seed=0)
    assert not torch.allclose(wi.forward(z), wa.forward(z))


def test_save_load_and_determinism(tmp_path):
    z, labels = _labelled_blobs()
    w1 = build_warp("class_scramble", fit_data=z, codebook_size=K,
                    labels=labels, class_mode="across", seed=1)
    w2 = build_warp("class_scramble", fit_data=z, codebook_size=K,
                    labels=labels, class_mode="across", seed=1)
    assert torch.allclose(w1.forward(z), w2.forward(z))    # deterministic on seed
    p = tmp_path / "cs.pt"
    w1.save(p)
    assert torch.allclose(ClassScrambleWarp.load(p).forward(z), w1.forward(z))
