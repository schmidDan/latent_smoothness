import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

_lpips_cache: dict = {}


def set_global_seed(seed: int) -> None:
    """Seed every RNG the library draws from, for reproducible fits.

    Seeds python ``random``, numpy, torch (CPU and, if available, all CUDA
    devices). This is the canonical determinism entry point — route stochastic
    warp construction through it rather than calling bare
    ``torch.manual_seed``/``np.random.seed``.

    ``torch.use_deterministic_algorithms(True)`` is gated on the environment
    variable ``LS_DETERMINISTIC=1`` so GPU runs that legitimately cannot use a
    deterministic kernel are not forced into a hard error; CPU runs and CI set
    the flag to lock determinism down fully.

    Args:
        seed: non-negative integer seed applied to all generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if os.environ.get("LS_DETERMINISTIC") == "1":
        torch.use_deterministic_algorithms(True)


def _get_logger(logger=None) -> logging.Logger:
    return logger if logger is not None else logging.getLogger(__name__)


def resolve_device(device: Optional[str] = None) -> str:
    """Return ``device`` unchanged, or ``"cuda"``/``"cpu"`` (auto) when ``None``."""
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def decision_margin(logits: Tensor, class_idx: int = 0) -> Tensor:
    """Decision margin of ``class_idx``: ``l_c - max_{j != c} l_j``.

    Positive iff ``class_idx`` is the ``argmax`` class, so its zero level set is
    the classifier's decision boundary for that class. A single-output
    classifier (``logits.shape[-1] == 1``) is treated as binary and its logit is
    returned as-is (``class_idx`` is then ignored).

    Args:
        logits: ``(..., C)`` classifier outputs.
        class_idx: the class whose margin is computed.

    Returns:
        ``(...)`` margins.
    """
    if logits.shape[-1] == 1:
        return logits[..., 0]
    others = torch.cat([logits[..., :class_idx], logits[..., class_idx + 1:]], dim=-1)
    return logits[..., class_idx] - others.max(dim=-1).values


def get_lpips(net: str = "vgg", device: Optional[str] = None):
    try:
        import lpips
    except ImportError as e:
        raise ImportError(
            "Perceptual path length needs the optional 'lpips' package: "
            "pip install 'latent_smoothness[ppl]'"
        ) from e
    device = resolve_device(device)
    key = (net, device)
    if key not in _lpips_cache:
        _lpips_cache[key] = lpips.LPIPS(net=net, verbose=False).to(device)
    return _lpips_cache[key]


def linear_interpolate(
    z_start: Tensor,
    z_end: Tensor,
    n_steps: int,
) -> Tensor:
    t = torch.linspace(0.0, 1.0, n_steps, device=z_start.device, dtype=z_start.dtype)
    return z_start.unsqueeze(1) + t.view(1, n_steps, 1) * (z_end - z_start).unsqueeze(1)


@contextmanager
def eval_mode(*modules: nn.Module, logger=None):
    module_names = [type(m).__name__ for m in modules]
    states = [m.training for m in modules]
    try:
        for m in modules:
            m.eval()
        yield
    except Exception as e:
        _get_logger(logger).error(
            f"[latent_smoothness] Exception in eval_mode context "
            f"(modules: {module_names}): {type(e).__name__}: {e}"
        )
        raise
    finally:
        for m, state in zip(modules, states):
            m.train(state)


@contextmanager
def timed_step(label: str, enabled: bool = True):
    """Print a start line and elapsed time for a diagnostic step (if ``enabled``).

    Writes to stderr so it does not pollute machine-readable stdout.
    """
    if enabled:
        print(f"  [suite] {label} ...", file=sys.stderr, flush=True)
    t0 = time.time()
    try:
        yield
    finally:
        if enabled:
            print(f"  [suite]   {label} done in {time.time() - t0:.1f}s",
                  file=sys.stderr, flush=True)
