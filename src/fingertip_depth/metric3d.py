"""Public Metric3D v2 adapter with the upstream CUDA-device constraint enforced."""

from __future__ import annotations

import torch

from ._metric3d_impl import (
    DepthPrediction,
    PreprocessGeometry,
    ensure_cached_checkpoint,
    prepare_metric3d_input,
    restore_metric_depth,
    verify_cached_checkpoint,
)
from ._metric3d_impl import (
    Metric3Dv2 as _Metric3Dv2,
)


class Metric3Dv2(_Metric3Dv2):
    """Pinned model restricted to cuda:0 because the upstream decoder hard-codes it."""

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("the pinned Metric3D v2 implementation requires a CUDA GPU")
        resolved = torch.device("cuda:0" if device == "auto" else device)
        if resolved.type != "cuda":
            raise ValueError("the pinned Metric3D v2 implementation only supports CUDA devices")
        if resolved.index not in (None, 0):
            raise ValueError("the pinned Metric3D v2 implementation only supports cuda:0")
        return torch.device("cuda:0")


__all__ = [
    "DepthPrediction",
    "Metric3Dv2",
    "PreprocessGeometry",
    "ensure_cached_checkpoint",
    "prepare_metric3d_input",
    "restore_metric_depth",
    "verify_cached_checkpoint",
]
