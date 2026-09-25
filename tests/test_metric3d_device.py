import pytest

from fingertip_depth.metric3d import Metric3Dv2


def test_metric3d_restricts_upstream_to_cuda_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert str(Metric3Dv2._resolve_device("auto")) == "cuda:0"
    assert str(Metric3Dv2._resolve_device("cuda")) == "cuda:0"
    with pytest.raises(ValueError, match="cuda:0"):
        Metric3Dv2._resolve_device("cuda:1")
    with pytest.raises(ValueError, match="CUDA"):
        Metric3Dv2._resolve_device("cpu")


def test_metric3d_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires a CUDA GPU"):
        Metric3Dv2._resolve_device("auto")
