'比較実験で使う UniDepth と Depth Pro の推論アダプターを提供します。任意依存ライブラリや重いモデルは必要になった時点で読み込みます。\n\nモデル取得や任意ライブラリの読み込みを比較処理から分離し、対応する依存環境でのみ推論器を初期化します。'

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .camera import CameraIntrinsics

UNIDEPTH_SOURCE_REPOSITORY = "https://github.com/lpiccinelli-eth/UniDepth"
UNIDEPTH_SOURCE_COMMIT = "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"
UNIDEPTH_HUB_REPOSITORY = f"lpiccinelli-eth/UniDepth:{UNIDEPTH_SOURCE_COMMIT}"
UNIDEPTH_HF_REPOSITORY = "lpiccinelli/unidepth-v2-vitl14"
UNIDEPTH_HF_REVISION = "52b349b514bd8b47642f67ac78cb7b5dc5c51dd9"
UNIDEPTH_CHECKPOINT_FILENAME = "model.safetensors"
UNIDEPTH_CHECKPOINT_SHA256 = (
    "ba73d3de735302ccc64a50f1e557122050c4b1893e6060b28dba05d6af3e67c6"
)

DEPTH_PRO_SOURCE_REPOSITORY = "https://github.com/apple/ml-depth-pro"
DEPTH_PRO_SOURCE_COMMIT = "9e65e4dbe9568d23c546fcec53302b10445e109e"
DEPTH_PRO_HF_REPOSITORY = "apple/DepthPro"
DEPTH_PRO_HF_REVISION = "ccd1350a774eb2248bcdfb3be430e38f1d3087ef"
DEPTH_PRO_CHECKPOINT_FILENAME = "depth_pro.pt"
DEPTH_PRO_CHECKPOINT_SHA256 = (
    "3eb35ca68168ad3d14cb150f8947a4edf85589941661fdb2686259c80685c0ce"
)

_UNIDEPTH_CAMERA_MODES = frozenset({"approx_k", "no_camera"})
_DEPTH_PRO_CAMERA_MODES = frozenset({"approx_focal", "estimated_focal"})


# 深度推定値と、推論に使ったモデル・入力サイズなどの情報をまとめて保持します。
@dataclass(frozen=True, slots=True)
class AlternativeDepthPrediction:
    """One original-resolution metric-depth result."""

    depth_m: np.ndarray
    inference_ms: float
    device: str
    extras: dict[str, Any]


# RGB画像が仕様を満たすことを検証します。
def _validate_rgb(rgb: np.ndarray) -> tuple[int, int]:
    if not isinstance(rgb, np.ndarray):
        raise TypeError("RGB input must be a numpy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB input must have shape (height, width, 3)")
    if rgb.dtype != np.uint8:
        raise ValueError("RGB input must use uint8 values")
    height, width = rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("RGB input must not be empty")
    return height, width


# 指定条件と利用可能な演算装置から推論デバイスを選びます。
def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return resolved


# CUDAを使う場合はGPU処理の完了を待ち、推論時間を正しく計測できるようにします。
def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# 指定したファイルの内容からSHA-256を計算します。
def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# チェックポイントの記録値と実データが一致するか照合します。
def verify_checkpoint(path: Path, *, expected_sha256: str, model_name: str) -> str:
    """Verify one pinned model artifact before it can be deserialized."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{model_name} checkpoint was not found: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{model_name} checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual}"
        )
    return actual


# Hugging Face Hubの取得関数を遅延インポートし、指定リビジョンのモデルファイルを取得します。
def _hf_hub_download(**kwargs: Any) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as error:  # pragma: no cover - depends on optional group
        raise RuntimeError(
            "huggingface-hub is required for alternative depth model downloads; "
            "run this command through the matching uv dependency group"
        ) from error
    return hf_hub_download(**kwargs)


# 指定リポジトリの固定チェックポイントを取得し、期待するSHA-256と一致するか検証します。
def _ensure_hf_checkpoint(
    *,
    repository: str,
    revision: str,
    filename: str,
    expected_sha256: str,
    model_name: str,
    downloader: Callable[..., str] | None = None,
) -> Path:
    download = _hf_hub_download if downloader is None else downloader
    checkpoint = Path(
        download(
            repo_id=repository,
            revision=revision,
            filename=filename,
            repo_type="model",
        )
    )
    verify_checkpoint(
        checkpoint,
        expected_sha256=expected_sha256,
        model_name=model_name,
    )
    return checkpoint


# 固定したUniDepth v2チェックポイントを取得し、SHA-256を検証してパスを返します。
def ensure_unidepth_checkpoint(
    *, downloader: Callable[..., str] | None = None
) -> Path:
    """Resolve and verify the immutable UniDepth V2-L safetensors artifact."""

    return _ensure_hf_checkpoint(
        repository=UNIDEPTH_HF_REPOSITORY,
        revision=UNIDEPTH_HF_REVISION,
        filename=UNIDEPTH_CHECKPOINT_FILENAME,
        expected_sha256=UNIDEPTH_CHECKPOINT_SHA256,
        model_name="UniDepth V2-L",
        downloader=downloader,
    )


# 固定したDepth Proチェックポイントを取得し、SHA-256を検証してパスを返します。
def ensure_depth_pro_checkpoint(
    *, downloader: Callable[..., str] | None = None
) -> Path:
    """Resolve and verify the immutable Depth Pro checkpoint."""

    return _ensure_hf_checkpoint(
        repository=DEPTH_PRO_HF_REPOSITORY,
        revision=DEPTH_PRO_HF_REVISION,
        filename=DEPTH_PRO_CHECKPOINT_FILENAME,
        expected_sha256=DEPTH_PRO_CHECKPOINT_SHA256,
        model_name="Depth Pro",
        downloader=downloader,
    )


# 深度出力をCPU上の有限なfloat32配列へ変換し、元画像と同じ寸法・非負値であることを検証します。
def _as_original_depth(
    value: Any,
    *,
    height: int,
    width: int,
    model_name: str,
) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{model_name} returned a non-tensor depth map")
    depth = value.detach().to(device="cpu", dtype=torch.float32).squeeze()
    if depth.ndim != 2:
        raise ValueError(f"{model_name} returned unexpected depth shape {tuple(value.shape)}")
    if tuple(depth.shape) != (height, width):
        raise ValueError(
            f"{model_name} depth does not match the original RGB size: "
            f"got {tuple(depth.shape)}, expected {(height, width)}"
        )
    depth_m = depth.numpy()
    if not np.all(np.isfinite(depth_m)):
        raise ValueError(f"{model_name} returned non-finite metric depth")
    if np.any(depth_m < 0.0):
        raise ValueError(f"{model_name} returned negative metric depth")
    return np.ascontiguousarray(depth_m, dtype=np.float32)


# 焦点距離と主点から3×3のカメラ内部パラメーター行列を作り、指定デバイスのTensorにします。
def _intrinsics_tensor(intrinsics: CameraIntrinsics, device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [
            [intrinsics.fx_px, 0.0, intrinsics.cx_px],
            [0.0, intrinsics.fy_px, intrinsics.cy_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


# UniDepthの出力行列から焦点距離と主点を読み取り、辞書形式で返します。
def _intrinsics_from_output(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, torch.Tensor):
        raise TypeError("UniDepth returned non-tensor intrinsics")
    matrix = value.detach().to(device="cpu", dtype=torch.float32).squeeze()
    if tuple(matrix.shape) != (3, 3):
        raise ValueError(f"UniDepth returned unexpected intrinsics shape {tuple(value.shape)}")
    values = matrix.numpy()
    return {
        "fx_px": float(values[0, 0]),
        "fy_px": float(values[1, 1]),
        "cx_px": float(values[0, 2]),
        "cy_px": float(values[1, 2]),
    }


# UniDepth v2モデルを遅延ロードし、画像からメートル単位の深度を推定します。
class UniDepthV2L:
    """Pinned UniDepth V2-L adapter for K-supplied and camera-free inference."""

    # 必要な引数を検証し、インスタンスの状態を初期化します。
    def __init__(
        self,
        *,
        device: str = "auto",
        checkpoint_path: Path | None = None,
        model: Any | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self._model = model
        self._load_details: dict[str, Any] = {}

    # モデル名・版・実行条件など推論モデルの情報を返します。
    @property
    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "backend": "unidepth_v2_l",
            "model": "UniDepth V2-L (ViT-L/14)",
            "source_repository": UNIDEPTH_SOURCE_REPOSITORY,
            "source_commit": UNIDEPTH_SOURCE_COMMIT,
            "checkpoint_repository": UNIDEPTH_HF_REPOSITORY,
            "checkpoint_revision": UNIDEPTH_HF_REVISION,
            "checkpoint_filename": UNIDEPTH_CHECKPOINT_FILENAME,
            "checkpoint_sha256": UNIDEPTH_CHECKPOINT_SHA256,
            "license": "CC-BY-NC-4.0",
            "device": str(self.device),
            "depth_unit": "metre",
            "depth_semantics": "optical_axis_z",
        }
        metadata.update(self._load_details)
        return metadata

    # モデル名・版・実行条件など推論モデルの情報を返します。
    @property
    def model_metadata(self) -> dict[str, Any]:
        return self.metadata

    # 固定チェックポイントを使ってUniDepth v2モデルを構築し、指定デバイスへ配置します。
    def _load_model(self) -> Any:
        if self._checkpoint_path is None:
            checkpoint = ensure_unidepth_checkpoint()
        else:
            checkpoint = self._checkpoint_path
            verify_checkpoint(
                checkpoint,
                expected_sha256=UNIDEPTH_CHECKPOINT_SHA256,
                model_name="UniDepth V2-L",
            )
        try:
            from safetensors.torch import load_file
        except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "safetensors is required for UniDepth V2-L; "
                "run this command through the UniDepth uv dependency group"
            ) from error

        model = torch.hub.load(
            UNIDEPTH_HUB_REPOSITORY,
            "UniDepth",
            version="v2",
            backbone="vitl14",
            pretrained=False,
            trust_repo=True,
            skip_validation=True,
        )
        state_dict = load_file(str(checkpoint), device="cpu")
        load_info = model.load_state_dict(state_dict, strict=False)
        missing = list(getattr(load_info, "missing_keys", ()))
        unexpected = list(getattr(load_info, "unexpected_keys", ()))
        self._load_details = {
            "checkpoint_path": str(checkpoint),
            "missing_key_count": len(missing),
            "unexpected_key_count": len(unexpected),
        }
        if missing or unexpected:
            raise RuntimeError(
                "pinned UniDepth checkpoint is incompatible with the pinned source: "
                f"{len(missing)} missing and {len(unexpected)} unexpected keys"
            )
        return model.to(self.device).eval()

    # 必要時にモデル本体を読み込み、再利用できる形で返します。
    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    # 入力からモデルの深度予測を計算し、値と推論情報を返します。
    def predict(
        self,
        rgb: np.ndarray,
        *,
        intrinsics: CameraIntrinsics | None,
        camera_mode: str,
    ) -> AlternativeDepthPrediction:
        height, width = _validate_rgb(rgb)
        if camera_mode not in _UNIDEPTH_CAMERA_MODES:
            choices = ", ".join(sorted(_UNIDEPTH_CAMERA_MODES))
            raise ValueError(f"unsupported UniDepth camera_mode {camera_mode!r}; choose {choices}")
        if camera_mode == "approx_k" and intrinsics is None:
            raise ValueError("UniDepth approx_k mode requires camera intrinsics")
        if camera_mode == "no_camera" and intrinsics is not None:
            raise ValueError("UniDepth no_camera mode requires intrinsics=None")

        rgb_tensor = torch.from_numpy(
            np.ascontiguousarray(rgb.transpose(2, 0, 1))
        ).to(self.device)
        camera = (
            _intrinsics_tensor(intrinsics, self.device)
            if camera_mode == "approx_k" and intrinsics is not None
            else None
        )

        model = self.model
        _synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.infer(rgb_tensor, camera)
        _synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0

        if not isinstance(output, Mapping) or "depth" not in output:
            raise TypeError("UniDepth returned an invalid inference result")
        depth_m = _as_original_depth(
            output["depth"],
            height=height,
            width=width,
            model_name="UniDepth",
        )
        extras: dict[str, Any] = {
            "camera_mode": camera_mode,
            "input_intrinsics": intrinsics.as_dict() if intrinsics is not None else None,
            "predicted_intrinsics": _intrinsics_from_output(output.get("intrinsics")),
            "depth_unit": "metre",
            "depth_semantics": "optical_axis_z",
            "source_commit": UNIDEPTH_SOURCE_COMMIT,
            "checkpoint_revision": UNIDEPTH_HF_REVISION,
            "checkpoint_sha256": UNIDEPTH_CHECKPOINT_SHA256,
        }
        return AlternativeDepthPrediction(
            depth_m=depth_m,
            inference_ms=inference_ms,
            device=str(self.device),
            extras=extras,
        )


# Depth Proモデルを遅延ロードし、画像から深度を推定する比較用アダプターです。
class DepthProEstimator:
    """Pinned Depth Pro adapter for supplied-focal and inferred-focal inference."""

    # 必要な引数を検証し、インスタンスの状態を初期化します。
    def __init__(
        self,
        *,
        device: str = "auto",
        checkpoint_path: Path | None = None,
        model: Any | None = None,
        transform: Callable[[np.ndarray], torch.Tensor] | None = None,
    ) -> None:
        if (model is None) != (transform is None):
            raise ValueError("model and transform must either both be supplied or both be omitted")
        self.device = _resolve_device(device)
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self._model = model
        self._transform = transform
        self._precision = torch.float16 if self.device.type == "cuda" else torch.float32
        self._load_details: dict[str, Any] = {}

    # モデル名・版・実行条件など推論モデルの情報を返します。
    @property
    def metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "backend": "depth_pro",
            "model": "Depth Pro",
            "source_repository": DEPTH_PRO_SOURCE_REPOSITORY,
            "source_commit": DEPTH_PRO_SOURCE_COMMIT,
            "checkpoint_repository": DEPTH_PRO_HF_REPOSITORY,
            "checkpoint_revision": DEPTH_PRO_HF_REVISION,
            "checkpoint_filename": DEPTH_PRO_CHECKPOINT_FILENAME,
            "checkpoint_sha256": DEPTH_PRO_CHECKPOINT_SHA256,
            "license": "Apple Machine Learning Research Model License",
            "device": str(self.device),
            "precision": str(self._precision).removeprefix("torch."),
            "depth_unit": "metre",
        }
        metadata.update(self._load_details)
        return metadata

    # モデル名・版・実行条件など推論モデルの情報を返します。
    @property
    def model_metadata(self) -> dict[str, Any]:
        return self.metadata

    # Depth Proのモデルと、入力画像をモデル用Tensorへ変換する前処理器を読み込みます。
    def _load_model_and_transform(self) -> tuple[Any, Callable[[np.ndarray], torch.Tensor]]:
        if self._checkpoint_path is None:
            checkpoint = ensure_depth_pro_checkpoint()
        else:
            checkpoint = self._checkpoint_path
            verify_checkpoint(
                checkpoint,
                expected_sha256=DEPTH_PRO_CHECKPOINT_SHA256,
                model_name="Depth Pro",
            )
        try:
            from depth_pro.depth_pro import (
                DEFAULT_MONODEPTH_CONFIG_DICT,
                create_model_and_transforms,
            )
        except ModuleNotFoundError as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "the pinned depth-pro package is required; "
                "run this command through the Depth Pro uv dependency group"
            ) from error

        config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(checkpoint))
        model, transform = create_model_and_transforms(
            config=config,
            device=self.device,
            precision=self._precision,
        )
        self._load_details = {"checkpoint_path": str(checkpoint)}
        return model.eval(), transform

    # Depth Proのモデルと前処理器を、未ロードの場合だけ初期化します。
    def _ensure_loaded(self) -> tuple[Any, Callable[[np.ndarray], torch.Tensor]]:
        if self._model is None or self._transform is None:
            self._model, self._transform = self._load_model_and_transform()
        return self._model, self._transform

    # 入力からモデルの深度予測を計算し、値と推論情報を返します。
    def predict(
        self,
        rgb: np.ndarray,
        *,
        intrinsics: CameraIntrinsics | None,
        camera_mode: str,
    ) -> AlternativeDepthPrediction:
        height, width = _validate_rgb(rgb)
        if camera_mode not in _DEPTH_PRO_CAMERA_MODES:
            choices = ", ".join(sorted(_DEPTH_PRO_CAMERA_MODES))
            raise ValueError(f"unsupported Depth Pro camera_mode {camera_mode!r}; choose {choices}")
        if camera_mode == "approx_focal" and intrinsics is None:
            raise ValueError("Depth Pro approx_focal mode requires camera intrinsics")
        if camera_mode == "estimated_focal" and intrinsics is not None:
            raise ValueError("Depth Pro estimated_focal mode requires intrinsics=None")

        model, transform = self._ensure_loaded()
        tensor = transform(rgb)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Depth Pro transform returned a non-tensor value")
        focal_input = (
            torch.tensor(
                intrinsics.fx_px,
                device=self.device,
                dtype=torch.float32,
            )
            if camera_mode == "approx_focal" and intrinsics is not None
            else None
        )

        _synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            output = model.infer(tensor, f_px=focal_input)
        _synchronize(self.device)
        inference_ms = (time.perf_counter() - started) * 1000.0

        if not isinstance(output, Mapping) or "depth" not in output:
            raise TypeError("Depth Pro returned an invalid inference result")
        depth_m = _as_original_depth(
            output["depth"],
            height=height,
            width=width,
            model_name="Depth Pro",
        )
        output_focal = output.get("focallength_px")
        if not isinstance(output_focal, torch.Tensor) or output_focal.numel() != 1:
            raise TypeError("Depth Pro returned an invalid focal-length value")
        output_focal_px = float(output_focal.detach().to(device="cpu", dtype=torch.float32))
        if not math.isfinite(output_focal_px) or output_focal_px <= 0.0:
            raise ValueError("Depth Pro returned a non-positive or non-finite focal length")

        extras: dict[str, Any] = {
            "camera_mode": camera_mode,
            "input_focal_px": intrinsics.fx_px if intrinsics is not None else None,
            "output_focal_px": output_focal_px,
            "depth_unit": "metre",
            "source_commit": DEPTH_PRO_SOURCE_COMMIT,
            "checkpoint_revision": DEPTH_PRO_HF_REVISION,
            "checkpoint_sha256": DEPTH_PRO_CHECKPOINT_SHA256,
        }
        return AlternativeDepthPrediction(
            depth_m=depth_m,
            inference_ms=inference_ms,
            device=str(self.device),
            extras=extras,
        )


__all__ = [
    "DEPTH_PRO_CHECKPOINT_FILENAME",
    "DEPTH_PRO_CHECKPOINT_SHA256",
    "DEPTH_PRO_HF_REPOSITORY",
    "DEPTH_PRO_HF_REVISION",
    "DEPTH_PRO_SOURCE_COMMIT",
    "UNIDEPTH_CHECKPOINT_FILENAME",
    "UNIDEPTH_CHECKPOINT_SHA256",
    "UNIDEPTH_HF_REPOSITORY",
    "UNIDEPTH_HF_REVISION",
    "UNIDEPTH_SOURCE_COMMIT",
    "AlternativeDepthPrediction",
    "DepthProEstimator",
    "UniDepthV2L",
    "ensure_depth_pro_checkpoint",
    "ensure_unidepth_checkpoint",
    "sha256_file",
    "verify_checkpoint",
]
