"""Verify Core ML predictions and deployment provenance on macOS."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DEFAULT_EXPORT_DIR = Path("outputs/phase8_student_coreml_iphone15")
DEFAULT_MANIFEST = DEFAULT_EXPORT_DIR / "export_manifest.json"
EXPORT_FORMAT = "fingertip-depth-student-coreml-export"
EXPORT_FORMAT_VERSION = 1
IMAGE_NAME = "image"
LANDMARK_NAME = "landmarks_xy"
OUTPUT_NAME = "depth_m"
IMAGE_SIZE = 224
LANDMARK_INDICES = (5, 6, 7, 8)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_EXPORT_DIR / "StudentDepth.mlpackage",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_EXPORT_DIR / "coreml_parity_inputs.npz",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help=(
            "checkpoint to hash; defaults to source.checkpoint_path in the export "
            "manifest, with a relocated outputs-directory fallback"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXPORT_DIR / "coreml_parity_report.json",
    )
    parser.add_argument("--mean-limit-m", type=float, default=0.001)
    parser.add_argument("--max-limit-m", type=float, default=0.003)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object: {path}")
    return value


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be an object")
    return value


def _sha256(value: object, *, field: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{field} must be a lowercase hexadecimal SHA-256")
    return digest


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(path: Path) -> dict[str, Any]:
    """Use the exact file/directory digest encoding used by the exporter."""

    path = path.resolve()
    if path.is_file():
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "byte_count": path.stat().st_size,
        }
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    byte_count = 0
    for child in sorted(value for value in path.rglob("*") if value.is_file()):
        relative = child.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(sha256_file(child).encode())
        byte_count += child.stat().st_size
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "byte_count": byte_count,
    }


def _verify_artifact(path: Path, declaration: object, *, field: str) -> dict[str, Any]:
    declared = _mapping(declaration, field=field)
    expected_hash = _sha256(declared.get("sha256"), field=f"{field}.sha256")
    expected_size = declared.get("byte_count")
    if not isinstance(expected_size, int) or expected_size < 0:
        raise ValueError(f"{field}.byte_count must be a non-negative integer")
    observed = artifact_record(path)
    if observed["sha256"] != expected_hash:
        raise ValueError(f"{field} SHA-256 differs from export manifest")
    if observed["byte_count"] != expected_size:
        raise ValueError(f"{field} byte count differs from export manifest")
    return {
        **observed,
        "declared_path": str(declared.get("path", "")),
        "verified_against_export_manifest": True,
    }


def _resolve_checkpoint(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    override: Path | None,
) -> Path:
    if override is not None:
        checkpoint = override.resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        return checkpoint

    source = _mapping(manifest.get("source"), field="source")
    declared = Path(str(source.get("checkpoint_path", "")))
    candidates = [declared]
    if declared.name:
        # Absolute paths in export manifests become stale when a repository is
        # copied to a Mac. The run and export directories remain siblings.
        candidates.append(manifest_path.parent.parent / declared.parent.name / declared.name)
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        "checkpoint from export manifest was not found; pass --checkpoint explicitly"
    )


def _validate_manifest_interface(manifest: Mapping[str, Any]) -> dict[str, Any]:
    interface = _mapping(manifest.get("interface"), field="interface")
    image = _mapping(interface.get("image"), field="interface.image")
    landmarks = _mapping(interface.get("landmarks"), field="interface.landmarks")
    output = _mapping(interface.get("output"), field="interface.output")

    expected_image = {
        "name": IMAGE_NAME,
        "coreml_type": "RGB image",
        "size": [IMAGE_SIZE, IMAGE_SIZE],
        "source_range": [0, 255],
    }
    for key, expected in expected_image.items():
        if image.get(key) != expected:
            raise ValueError(f"interface.image.{key} differs from the iOS contract")
    scale = image.get("coreml_scale")
    if not isinstance(scale, (int, float)) or not np.isclose(
        float(scale), 1.0 / 255.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("interface.image.coreml_scale differs from 1/255")

    if landmarks.get("name") != LANDMARK_NAME:
        raise ValueError("interface.landmarks.name differs from the iOS contract")
    if landmarks.get("shape") != [1, len(LANDMARK_INDICES), 2]:
        raise ValueError("interface.landmarks.shape differs from the iOS contract")
    if landmarks.get("indices") != list(LANDMARK_INDICES):
        raise ValueError("interface.landmarks.indices differs from the iOS contract")
    if landmarks.get("coordinate_range") != [0.0, 1.0]:
        raise ValueError("interface.landmarks.coordinate_range differs from the iOS contract")
    if landmarks.get("relative_z_included") is not False:
        raise ValueError("interface.landmarks.relative_z_included must be false")

    if output.get("name") != OUTPUT_NAME or output.get("shape") != [1, 1]:
        raise ValueError("interface.output differs from the iOS contract")
    if output.get("unit") != "metre":
        raise ValueError("interface.output.unit must be metre")
    return {
        "image": dict(image),
        "landmarks": dict(landmarks),
        "output": dict(output),
        "verified": True,
    }


def validate_export_provenance(
    *,
    manifest_path: Path,
    model_path: Path,
    fixture_path: Path,
    checkpoint_path: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify artifact bytes and manifest claims before Core ML is loaded."""

    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path)
    if (
        manifest.get("format") != EXPORT_FORMAT
        or manifest.get("format_version") != EXPORT_FORMAT_VERSION
    ):
        raise ValueError("unsupported Core ML export manifest")

    artifacts = _mapping(manifest.get("artifacts"), field="artifacts")
    source = _mapping(manifest.get("source"), field="source")
    checkpoint = _resolve_checkpoint(manifest_path, manifest, checkpoint_path)
    expected_checkpoint_hash = _sha256(
        source.get("checkpoint_sha256"), field="source.checkpoint_sha256"
    )
    checkpoint_hash = sha256_file(checkpoint)
    if checkpoint_hash != expected_checkpoint_hash:
        raise ValueError("checkpoint SHA-256 differs from export manifest")

    provenance = {
        "export_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "format": EXPORT_FORMAT,
            "format_version": EXPORT_FORMAT_VERSION,
        },
        "coreml_model": _verify_artifact(
            model_path, artifacts.get("coreml"), field="artifacts.coreml"
        ),
        "parity_fixture": _verify_artifact(
            fixture_path,
            artifacts.get("parity_fixture"),
            field="artifacts.parity_fixture",
        ),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": checkpoint_hash,
            "verified_against_export_manifest": True,
        },
        "manifest_interface": _validate_manifest_interface(manifest),
    }
    return manifest, provenance


def _enum_name(message: Any, field: str) -> str:
    descriptor = message.DESCRIPTOR.fields_by_name[field].enum_type
    numeric_value = int(getattr(message, field))
    value = descriptor.values_by_number.get(numeric_value)
    return value.name if value is not None else f"UNKNOWN_{numeric_value}"


def _feature_contract(feature: Any) -> dict[str, Any]:
    kind = feature.type.WhichOneof("Type")
    if kind == "imageType":
        image = feature.type.imageType
        return {
            "kind": kind,
            "width": int(image.width),
            "height": int(image.height),
            "color_space": _enum_name(image, "colorSpace"),
        }
    if kind == "multiArrayType":
        array = feature.type.multiArrayType
        return {
            "kind": kind,
            "shape": [int(value) for value in array.shape],
            "data_type": _enum_name(array, "dataType"),
        }
    return {"kind": str(kind)}


def extract_coreml_contract(model: Any) -> dict[str, Any]:
    description = model.get_spec().description
    return {
        "inputs": {feature.name: _feature_contract(feature) for feature in description.input},
        "outputs": {feature.name: _feature_contract(feature) for feature in description.output},
        "user_defined_metadata": dict(description.metadata.userDefined),
    }


def validate_coreml_contract(
    contract: Mapping[str, Any],
    *,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    inputs = _mapping(contract.get("inputs"), field="Core ML inputs")
    outputs = _mapping(contract.get("outputs"), field="Core ML outputs")
    metadata = _mapping(
        contract.get("user_defined_metadata"),
        field="Core ML user-defined metadata",
    )
    if set(inputs) != {IMAGE_NAME, LANDMARK_NAME}:
        raise ValueError("Core ML input names differ from the iOS contract")
    if set(outputs) != {OUTPUT_NAME}:
        raise ValueError("Core ML output names differ from the iOS contract")

    expected_image = {
        "kind": "imageType",
        "width": IMAGE_SIZE,
        "height": IMAGE_SIZE,
        "color_space": "RGB",
    }
    if dict(_mapping(inputs[IMAGE_NAME], field="Core ML image input")) != expected_image:
        raise ValueError("Core ML image input type differs from the iOS contract")
    expected_landmarks = {
        "kind": "multiArrayType",
        "shape": [1, len(LANDMARK_INDICES), 2],
        "data_type": "FLOAT32",
    }
    if dict(_mapping(inputs[LANDMARK_NAME], field="Core ML landmark input")) != expected_landmarks:
        raise ValueError("Core ML landmark input type differs from the iOS contract")
    expected_output = {
        "kind": "multiArrayType",
        "shape": [1, 1],
        "data_type": "FLOAT32",
    }
    if dict(_mapping(outputs[OUTPUT_NAME], field="Core ML depth output")) != expected_output:
        raise ValueError("Core ML depth output type differs from the iOS contract")

    expected_hash = _sha256(
        expected_checkpoint_sha256,
        field="expected checkpoint SHA-256",
    )
    if metadata.get("checkpoint_sha256") != expected_hash:
        raise ValueError("Core ML checkpoint metadata differs from export manifest")
    if metadata.get("input_landmark_indices") != ",".join(map(str, LANDMARK_INDICES)):
        raise ValueError("Core ML landmark-index metadata differs from the iOS contract")
    if metadata.get("image_resize") != "direct_bicubic_224x224_no_crop":
        raise ValueError("Core ML image-resize metadata differs from the iOS contract")
    return {
        "inputs": {name: dict(value) for name, value in inputs.items()},
        "outputs": {name: dict(value) for name, value in outputs.items()},
        "user_defined_metadata": dict(metadata),
        "verified": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        not np.isfinite(args.mean_limit_m)
        or not np.isfinite(args.max_limit_m)
        or args.mean_limit_m <= 0.0
        or args.max_limit_m <= 0.0
    ):
        raise ValueError("parity limits must be finite and positive")

    manifest, provenance = validate_export_provenance(
        manifest_path=args.manifest,
        model_path=args.model,
        fixture_path=args.fixture,
        checkpoint_path=args.checkpoint,
    )
    if platform.system() != "Darwin":
        raise RuntimeError("Core ML runtime verification must run on macOS")

    import coremltools as ct

    model = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.ALL)
    source = _mapping(manifest.get("source"), field="source")
    model_contract = validate_coreml_contract(
        extract_coreml_contract(model),
        expected_checkpoint_sha256=str(source.get("checkpoint_sha256")),
    )

    with np.load(args.fixture, allow_pickle=False) as fixture:
        images = fixture["images_rgb_uint8"]
        landmarks = fixture["landmarks_xy"]
        expected = fixture["pytorch_depth_m"].astype(np.float64)
        frame_indices = fixture["frame_index"]
    count = images.shape[0]
    if images.dtype != np.uint8 or images.shape != (count, IMAGE_SIZE, IMAGE_SIZE, 3):
        raise ValueError("parity image array must be uint8 [N,224,224,3]")
    if landmarks.dtype != np.float32 or landmarks.shape != (
        count,
        len(LANDMARK_INDICES),
        2,
    ):
        raise ValueError("parity landmark array must be float32 [N,4,2]")
    if expected.shape != (count,) or frame_indices.shape != (count,):
        raise ValueError("parity fixture arrays have inconsistent lengths")
    if count == 0 or not np.all(np.isfinite(expected)) or np.any(expected <= 0.0):
        raise ValueError("parity expected depths must be non-empty, finite, and positive")
    if not np.all(np.isfinite(landmarks)) or np.any((landmarks < 0.0) | (landmarks > 1.0)):
        raise ValueError("parity landmarks must be finite and in [0,1]")

    validation = _mapping(manifest.get("validation"), field="validation")
    fixture_claim = _mapping(
        validation.get("parity_fixture"),
        field="validation.parity_fixture",
    )
    if fixture_claim.get("sample_count") != count:
        raise ValueError("parity fixture sample count differs from export manifest")
    if fixture_claim.get("image_shape") != list(images.shape):
        raise ValueError("parity fixture image shape differs from export manifest")
    if fixture_claim.get("landmark_shape") != list(landmarks.shape):
        raise ValueError("parity fixture landmark shape differs from export manifest")

    observed: list[float] = []
    latency_ms: list[float] = []
    for index, (image, landmark_xy) in enumerate(
        zip(images, landmarks, strict=True),
        start=1,
    ):
        started = time.perf_counter()
        result = model.predict(
            {
                IMAGE_NAME: Image.fromarray(image, mode="RGB"),
                LANDMARK_NAME: landmark_xy[np.newaxis].astype(np.float32, copy=False),
            }
        )
        latency_ms.append((time.perf_counter() - started) * 1000.0)
        observed.append(float(np.asarray(result[OUTPUT_NAME]).reshape(-1)[0]))
        if index % 50 == 0 or index == len(images):
            print(f"Core ML parity {index}/{len(images)}", flush=True)

    observed_array = np.asarray(observed, dtype=np.float64)
    if not np.all(np.isfinite(observed_array)):
        raise ValueError("Core ML predictions must be finite")
    absolute_error = np.abs(observed_array - expected)
    mean_error_m = float(np.mean(absolute_error))
    max_error_m = float(np.max(absolute_error))
    passed = mean_error_m <= args.mean_limit_m and max_error_m <= args.max_limit_m
    worst_index = int(np.argmax(absolute_error))
    report = {
        "format": "fingertip-depth-coreml-parity-report",
        "format_version": 2,
        "passed": passed,
        "sample_count": len(images),
        "provenance": provenance,
        "model_contract": model_contract,
        "absolute_difference": {
            "mean_m": mean_error_m,
            "mean_mm": mean_error_m * 1000.0,
            "max_m": max_error_m,
            "max_mm": max_error_m * 1000.0,
            "mean_limit_m": args.mean_limit_m,
            "max_limit_m": args.max_limit_m,
            "worst_frame_index": int(frame_indices[worst_index]),
        },
        "latency_ms": {
            "median": float(np.median(latency_ms)),
            "p95": float(np.percentile(latency_ms, 95)),
            "mean": float(np.mean(latency_ms)),
        },
        "environment": {
            "macos": platform.mac_ver()[0],
            "machine": platform.machine(),
            "coremltools": str(ct.__version__),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as destination:
        json.dump(report, destination, ensure_ascii=False, indent=2)
        destination.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
