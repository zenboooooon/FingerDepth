from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "verify_student_coreml.py"
_SPEC = importlib.util.spec_from_file_location("verify_student_coreml_for_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
verification = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verification)


def _coreml_contract(checkpoint_sha256: str) -> dict[str, Any]:
    return {
        "inputs": {
            "image": {
                "kind": "imageType",
                "width": 224,
                "height": 224,
                "color_space": "RGB",
            },
            "landmarks_xy": {
                "kind": "multiArrayType",
                "shape": [1, 4, 2],
                "data_type": "FLOAT32",
            },
        },
        "outputs": {
            "depth_m": {
                "kind": "multiArrayType",
                "shape": [1, 1],
                "data_type": "FLOAT32",
            }
        },
        "user_defined_metadata": {
            "checkpoint_sha256": checkpoint_sha256,
            "input_landmark_indices": "5,6,7,8",
            "image_resize": "direct_bicubic_224x224_no_crop",
        },
    }


def _manifest(
    *,
    model_path: Path,
    fixture_path: Path,
    checkpoint_path: Path,
    sample_count: int = 1,
) -> dict[str, Any]:
    return {
        "format": verification.EXPORT_FORMAT,
        "format_version": verification.EXPORT_FORMAT_VERSION,
        "source": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": verification.sha256_file(checkpoint_path),
        },
        "interface": {
            "image": {
                "name": "image",
                "coreml_type": "RGB image",
                "size": [224, 224],
                "source_range": [0, 255],
                "coreml_scale": 1.0 / 255.0,
            },
            "landmarks": {
                "name": "landmarks_xy",
                "shape": [1, 4, 2],
                "indices": [5, 6, 7, 8],
                "coordinate_range": [0.0, 1.0],
                "relative_z_included": False,
            },
            "output": {
                "name": "depth_m",
                "shape": [1, 1],
                "unit": "metre",
            },
        },
        "validation": {
            "parity_fixture": {
                "sample_count": sample_count,
                "image_shape": [sample_count, 224, 224, 3],
                "landmark_shape": [sample_count, 4, 2],
            }
        },
        "artifacts": {
            "coreml": verification.artifact_record(model_path),
            "parity_fixture": verification.artifact_record(fixture_path),
        },
    }


def _artifacts(tmp_path: Path, *, real_fixture: bool = False) -> dict[str, Path]:
    model = tmp_path / "StudentDepth.mlpackage"
    (model / "Data").mkdir(parents=True)
    (model / "Manifest.json").write_text('{"model":1}\n', encoding="utf-8")
    (model / "Data" / "weights.bin").write_bytes(b"weights")

    fixture = tmp_path / "coreml_parity_inputs.npz"
    if real_fixture:
        np.savez_compressed(
            fixture,
            images_rgb_uint8=np.zeros((1, 224, 224, 3), dtype=np.uint8),
            landmarks_xy=np.full((1, 4, 2), 0.5, dtype=np.float32),
            pytorch_depth_m=np.asarray([0.25], dtype=np.float32),
            frame_index=np.asarray([17], dtype=np.int32),
        )
    else:
        fixture.write_bytes(b"fixture")

    checkpoint = tmp_path / "best_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    manifest = tmp_path / "export_manifest.json"
    manifest.write_text(
        json.dumps(
            _manifest(
                model_path=model,
                fixture_path=fixture,
                checkpoint_path=checkpoint,
            )
        ),
        encoding="utf-8",
    )
    return {
        "model": model,
        "fixture": fixture,
        "checkpoint": checkpoint,
        "manifest": manifest,
    }


def test_validate_export_provenance_records_all_hashes(tmp_path: Path) -> None:
    paths = _artifacts(tmp_path)

    manifest, provenance = verification.validate_export_provenance(
        manifest_path=paths["manifest"],
        model_path=paths["model"],
        fixture_path=paths["fixture"],
        checkpoint_path=paths["checkpoint"],
    )

    assert manifest["format"] == verification.EXPORT_FORMAT
    assert provenance["export_manifest"]["sha256"] == verification.sha256_file(paths["manifest"])
    assert (
        provenance["coreml_model"]["sha256"]
        == verification.artifact_record(paths["model"])["sha256"]
    )
    assert provenance["parity_fixture"]["sha256"] == verification.sha256_file(paths["fixture"])
    assert provenance["checkpoint"]["sha256"] == verification.sha256_file(paths["checkpoint"])
    assert provenance["manifest_interface"]["verified"] is True


@pytest.mark.parametrize("tampered", ["model", "fixture", "checkpoint"])
def test_validate_export_provenance_rejects_tampering(
    tmp_path: Path,
    tampered: str,
) -> None:
    paths = _artifacts(tmp_path)
    if tampered == "model":
        (paths["model"] / "Data" / "weights.bin").write_bytes(b"changed")
    else:
        paths[tampered].write_bytes(b"changed")

    with pytest.raises(ValueError, match="SHA-256 differs"):
        verification.validate_export_provenance(
            manifest_path=paths["manifest"],
            model_path=paths["model"],
            fixture_path=paths["fixture"],
            checkpoint_path=paths["checkpoint"],
        )


def test_validate_coreml_contract_checks_metadata_and_interface() -> None:
    checkpoint_hash = "a" * 64
    verified = verification.validate_coreml_contract(
        _coreml_contract(checkpoint_hash),
        expected_checkpoint_sha256=checkpoint_hash,
    )
    assert verified["verified"] is True

    wrong_metadata = _coreml_contract(checkpoint_hash)
    wrong_metadata["user_defined_metadata"]["checkpoint_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="checkpoint metadata"):
        verification.validate_coreml_contract(
            wrong_metadata,
            expected_checkpoint_sha256=checkpoint_hash,
        )

    wrong_shape = _coreml_contract(checkpoint_hash)
    wrong_shape["inputs"]["landmarks_xy"]["shape"] = [1, 8]
    with pytest.raises(ValueError, match="landmark input type"):
        verification.validate_coreml_contract(
            wrong_shape,
            expected_checkpoint_sha256=checkpoint_hash,
        )


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--mean-limit-m", "inf"), ("--max-limit-m", "nan")],
)
def test_main_rejects_nonfinite_limits(flag: str, value: str) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        verification.main([flag, value])


def test_main_records_preflight_in_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _artifacts(tmp_path, real_fixture=True)
    checkpoint_hash = verification.sha256_file(paths["checkpoint"])
    contract = _coreml_contract(checkpoint_hash)

    class FakeModel:
        def predict(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
            assert set(inputs) == {"image", "landmarks_xy"}
            return {"depth_m": np.asarray([[0.25]], dtype=np.float32)}

    fake_coremltools = types.SimpleNamespace(
        __version__="test",
        ComputeUnit=types.SimpleNamespace(ALL="all"),
        models=types.SimpleNamespace(MLModel=lambda *args, **kwargs: FakeModel()),
    )
    monkeypatch.setitem(sys.modules, "coremltools", fake_coremltools)
    monkeypatch.setattr(verification.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(verification, "extract_coreml_contract", lambda model: contract)

    output = tmp_path / "report.json"
    result = verification.main(
        [
            "--model",
            str(paths["model"]),
            "--fixture",
            str(paths["fixture"]),
            "--manifest",
            str(paths["manifest"]),
            "--checkpoint",
            str(paths["checkpoint"]),
            "--output",
            str(output),
        ]
    )

    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["format_version"] == 2
    assert report["passed"] is True
    assert report["provenance"]["export_manifest"]["sha256"] == verification.sha256_file(
        paths["manifest"]
    )
    assert report["provenance"]["checkpoint"]["sha256"] == checkpoint_hash
    assert report["model_contract"]["verified"] is True
