from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import fingertip_depth.training_pipeline.discovery as discovery_module
from fingertip_depth.training_pipeline import (
    DuplicateVideoError,
    VideoDiscoveryError,
    discover_videos,
    load_pipeline_config,
)


def _load_config(project_root: Path, *, overrides: str = ""):
    config_path = project_root / "configs" / "training_pipeline.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        f"""
[input]
train_dir = "data/training_videos/train"
validation_dir = "data/training_videos/validation"
extensions = [".mov", ".mp4", ".m4v"]
default_focal_35mm_mm = 36.0

{overrides}
""",
        encoding="utf-8",
    )
    return load_pipeline_config(config_path, project_root=project_root)


def _video(project_root: Path, relative_path: str, content: bytes) -> Path:
    path = project_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_discovery_is_recursive_deterministic_and_applies_focal_override(tmp_path: Path) -> None:
    alpha = _video(tmp_path, "data/training_videos/train/nested/Alpha Clip.MOV", b"alpha")
    zeta = _video(tmp_path, "data/training_videos/train/zeta.mp4", b"zeta")
    valid = _video(tmp_path, "data/training_videos/validation/検証.m4v", b"valid")
    _video(tmp_path, "data/training_videos/train/ignored.avi", b"ignored")
    config = _load_config(
        tmp_path,
        overrides="""
[video_overrides."data/training_videos/train/nested/Alpha Clip.MOV"]
focal_35mm_mm = 26.0
""",
    )

    videos = discover_videos(config)

    assert [video.path for video in videos] == [alpha.resolve(), zeta.resolve(), valid.resolve()]
    assert [video.split for video in videos] == ["train", "train", "validation"]
    assert videos[0].focal_35mm_mm == 26.0
    assert videos[1].focal_35mm_mm == 36.0
    alpha_hash = hashlib.sha256(b"alpha").hexdigest()
    valid_hash = hashlib.sha256(b"valid").hexdigest()
    assert videos[0].sequence_id == f"alpha-clip-{alpha_hash[:12]}"
    assert videos[2].sequence_id == f"video-{valid_hash[:12]}"
    assert videos[0].source_sha256 == alpha_hash
    assert videos[0].sha256 == alpha_hash
    assert videos[0].relative_path == Path("data/training_videos/train/nested/Alpha Clip.MOV")
    assert videos[0].size_bytes == len(b"alpha")


def test_discovery_rejects_byte_identical_videos_across_splits(tmp_path: Path) -> None:
    _video(tmp_path, "data/training_videos/train/a.mov", b"same")
    _video(tmp_path, "data/training_videos/validation/b.mp4", b"same")
    config = _load_config(tmp_path)

    with pytest.raises(DuplicateVideoError, match="byte-identical"):
        discover_videos(config)


@pytest.mark.parametrize("empty_split", ["train", "validation"])
def test_discovery_requires_both_splits(tmp_path: Path, empty_split: str) -> None:
    train_dir = tmp_path / "data/training_videos/train"
    validation_dir = tmp_path / "data/training_videos/validation"
    train_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    if empty_split == "train":
        _video(tmp_path, "data/training_videos/validation/valid.mov", b"valid")
    else:
        _video(tmp_path, "data/training_videos/train/train.mov", b"train")
    config = _load_config(tmp_path)

    with pytest.raises(VideoDiscoveryError, match=f"{empty_split}.*no supported videos"):
        discover_videos(config)


def test_discovery_rejects_unused_video_override(tmp_path: Path) -> None:
    _video(tmp_path, "data/training_videos/train/train.mov", b"train")
    _video(tmp_path, "data/training_videos/validation/valid.mov", b"valid")
    config = _load_config(
        tmp_path,
        overrides="""
[video_overrides."data/training_videos/train/missing.mov"]
focal_35mm_mm = 28.0
""",
    )

    with pytest.raises(VideoDiscoveryError, match="does not match a discovered video"):
        discover_videos(config)


def test_discovery_rejects_zero_byte_video_before_hashing(tmp_path: Path) -> None:
    empty = _video(tmp_path, "data/training_videos/train/empty.mov", b"")
    _video(tmp_path, "data/training_videos/validation/valid.mov", b"valid")
    config = _load_config(tmp_path)

    with pytest.raises(VideoDiscoveryError, match="training video is empty"):
        discover_videos(config)
    assert empty.stat().st_size == 0


def test_discovery_rejects_video_changed_while_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    changing = _video(tmp_path, "data/training_videos/train/changing.mov", b"first")
    _video(tmp_path, "data/training_videos/validation/valid.mov", b"valid")
    config = _load_config(tmp_path)
    original_hash = discovery_module.sha256_file

    def hash_then_change(path: Path) -> str:
        digest = original_hash(path)
        if path == changing:
            path.write_bytes(path.read_bytes() + b"changed")
        return digest

    monkeypatch.setattr(discovery_module, "sha256_file", hash_then_change)

    with pytest.raises(VideoDiscoveryError, match="changed while it was being hashed"):
        discover_videos(config)
