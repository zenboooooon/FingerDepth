"""Deterministic discovery and identity assignment for training videos."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .config import PipelineConfig

VideoSplit = Literal["train", "validation"]
_HASH_CHUNK_SIZE = 1024 * 1024
_MAX_SLUG_LENGTH = 80


class VideoDiscoveryError(ValueError):
    """Raised when the input video collection is unsafe or incomplete."""


class DuplicateVideoError(VideoDiscoveryError):
    """Raised when two paths contain exactly the same source bytes."""


@dataclass(frozen=True, slots=True)
class DiscoveredVideo:
    """One immutable source-video descriptor used by later pipeline stages."""

    path: Path
    relative_path: Path
    split: VideoSplit
    sequence_id: str
    source_sha256: str
    size_bytes: int
    focal_35mm_mm: float

    @property
    def sha256(self) -> str:
        """Compatibility-friendly short name for the source byte digest."""

        return self.source_sha256

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.relative_path.as_posix(),
            "split": self.split,
            "sequence_id": self.sequence_id,
            "source_sha256": self.source_sha256,
            "size_bytes": self.size_bytes,
            "focal_35mm_mm": self.focal_35mm_mm,
        }


def sha256_file(path: Path) -> str:
    """Hash a source file without loading the complete video into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha256(path: Path) -> tuple[str, int]:
    before = path.stat()
    if before.st_size == 0:
        raise VideoDiscoveryError(f"training video is empty: {path}")
    digest = sha256_file(path)
    after = path.stat()
    before_identity = (before.st_size, before.st_mtime_ns)
    after_identity = (after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise VideoDiscoveryError(
            f"training video changed while it was being hashed (copy still in progress?): {path}"
        )
    return digest, before.st_size


def safe_slug(stem: str) -> str:
    """Return a stable lowercase ASCII slug, falling back for non-ASCII-only names."""

    normalized = unicodedata.normalize("NFKD", stem)
    ascii_stem = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_stem).strip("-")
    if not slug:
        return "video"
    truncated = slug[:_MAX_SLUG_LENGTH].rstrip("-")
    return truncated or "video"


def sequence_id_for(path: Path, source_sha256: str) -> str:
    """Build the filesystem-safe sequence identity from name and source content."""

    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    return f"{safe_slug(path.stem)}-{source_sha256[:12]}"


def _relative_to_project(path: Path, project_root: Path) -> Path:
    try:
        return path.relative_to(project_root)
    except ValueError as error:
        raise VideoDiscoveryError(f"video resolves outside the project root: {path}") from error


def _video_paths(root: Path, extensions: tuple[str, ...], *, split: VideoSplit) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"{split} video directory does not exist: {root}")
    paths = [
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(
        paths,
        key=lambda path: (
            path.relative_to(root).as_posix().casefold(),
            path.relative_to(root).as_posix(),
        ),
    )


def discover_videos(config: PipelineConfig) -> tuple[DiscoveredVideo, ...]:
    """Discover, hash and validate all configured train and validation videos.

    Results are always ordered by split (train before validation), then by the video's path
    relative to its split directory. Byte-identical files are rejected across both splits.
    """

    project_root = config.project_root.resolve()
    split_paths: list[tuple[VideoSplit, list[Path]]] = []
    for split, root in (
        ("train", config.input.train_dir),
        ("validation", config.input.validation_dir),
    ):
        paths = _video_paths(root, config.input.extensions, split=split)
        if not paths:
            raise VideoDiscoveryError(
                f"{split} video directory contains no supported videos: {root}"
            )
        split_paths.append((split, paths))

    discovered: list[DiscoveredVideo] = []
    digest_paths: dict[str, Path] = {}
    sequence_paths: dict[str, Path] = {}
    observed_paths: set[Path] = set()
    for split, paths in split_paths:
        for path in paths:
            relative_path = _relative_to_project(path, project_root)
            source_sha256, size_bytes = _stable_sha256(path)
            previous = digest_paths.get(source_sha256)
            if previous is not None:
                raise DuplicateVideoError(
                    "byte-identical training videos are not allowed: "
                    f"{_relative_to_project(previous, project_root)} and {relative_path}"
                )
            digest_paths[source_sha256] = path
            sequence_id = sequence_id_for(path, source_sha256)
            previous_sequence = sequence_paths.get(sequence_id)
            if previous_sequence is not None:
                raise VideoDiscoveryError(
                    f"sequence ID collision for {sequence_id}: {previous_sequence} and {path}"
                )
            sequence_paths[sequence_id] = path
            observed_paths.add(path)
            override = config.input.video_overrides.get(path)
            focal = (
                config.input.default_focal_35mm_mm if override is None else override.focal_35mm_mm
            )
            discovered.append(
                DiscoveredVideo(
                    path=path,
                    relative_path=relative_path,
                    split=split,
                    sequence_id=sequence_id,
                    source_sha256=source_sha256,
                    size_bytes=size_bytes,
                    focal_35mm_mm=focal,
                )
            )

    unused_overrides = sorted(
        set(config.input.video_overrides) - observed_paths,
        key=lambda path: path.as_posix(),
    )
    if unused_overrides:
        formatted = ", ".join(
            _relative_to_project(path, project_root).as_posix() for path in unused_overrides
        )
        raise VideoDiscoveryError(f"video override does not match a discovered video: {formatted}")
    return tuple(discovered)


__all__ = [
    "DiscoveredVideo",
    "DuplicateVideoError",
    "VideoDiscoveryError",
    "VideoSplit",
    "discover_videos",
    "safe_slug",
    "sequence_id_for",
    "sha256_file",
]
