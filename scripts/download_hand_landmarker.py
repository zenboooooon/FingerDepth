"""Download and verify the versioned MediaPipe Hand Landmarker model asset."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
import urllib.request
from pathlib import Path

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path) -> str:
    actual = sha256(path)
    if actual != HAND_LANDMARKER_SHA256:
        raise RuntimeError(
            f"Hand Landmarker SHA-256 mismatch: expected {HAND_LANDMARKER_SHA256}, got {actual}"
        )
    return actual


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/hand_landmarker.task"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.is_file():
        print(f"verified: {args.output} sha256={verify(args.output)}")
        return 0

    with tempfile.NamedTemporaryFile(
        prefix="hand_landmarker_", suffix=".task", delete=False, dir=args.output.parent
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        urllib.request.urlretrieve(HAND_LANDMARKER_URL, temporary_path)
        digest = verify(temporary_path)
        temporary_path.replace(args.output)
    finally:
        temporary_path.unlink(missing_ok=True)
    print(f"downloaded and verified: {args.output} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
