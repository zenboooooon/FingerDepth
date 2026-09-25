"""Create a short deterministic video from one image for API smoke testing only."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--fps", type=float, default=6.0)
    args = parser.parse_args()
    if args.frames <= 0 or args.fps <= 0:
        raise ValueError("frames and fps must be positive")

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"unable to read {args.input}")
    height, width = image.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise OSError(f"unable to create {args.output}")
    try:
        for index in range(args.frames):
            phase = 2.0 * math.pi * index / args.frames
            scale = 1.0 + 0.015 * math.sin(phase)
            transform = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), 0, scale)
            transform[0, 2] += 2.0 * math.cos(phase)
            frame = cv2.warpAffine(
                image,
                transform,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
            writer.write(frame)
    finally:
        writer.release()
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
