# Training videos

This directory is the only supported drop location for the incremental student-training
pipeline. Original videos are inputs: do not edit or transcode them in place after a run.

## Directories

- `train/`: put every new video intended for training here.
- `validation/`: fixed, curated sequence-held-out validation videos only.

Adding a training video must not change the validation set. Do not copy a clip, a re-encode,
or another segment from the same capture session across `train/` and `validation/`; the
pipeline rejects exact source/frame overlap, but cannot prove that every near-duplicate came
from the same session.

Supported filename extensions are `.mov`, `.mp4`, and `.m4v` (case-insensitive). Use a stable,
descriptive filename such as `iphone15_session_20260926_01.mov`; the pipeline combines its
normalized slug with a content-hash suffix to create a collision-resistant sequence ID. A
supported extension does not guarantee acceptance: the video must decode completely with a
positive frame rate and stable dimensions.

## Run

From the repository root:

```bash
cp /path/to/new_capture.MOV data/training_videos/train/new_capture.MOV
uv run --locked fingertip-train status
uv run --locked fingertip-train run --dry-run
uv run --locked fingertip-train run
```

Unchanged sources and verified intermediate artifacts are reused. Use `--force-train` only
when a new training run is required for an otherwise identical dataset and configuration.

### Interrupted label generation

Run the same command again after an interruption:

```bash
uv run --locked fingertip-train run
```

The pipeline retains a private per-video workspace. It reuses a fully verified preparation
manifest and resumes Depth Pro after the last hash-chained, fsynced checkpoint. Set
`teacher.checkpoint_interval_frames` in `configs/training_pipeline.toml` to choose the number
of frame outcomes per checkpoint (the default is 100); the final partial batch is also saved.
A torn unterminated journal tail is discarded and recomputed; committed corruption or any
mismatch in the video, focal metadata, model, configuration, or relevant implementation fails
closed. The immutable video cache is published only after every declared artifact passes
validation.

## Camera metadata

The default in `configs/training_pipeline.toml` is **36 mm-equivalent**, matching the existing
iPhone video experiments. It is an uncalibrated approximation, not a measured focal length.
For another camera or zoom setting, add an explicit project-relative override before running:

```toml
[video_overrides."data/training_videos/train/new_capture.MOV"]
focal_35mm_mm = 26.0
```

Changing focal metadata changes the processing identity and produces a separate verified
cache. Never rename or replace a previously processed file to conceal changed bytes; add it
under a new sequence ID instead.

The video files and generated artifacts are intentionally excluded from Git. The README,
configuration, manifests, hashes, and code provide provenance, but the raw bytes still need a
separate backed-up storage location when runs must be reproduced on another machine.
