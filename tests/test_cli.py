from pathlib import Path

import pytest

from fingertip_depth import cli


def test_phase1_help_documents_supported_device_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as exit_info:
        parser.parse_args(["phase1", "--help"])

    assert exit_info.value.code == 0
    assert "auto or cuda:0" in capsys.readouterr().out


def test_phase1_video_saves_depth_frames_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake video")
    calls: list[dict[str, object]] = []

    def fake_run_video(**kwargs: object) -> dict[str, bool]:
        calls.append(kwargs)
        return {"ok": True}

    monkeypatch.setattr(cli, "run_video", fake_run_video)

    exit_code = cli.main(
        [
            "phase1",
            "--input",
            str(input_path),
            "--output-dir",
            str(tmp_path / "output"),
            "--fx-px",
            "500",
            "--device",
            "cuda:0",
        ]
    )

    assert exit_code == 0
    assert len(calls) == 1
    assert calls[0]["phase"] == 1
    assert calls[0]["save_depth_frames"] is True
    assert calls[0]["hand_model_path"] is None


def test_input_kind_auto_detects_heic_image() -> None:
    assert cli._input_kind(Path("frame.HEIC"), "auto") == "image"
