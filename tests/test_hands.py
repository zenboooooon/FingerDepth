from types import SimpleNamespace

import pytest

from fingertip_depth.constants import (
    DEFAULT_FEATURE_LANDMARK_INDICES,
    DEFAULT_TARGET_LANDMARK_INDEX,
    HAND_LANDMARK_NAMES,
)
from fingertip_depth.hands import HandLandmarker, parse_hand_landmark_selection


def test_extracts_only_index_finger_tip() -> None:
    landmarks = [SimpleNamespace(x=0.1, y=0.2) for _ in range(21)]
    landmarks[8] = SimpleNamespace(x=0.5, y=0.25)
    category = SimpleNamespace(category_name="Right", display_name="", score=0.9)
    result = SimpleNamespace(hand_landmarks=[landmarks], handedness=[[category]])
    detections = HandLandmarker._extract(result, width=640, height=480)
    assert len(detections) == 1
    detection = detections[0]
    assert detection.landmark_index == 8
    assert (detection.u_px, detection.v_px) == (320, 120)
    assert detection.handedness == "Right"
    assert detection.handedness_score == pytest.approx(0.9)


def test_no_hand_is_an_empty_result() -> None:
    result = SimpleNamespace(hand_landmarks=[], handedness=[])
    assert HandLandmarker._extract(result, width=640, height=480) == []


def test_official_landmark_names_and_index_finger_defaults() -> None:
    assert len(HAND_LANDMARK_NAMES) == 21
    assert HAND_LANDMARK_NAMES[0] == "WRIST"
    assert HAND_LANDMARK_NAMES[5:9] == (
        "INDEX_FINGER_MCP",
        "INDEX_FINGER_PIP",
        "INDEX_FINGER_DIP",
        "INDEX_FINGER_TIP",
    )
    assert HAND_LANDMARK_NAMES[20] == "PINKY_TIP"
    assert DEFAULT_FEATURE_LANDMARK_INDICES == (5, 6, 7, 8)
    assert DEFAULT_TARGET_LANDMARK_INDEX == 8


def test_parses_ordered_names_indices_and_all() -> None:
    assert parse_hand_landmark_selection(
        "INDEX_FINGER_TIP, 5, index_finger_pip,7"
    ) == (8, 5, 6, 7)
    assert parse_hand_landmark_selection("all") == tuple(range(21))


@pytest.mark.parametrize(
    "value",
    ["", "5,,8", "-1", "21", "UNKNOWN", "5,INDEX_FINGER_MCP", "all,8"],
)
def test_rejects_invalid_landmark_selections(value: str) -> None:
    with pytest.raises(ValueError):
        parse_hand_landmark_selection(value)


def test_extract_hands_keeps_all_landmarks_and_marks_out_of_frame() -> None:
    landmarks = [
        SimpleNamespace(x=0.1, y=0.2, z=-index / 100.0) for index in range(21)
    ]
    landmarks[8] = SimpleNamespace(x=0.5, y=0.25, z=-0.125)
    landmarks[7] = SimpleNamespace(x=1.01, y=0.25, z=-0.1)
    category = SimpleNamespace(category_name="Right", display_name="", score=0.9)
    result = SimpleNamespace(hand_landmarks=[landmarks], handedness=[[category]])

    detections = HandLandmarker._extract_hands(result, width=640, height=480)

    assert len(detections) == 1
    hand = detections[0]
    assert hand.hand_index == 0
    assert hand.handedness == "Right"
    assert hand.handedness_score == pytest.approx(0.9)
    assert len(hand.landmarks) == 21
    tip = hand.landmark(8)
    assert tip is not None
    assert tip.landmark_name == "INDEX_FINGER_TIP"
    assert tip.z_mediapipe_relative == pytest.approx(-0.125)
    assert (tip.u_px, tip.v_px, tip.in_frame) == (320, 120, True)
    dip = hand.landmark(7)
    assert dip is not None
    assert (dip.u_px, dip.v_px, dip.in_frame) == (None, None, False)
    assert hand.as_dict()["landmarks"][8] == tip.as_dict()


def test_video_timestamps_are_strictly_increasing_even_above_1000_fps() -> None:
    previous = None
    timestamps = []
    for frame_index in range(5):
        current = HandLandmarker.next_video_timestamp_ms(frame_index, 2000.0, previous)
        timestamps.append(current)
        previous = current
    assert timestamps == [0, 1, 2, 3, 4]
