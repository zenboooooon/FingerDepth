'実験で固定して使うカメラ条件、MediaPipeの手指ランドマーク番号などの定数を定義します。'

FINGERTIP_LANDMARK_INDEX = 8
FINGERTIP_LANDMARK_NAME = "INDEX_FINGER_TIP"

# MediaPipe Hand Landmarkerの21点構成を使います。タプルの位置をランドマーク番号と一致させ、
# MediaPipeの番号とずれないようにすることで、呼び出し側が
# MediaPipe本体を読み込まずに特徴量の並びを検証できます。
HAND_LANDMARK_NAMES = (
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
)

# 生徒モデルの初期特徴量には、人差し指の関節から先端までを含めます。
# ランドマーク8を単一画素の疑似ラベル対象として使います。
DEFAULT_FEATURE_LANDMARK_INDICES = (5, 6, 7, 8)
DEFAULT_TARGET_LANDMARK_INDEX = FINGERTIP_LANDMARK_INDEX

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
