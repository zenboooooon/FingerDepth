"""Fixed experiment and MediaPipe hand-landmark constants."""

FINGERTIP_LANDMARK_INDEX = 8
FINGERTIP_LANDMARK_NAME = "INDEX_FINGER_TIP"

# MediaPipe Hand Landmarker uses this stable 21-landmark topology. Keep the
# tuple index aligned with MediaPipe's landmark index so callers can validate
# serialized feature order without importing MediaPipe itself.
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

# The initial student-model feature set covers the complete index-finger chain.
# Landmark 8 remains the single-pixel pseudo-label target for backward
# compatibility with phases 1 and 2.
DEFAULT_FEATURE_LANDMARK_INDICES = (5, 6, 7, 8)
DEFAULT_TARGET_LANDMARK_INDEX = FINGERTIP_LANDMARK_INDEX

# Pin both the upstream implementation and the checkpoint selected by hubconf.py.
METRIC3D_HUB_REPO = "YvanYin/Metric3D:eb5b6fac0dc155e4e52f576e304fbf11655ff339"
METRIC3D_HUB_MODEL = "metric3d_vit_small"
METRIC3D_CHECKPOINT_URL = (
    "https://huggingface.co/JUGGHM/Metric3D/resolve/main/metric_depth_vit_small_800k.pth"
)
METRIC3D_INPUT_HEIGHT = 616
METRIC3D_INPUT_WIDTH = 1064
METRIC3D_CANONICAL_FOCAL_PX = 1000.0
METRIC3D_MAX_DEPTH_M = 300.0

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
