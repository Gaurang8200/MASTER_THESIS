"""Single source of truth for the gesture classes.

Training and the runtime pipeline both import this module, so the class order in
the dataset can never drift away from the class order the detector assumes.
"""

from __future__ import annotations

from enum import Enum


class GestureName(str, Enum):
    """The four gesture classes the YOLO11s Detect model is trained on."""

    OPEN_PALM_START = "open_palm_start"
    CLOSED_PALM_STOP = "closed_palm_stop"
    POINTING_FINGER = "pointing_finger"
    INDEX_FINGERTIP = "index_fingertip"


GESTURE_CLASS_ORDER: tuple[str, ...] = (
    GestureName.OPEN_PALM_START.value,
    GestureName.CLOSED_PALM_STOP.value,
    GestureName.POINTING_FINGER.value,
    GestureName.INDEX_FINGERTIP.value,
)

CLASS_IDS: dict[str, int] = {name: index for index, name in enumerate(GESTURE_CLASS_ORDER)}


def class_names() -> dict[int, str]:
    """Class index to name mapping in the order used for training."""
    return {index: name for index, name in enumerate(GESTURE_CLASS_ORDER)}


def gesture_by_class_id() -> dict[int, GestureName]:
    return {index: GestureName(name) for index, name in enumerate(GESTURE_CLASS_ORDER)}
