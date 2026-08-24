# MO_Changes
"""Single source of truth for the gesture classes.

The order here is the order the model was trained on. The configuration loader
and the detector both check against it, so a checkpoint trained on a different
order is rejected instead of quietly reading an open palm as a fingertip.
"""

from __future__ import annotations

from enum import Enum


class GestureName(str, Enum):
    """Gesture names understood by the runtime."""

    OPEN_PALM_START = "open_palm_start"
    CLOSED_PALM_STOP = "closed_palm_stop"
    POINTING_FINGER = "pointing_finger"
    INDEX_FINGERTIP = "index_fingertip"


GESTURE_CLASS_ORDER: tuple[str, ...] = (
    GestureName.OPEN_PALM_START.value,
    GestureName.POINTING_FINGER.value,
    GestureName.INDEX_FINGERTIP.value,
)

CLASS_IDS: dict[str, int] = {name: index for index, name in enumerate(GESTURE_CLASS_ORDER)}


def class_names() -> dict[int, str]:
    """Class index to name mapping in the order used for training."""
    return {index: name for index, name in enumerate(GESTURE_CLASS_ORDER)}


def gesture_by_class_id() -> dict[int, GestureName]:
    return {index: GestureName(name) for index, name in enumerate(GESTURE_CLASS_ORDER)}
