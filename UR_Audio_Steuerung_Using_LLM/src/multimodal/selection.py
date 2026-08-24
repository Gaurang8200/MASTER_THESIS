# MO_Changes
"""Resolve a spoken pointing reference against overview detections."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


POINTING_PATTERNS = (
    r"\bthis\s+(?:object|one|item|thing|cylinder|box|marker)\b",
    r"\bthat\s+(?:object|one|item|thing|cylinder|box|marker)\b",
    r"^(?:please\s+)?pick\s*up(?:\s+please)?[.!]?$",
    r"\bpick\s+(?:this|that|it)\s+up\b",
    r"\bgrab\s+(?:this|that|it)\b",
    r"\bdies(?:es|en|e)?\s+(?:objekt|ding|zylinder|quader|marker)\b",
    r"\bdas\s+(?:objekt|ding|da)\b",
    r"\bnimm\s+(?:dies|das|es)\b",
    r"\bheb\s+(?:dies|das|es)\s+auf\b",
)

AFFIRMATIVE_PATTERNS = (
    r"\byes\b",
    r"\byeah\b",
    r"\byep\b",
    r"\bcorrect\b",
    r"\bconfirm\b",
    r"\bja\b",
    r"\bgenau\b",
    r"\bbestätig",
)

NEGATIVE_PATTERNS = (
    r"\bno\b",
    r"\bnope\b",
    r"\bcancel\b",
    r"\bstop\b",
    r"\bnein\b",
    r"\babbrechen\b",
)

DROP_HERE_PATTERNS = (
    r"\bdrop\s+(?:it\s+)?here\b",
    r"\bplace\s+(?:it\s+)?here\b",
    r"\bput\s+(?:it\s+)?here\b",
    r"\bhier\s+ab(?:legen|stellen)\b",
    r"\bleg\s+(?:es\s+)?hier\b",
    r"\bstell\s+(?:es\s+)?hier\b",
)

ZONE_PATTERNS = {
    "Zone_1": r"\bzone\s*(?:one|1|eins)\b",
    "Zone_2": r"\bzone\s*(?:two|2|zwei)\b",
    "Zone_3": r"\bzone\s*(?:three|3|drei)\b",
}


@dataclass(frozen=True)
class MultimodalResolution:
    required: bool
    accepted: bool
    reason: str
    selected_object: dict[str, object] | None = None
    object_index: int | None = None


def is_pointing_reference(text: str, extracted_info: dict[str, object]) -> bool:
    if extracted_info.get("selection_mode") == "gesture":
        return True
    normalized = " ".join(text.lower().split())
    return any(re.search(pattern, normalized) for pattern in POINTING_PATTERNS)


def is_affirmative(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(re.search(pattern, normalized) for pattern in AFFIRMATIVE_PATTERNS)


def is_negative(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(re.search(pattern, normalized) for pattern in NEGATIVE_PATTERNS)


def is_drop_here(text: str) -> bool:
    normalized = " ".join(text.lower().split())
    return any(re.search(pattern, normalized) for pattern in DROP_HERE_PATTERNS)


def extract_zone(text: str) -> str | None:
    normalized = " ".join(text.lower().split())
    for zone_name, pattern in ZONE_PATTERNS.items():
        if re.search(pattern, normalized):
            return zone_name
    return None


def _frame_size_from_detection(detection_data: dict[str, object]) -> tuple[int, int] | None:
    metadata = detection_data.get("metadata")
    if not isinstance(metadata, dict):
        return None
    width = metadata.get("image_width")
    height = metadata.get("image_height")
    if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
        return width, height
    image_path = metadata.get("image_path")
    if not isinstance(image_path, str) or not Path(image_path).is_file():
        return None
    try:
        import cv2

        image = cv2.imread(image_path)
    except ImportError:
        return None
    if image is None:
        return None
    return int(image.shape[1]), int(image.shape[0])


def _normalized_box(box: Sequence[float], width: int, height: int) -> tuple[float, ...]:
    return (
        float(box[0]) / width,
        float(box[1]) / height,
        float(box[2]) / width,
        float(box[3]) / height,
    )


def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _center_distance(first: Sequence[float], second: Sequence[float]) -> float:
    first_x = (first[0] + first[2]) / 2.0
    first_y = (first[1] + first[3]) / 2.0
    second_x = (second[0] + second[2]) / 2.0
    second_y = (second[1] + second[3]) / 2.0
    return math.hypot(first_x - second_x, first_y - second_y)


def resolve_multimodal_selection(
    text: str,
    extracted_info: dict[str, object],
    gesture_result: dict[str, object],
    detection_data: dict[str, object],
) -> MultimodalResolution:
    if not is_pointing_reference(text, extracted_info):
        return MultimodalResolution(False, True, "explicit_speech_selection")

    if not gesture_result.get("safe_to_use") or gesture_result.get("status") != "selected":
        return MultimodalResolution(
            True,
            False,
            str(gesture_result.get("reason", "gesture_result_missing")),
        )

    selected_live = gesture_result.get("selected_object")
    if not isinstance(selected_live, dict):
        return MultimodalResolution(True, False, "gesture_result_invalid")

    live_class = str(selected_live.get("class_name", "")).lower()
    objects = detection_data.get("objects")
    if not isinstance(objects, list):
        return MultimodalResolution(True, False, "object_not_in_detection_list")
    candidates = [
        item
        for item in objects
        if isinstance(item, dict) and str(item.get("class_name", "")).lower() == live_class
    ]
    if not candidates:
        return MultimodalResolution(True, False, "object_not_in_detection_list")
    overview_size = _frame_size_from_detection(detection_data)
    live_width = gesture_result.get("frame_width")
    live_height = gesture_result.get("frame_height")
    live_box = selected_live.get("bbox")
    if (
        overview_size is None
        or not isinstance(live_width, int)
        or not isinstance(live_height, int)
        or not isinstance(live_box, list)
        or len(live_box) != 4
    ):
        return MultimodalResolution(True, False, "overview_geometry_missing")

    normalized_live = _normalized_box(live_box, live_width, live_height)
    ranked: list[tuple[float, float, float, dict[str, object]]] = []
    overview_width, overview_height = overview_size
    for item in candidates:
        box = item.get("bbox")
        if not isinstance(box, list) or len(box) != 4:
            continue
        normalized_overview = _normalized_box(box, overview_width, overview_height)
        overlap = _box_iou(normalized_live, normalized_overview)
        distance = _center_distance(normalized_live, normalized_overview)
        score = overlap + max(0.0, 1.0 - distance) * 0.25
        ranked.append((score, overlap, distance, item))

    ranked.sort(key=lambda entry: entry[0], reverse=True)
    if not ranked:
        return MultimodalResolution(True, False, "overview_geometry_missing")
    best_score, best_overlap, best_distance, chosen = ranked[0]
    if best_overlap < 0.10 and best_distance > 0.12:
        return MultimodalResolution(True, False, "object_not_in_detection_list")
    if len(ranked) > 1 and best_score - ranked[1][0] < 0.08:
        return MultimodalResolution(True, False, "overview_match_ambiguous")

    class_objects = [
        item
        for item in objects
        if isinstance(item, dict) and str(item.get("class_name", "")).lower() == live_class
    ]
    return MultimodalResolution(
        True,
        True,
        "gesture_object_matched",
        chosen,
        class_objects.index(chosen),
    )
