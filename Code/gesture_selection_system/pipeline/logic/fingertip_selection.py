"""Fingertip geometry and object selection.

No calibration is required here. The fingertip point and the object boxes are
both in image coordinates, so the whole decision is plain geometry.

The detector of the existing pick and drop system reports bounding boxes and no
segmentation masks, so a fingertip selects an object by sitting inside its box.
Where boxes overlap the smallest one wins, which resolves to the object the
operator is actually on rather than a large box that merely surrounds it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from config import SelectionConfig
from schemas import BoundingBox, DetectedObject

LOGGER = logging.getLogger(__name__)


def bbox_center(box: BoundingBox) -> tuple[float, float]:
    """Center point of the index fingertip box in pixels."""
    return box.center


def point_in_box(point: tuple[float, float], box: BoundingBox) -> bool:
    x, y = float(point[0]), float(point[1])
    return box.x1 <= x <= box.x2 and box.y1 <= y <= box.y2


def center_distance_ratio(point: tuple[float, float], box: BoundingBox) -> float:
    """How far the point sits from the box center, in half box widths.

    Zero is dead center and one is the box edge. Used by the optional stricter
    rule that stops a fingertip near a box corner from selecting it.
    """
    center_x, center_y = box.center
    half_w = max(box.width / 2.0, 1e-6)
    half_h = max(box.height / 2.0, 1e-6)
    return max(abs(float(point[0]) - center_x) / half_w, abs(float(point[1]) - center_y) / half_h)


def point_in_polygon(point: tuple[float, float], polygon: np.ndarray) -> bool:
    """Ray casting test used for the calibrated table workspace."""
    poly = np.asarray(polygon, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
        return False

    x, y = float(point[0]), float(point[1])
    inside = False
    count = poly.shape[0]
    previous = count - 1
    for current in range(count):
        x_i, y_i = poly[current]
        x_j, y_j = poly[previous]
        if (y_i > y) != (y_j > y):
            crossing_x = (x_j - x_i) * (y - y_i) / (y_j - y_i) + x_i
            if x < crossing_x:
                inside = not inside
        previous = current
    return inside


@dataclass(frozen=True)
class TouchResult:
    """Which object the fingertip is on in the current frame."""

    touched: DetectedObject | None
    considered: int
    skipped_low_confidence: int
    skipped_off_center: int

    @property
    def object_id(self) -> str | None:
        return self.touched.object_id if self.touched is not None else None


def find_touched_object(
    objects: Iterable[DetectedObject],
    center: tuple[float, float],
    confidence_threshold: float,
    max_center_distance_ratio: float | None = None,
) -> TouchResult:
    """Return the smallest object box the fingertip sits inside."""
    best: DetectedObject | None = None
    considered = 0
    skipped_low_confidence = 0
    skipped_off_center = 0

    for candidate in objects:
        if candidate.confidence < confidence_threshold:
            skipped_low_confidence += 1
            continue

        considered += 1
        if not point_in_box(center, candidate.box):
            continue
        if (
            max_center_distance_ratio is not None
            and center_distance_ratio(center, candidate.box) > max_center_distance_ratio
        ):
            skipped_off_center += 1
            continue
        if best is None or candidate.box.area < best.box.area:
            best = candidate

    return TouchResult(
        touched=best,
        considered=considered,
        skipped_low_confidence=skipped_low_confidence,
        skipped_off_center=skipped_off_center,
    )


@dataclass(frozen=True)
class HoldState:
    """Result of one hold step."""

    key: str | None
    held_s: float
    confirmed_key: str | None
    just_confirmed: bool


class HoldTimer:
    """Confirms a key once it has been held without interruption long enough.

    Used for the object selection hold and for the place point hold. Working in
    seconds keeps the operator experience the same whatever frame rate the
    machine reaches.
    """

    def __init__(self, required_seconds: float) -> None:
        if required_seconds <= 0.0:
            raise ValueError("required_seconds must be positive")
        self._required = required_seconds
        self._key: str | None = None
        self._since: float | None = None
        self._confirmed: str | None = None

    @property
    def required_seconds(self) -> float:
        return self._required

    @property
    def confirmed_key(self) -> str | None:
        return self._confirmed

    def reset(self) -> None:
        self._key = None
        self._since = None
        self._confirmed = None

    def update(self, key: str | None, now: float) -> HoldState:
        if key is None:
            self._key = None
            self._since = None
            return HoldState(None, 0.0, self._confirmed, False)

        if key != self._key:
            self._key = key
            self._since = now

        held = now - self._since if self._since is not None else 0.0
        just_confirmed = False
        if held >= self._required and self._confirmed != key:
            self._confirmed = key
            just_confirmed = True

        return HoldState(key, held, self._confirmed, just_confirmed)


@dataclass(frozen=True)
class ObjectSelectionDecision:
    """Outcome of one object selection step."""

    candidate_id: str | None
    selected: DetectedObject | None
    held_s: float
    just_selected: bool
    notes: tuple[str, ...] = ()

    @property
    def selected_id(self) -> str | None:
        return self.selected.object_id if self.selected is not None else None


class ObjectSelector:
    """Selects the object the fingertip keeps pointing at.

    Image coordinates only. The fingertip pixel comes from the YOLO
    index_fingertip box and the object boxes come from the existing detector, so
    no camera calibration takes part in this decision.
    """

    def __init__(self, selection: SelectionConfig, object_confidence: float, hold_seconds: float):
        self._selection = selection
        self._object_confidence = object_confidence
        self._timer = HoldTimer(hold_seconds)
        self._selected: DetectedObject | None = None
        self._held_s = 0.0

    @property
    def selected(self) -> DetectedObject | None:
        return self._selected

    @property
    def held_s(self) -> float:
        return self._held_s

    def reset(self) -> None:
        self._timer.reset()
        self._selected = None
        self._held_s = 0.0

    def update(
        self,
        center: tuple[float, float] | None,
        objects: Sequence[DetectedObject],
        active: bool,
        now: float,
    ) -> ObjectSelectionDecision:
        notes: list[str] = []

        if not active or center is None:
            self._timer.update(None, now)
            if self._selected is not None and not self._selection.hold_selection_until_mode_off:
                self.reset()
                notes.append("selection_cleared:fingertip_lost")
            return self._decision(None, False, notes)

        touch = find_touched_object(
            objects=objects,
            center=center,
            confidence_threshold=self._object_confidence,
            max_center_distance_ratio=self._selection.max_center_distance_ratio,
        )
        if touch.skipped_low_confidence:
            notes.append(f"objects_below_confidence:{touch.skipped_low_confidence}")
        if touch.skipped_off_center:
            notes.append(f"objects_too_far_from_center:{touch.skipped_off_center}")
        if touch.touched is None and touch.considered:
            notes.append("fingertip_not_on_any_object")

        state = self._timer.update(touch.object_id, now)
        just_selected = False

        if state.just_confirmed and touch.touched is not None:
            self._selected = touch.touched
            self._held_s = state.held_s
            just_selected = True
            notes.append("object_selected")
            LOGGER.info(
                "object_selected id=%s class=%s held_s=%.1f",
                touch.touched.object_id,
                touch.touched.class_name,
                state.held_s,
            )
        elif self._selected is not None:
            if touch.object_id == self._selected.object_id and touch.touched is not None:
                self._selected = touch.touched
                self._held_s = state.held_s
            elif not self._selection.hold_selection_until_mode_off:
                self.reset()
                notes.append("selection_cleared:fingertip_left_object")

        return self._decision(touch.object_id, just_selected, notes)

    def _decision(
        self, candidate_id: str | None, just_selected: bool, notes: Sequence[str]
    ) -> ObjectSelectionDecision:
        return ObjectSelectionDecision(
            candidate_id=candidate_id,
            selected=self._selected,
            held_s=round(self._held_s, 2),
            just_selected=just_selected,
            notes=tuple(notes),
        )


def place_grid_key(center: tuple[float, float], cell_px: int = 20) -> str:
    """Quantize a pixel to a grid cell so small hand tremor still counts as stable."""
    if cell_px < 1:
        raise ValueError("cell_px must be at least 1")
    return f"place:{int(center[0]) // cell_px},{int(center[1]) // cell_px}"
