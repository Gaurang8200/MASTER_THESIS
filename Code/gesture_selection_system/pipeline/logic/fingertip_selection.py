"""Fingertip geometry and object selection.

No calibration is required for fingertip to mask object selection because the
fingertip point and the object masks are both in image coordinates. Everything
in this module is pure geometry over numpy arrays and stability counting.

A fingertip only selects an object when the probe circle actually overlaps the
segmentation mask, so a finger that is near an object but outside its mask never
selects it.
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


def circle_mask_overlap(mask: np.ndarray, center: tuple[float, float], radius: int) -> int:
    """Count mask pixels covered by the fingertip probe.

    A radius of zero tests the single center pixel, which is the strictest
    reading of the selection rule.
    """
    if mask is None or mask.ndim < 2:
        return 0

    height, width = mask.shape[:2]
    center_x, center_y = float(center[0]), float(center[1])

    if radius <= 0:
        x_index = int(round(center_x))
        y_index = int(round(center_y))
        if 0 <= x_index < width and 0 <= y_index < height:
            return int(bool(mask[y_index, x_index]))
        return 0

    x_start = max(int(np.floor(center_x - radius)), 0)
    x_end = min(int(np.ceil(center_x + radius)) + 1, width)
    y_start = max(int(np.floor(center_y - radius)), 0)
    y_end = min(int(np.ceil(center_y + radius)) + 1, height)
    if x_start >= x_end or y_start >= y_end:
        return 0

    ys, xs = np.ogrid[y_start:y_end, x_start:x_end]
    probe = (xs - center_x) ** 2 + (ys - center_y) ** 2 <= float(radius) ** 2
    window = mask[y_start:y_end, x_start:x_end].astype(bool, copy=False)
    return int(np.count_nonzero(np.logical_and(window, probe)))


@dataclass(frozen=True)
class TouchResult:
    """Which object the fingertip is touching in the current frame."""

    touched: DetectedObject | None
    overlap_px: int
    considered: int
    skipped_low_confidence: int
    skipped_missing_mask: int

    @property
    def object_id(self) -> str | None:
        return self.touched.object_id if self.touched is not None else None


def find_touched_object(
    objects: Iterable[DetectedObject],
    center: tuple[float, float],
    frame_shape: tuple[int, int],
    radius: int,
    min_overlap_px: int,
    confidence_threshold: float,
) -> TouchResult:
    """Return the object whose mask the fingertip probe overlaps most.

    Objects without a usable mask are skipped rather than approximated by their
    bounding box, because a box hit would select an object the finger only
    hovers next to.
    """
    best: DetectedObject | None = None
    best_overlap = 0
    considered = 0
    skipped_low_confidence = 0
    skipped_missing_mask = 0

    for candidate in objects:
        if candidate.confidence < confidence_threshold:
            skipped_low_confidence += 1
            continue
        if not candidate.has_mask_for(frame_shape):
            skipped_missing_mask += 1
            continue

        considered += 1
        overlap = circle_mask_overlap(candidate.mask, center, radius)
        if overlap < min_overlap_px:
            continue
        if overlap > best_overlap or (
            overlap == best_overlap
            and best is not None
            and candidate.confidence > best.confidence
        ):
            best = candidate
            best_overlap = overlap

    return TouchResult(
        touched=best,
        overlap_px=best_overlap if best is not None else 0,
        considered=considered,
        skipped_low_confidence=skipped_low_confidence,
        skipped_missing_mask=skipped_missing_mask,
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
    overlap_px: int
    held_s: float
    just_selected: bool
    notes: tuple[str, ...] = ()

    @property
    def selected_id(self) -> str | None:
        return self.selected.object_id if self.selected is not None else None


class ObjectSelector:
    """Selects the object the fingertip keeps touching.

    Image coordinates only. The fingertip pixel comes from the YOLO
    index_fingertip box and the masks come from the object segmentation system,
    so no camera calibration takes part in this decision.
    """

    def __init__(self, selection: SelectionConfig, object_confidence: float, hold_seconds: float):
        self._selection = selection
        self._object_confidence = object_confidence
        self._timer = HoldTimer(hold_seconds)
        self._selected: DetectedObject | None = None
        self._overlap_px = 0
        self._held_s = 0.0

    @property
    def selected(self) -> DetectedObject | None:
        return self._selected

    @property
    def overlap_px(self) -> int:
        return self._overlap_px

    @property
    def held_s(self) -> float:
        return self._held_s

    def reset(self) -> None:
        self._timer.reset()
        self._selected = None
        self._overlap_px = 0
        self._held_s = 0.0

    def update(
        self,
        center: tuple[float, float] | None,
        objects: Sequence[DetectedObject],
        frame_shape: tuple[int, int],
        active: bool,
        now: float,
    ) -> ObjectSelectionDecision:
        notes: list[str] = []

        if not active or center is None:
            self._timer.update(None, now)
            if self._selected is not None and not self._selection.hold_selection_until_mode_off:
                self.reset()
                notes.append("selection_cleared:fingertip_lost")
            return self._decision(None, 0, False, notes)

        touch = find_touched_object(
            objects=objects,
            center=center,
            frame_shape=frame_shape,
            radius=self._selection.fingertip_radius_px,
            min_overlap_px=self._selection.min_mask_overlap_px,
            confidence_threshold=self._object_confidence,
        )
        if touch.skipped_low_confidence:
            notes.append(f"objects_below_confidence:{touch.skipped_low_confidence}")
        if touch.skipped_missing_mask:
            notes.append(f"objects_without_mask:{touch.skipped_missing_mask}")
        if touch.touched is None and touch.considered:
            notes.append("fingertip_not_inside_any_mask")

        state = self._timer.update(touch.object_id, now)
        just_selected = False

        if state.just_confirmed and touch.touched is not None:
            self._selected = touch.touched
            self._overlap_px = touch.overlap_px
            self._held_s = state.held_s
            just_selected = True
            notes.append("object_selected")
            LOGGER.info(
                "object_selected id=%s class=%s overlap_px=%d held_s=%.1f",
                touch.touched.object_id,
                touch.touched.class_name,
                touch.overlap_px,
                state.held_s,
            )
        elif self._selected is not None:
            if touch.object_id == self._selected.object_id and touch.touched is not None:
                self._selected = touch.touched
                self._overlap_px = touch.overlap_px
                self._held_s = state.held_s
            elif not self._selection.hold_selection_until_mode_off:
                self.reset()
                notes.append("selection_cleared:fingertip_left_object")

        return self._decision(touch.object_id, touch.overlap_px, just_selected, notes)

    def _decision(
        self,
        candidate_id: str | None,
        overlap_px: int,
        just_selected: bool,
        notes: Sequence[str],
    ) -> ObjectSelectionDecision:
        return ObjectSelectionDecision(
            candidate_id=candidate_id,
            selected=self._selected,
            overlap_px=self._overlap_px if self._selected is not None else overlap_px,
            held_s=round(self._held_s, 2),
            just_selected=just_selected,
            notes=tuple(notes),
        )


def place_grid_key(center: tuple[float, float], cell_px: int = 20) -> str:
    """Quantize a pixel to a grid cell so small hand tremor still counts as stable."""
    if cell_px < 1:
        raise ValueError("cell_px must be at least 1")
    return f"place:{int(center[0]) // cell_px},{int(center[1]) // cell_px}"


def polygon_bounds(polygon: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    """Axis aligned bounds of the workspace polygon, useful for drawing and tests."""
    poly = np.asarray(polygon, dtype=np.float64)
    return (
        float(poly[:, 0].min()),
        float(poly[:, 1].min()),
        float(poly[:, 0].max()),
        float(poly[:, 1].max()),
    )
