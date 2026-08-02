"""OpenCV overlays for the gesture demos.

Drawing is kept out of the pipeline so that the interaction logic stays
headless and testable.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from schemas import DetectedObject, GestureFrame, GestureName, PipelineOutput, SelectionMode

FONT = cv2.FONT_HERSHEY_SIMPLEX

GESTURE_COLORS: dict[GestureName, tuple[int, int, int]] = {
    GestureName.OPEN_PALM_START: (60, 200, 60),
    GestureName.CLOSED_PALM_STOP: (40, 40, 220),
    GestureName.POINTING_FINGER: (230, 170, 40),
    GestureName.INDEX_FINGERTIP: (240, 240, 60),
}
OBJECT_COLOR = (180, 180, 180)
SELECTED_COLOR = (0, 215, 255)
WORKSPACE_COLOR = (120, 120, 255)
PLACE_COLOR = (255, 120, 200)


def draw_gestures(frame: np.ndarray, gesture_frame: GestureFrame) -> np.ndarray:
    for detection in gesture_frame.detections:
        color = GESTURE_COLORS.get(detection.gesture, (255, 255, 255))
        x1, y1, x2, y2 = detection.box.as_int_tuple()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.gesture.value} {detection.confidence:.2f}"
        _label(frame, label, (x1, y1), color)
    return frame


def draw_objects(
    frame: np.ndarray,
    objects: Sequence[DetectedObject],
    selected_id: str | None,
    show_masks: bool,
    mask_alpha: float,
) -> np.ndarray:
    if show_masks and objects:
        tint = frame.copy()
        for item in objects:
            if item.mask is None or item.mask.shape[:2] != frame.shape[:2]:
                continue
            color = SELECTED_COLOR if item.object_id == selected_id else OBJECT_COLOR
            tint[item.mask.astype(bool)] = color
        cv2.addWeighted(tint, mask_alpha, frame, 1.0 - mask_alpha, 0.0, frame)

    for item in objects:
        selected = item.object_id == selected_id
        color = SELECTED_COLOR if selected else OBJECT_COLOR
        x1, y1, x2, y2 = item.box.as_int_tuple()
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if selected else 1)
        _label(frame, f"{item.object_id} {item.class_name} {item.confidence:.2f}", (x1, y1), color)
    return frame


def draw_workspace(frame: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [points], isClosed=True, color=WORKSPACE_COLOR, thickness=1)
    return frame


def draw_fingertip(
    frame: np.ndarray, center: tuple[float, float], probe_radius: int, marker_radius: int, active: bool
) -> np.ndarray:
    point = (int(round(center[0])), int(round(center[1])))
    color = GESTURE_COLORS[GestureName.INDEX_FINGERTIP] if active else (150, 150, 150)
    cv2.circle(frame, point, marker_radius, color, 2)
    if probe_radius > 0:
        cv2.circle(frame, point, probe_radius, color, 1)
    cv2.drawMarker(frame, point, color, cv2.MARKER_CROSS, 12, 1)
    return frame


def draw_place_point(frame: np.ndarray, center: tuple[float, float]) -> np.ndarray:
    point = (int(round(center[0])), int(round(center[1])))
    cv2.drawMarker(frame, point, PLACE_COLOR, cv2.MARKER_TILTED_CROSS, 22, 2)
    cv2.circle(frame, point, 14, PLACE_COLOR, 1)
    return frame


def draw_hud(frame: np.ndarray, lines: Sequence[str], mode: SelectionMode) -> np.ndarray:
    height = 22 * len(lines) + 14
    cv2.rectangle(frame, (0, 0), (430, height), (25, 25, 25), -1)
    mode_color = (60, 220, 60) if mode is SelectionMode.ON else (60, 60, 220)
    cv2.rectangle(frame, (0, 0), (430, height), mode_color, 2)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (10, 24 + index * 22), FONT, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    return frame


def hud_lines(result: PipelineOutput, extra: Sequence[str] = ()) -> list[str]:
    lines = [
        f"mode: {result.selection_mode.value.upper()}"
        + (f"  ({result.mode_transition.value})" if result.mode_transition else ""),
        f"latency: {result.latency.total_ms:.0f} ms"
        + ("" if result.latency.within_budget else "  OVER BUDGET"),
    ]
    if result.selected_object is not None:
        selected = result.selected_object
        lines.append(f"selected: {selected.object_id} {selected.class_name}")
    elif result.candidate_object_id is not None:
        lines.append(f"candidate: {result.candidate_object_id}")
    else:
        lines.append("selected: none")
    if result.place_point is not None:
        pose = result.place_point.pose
        lines.append(f"place: x={pose.x_m:.3f} y={pose.y_m:.3f} m")
    if result.degraded:
        lines.append("degraded: robot execution blocked")
    lines.extend(extra)
    return lines


def render_pipeline_frame(
    frame: np.ndarray,
    result: PipelineOutput,
    objects: Sequence[DetectedObject],
    gesture_frame: GestureFrame,
    polygon: np.ndarray,
    show_masks: bool,
    mask_alpha: float,
    show_workspace: bool,
    show_hud: bool,
    marker_radius: int,
    extra_hud: Sequence[str] = (),
) -> np.ndarray:
    """Compose the full overlay for one processed frame."""
    canvas = frame.copy()
    selected_id = result.selected_object.object_id if result.selected_object else None
    draw_objects(canvas, objects, selected_id, show_masks, mask_alpha)
    if show_workspace:
        draw_workspace(canvas, polygon)
    draw_gestures(canvas, gesture_frame)
    if result.fingertip is not None:
        draw_fingertip(
            canvas,
            (result.fingertip.center_px.x, result.fingertip.center_px.y),
            result.fingertip.probe_radius_px,
            marker_radius,
            result.selection_mode is SelectionMode.ON,
        )
    if result.place_point is not None:
        draw_place_point(canvas, (result.place_point.pixel.x, result.place_point.pixel.y))
    if show_hud:
        draw_hud(canvas, hud_lines(result, extra_hud), result.selection_mode)
    return canvas


def _label(frame: np.ndarray, text: str, anchor: tuple[int, int], color: tuple[int, int, int]) -> None:
    x, y = anchor
    y = max(y, 18)
    (text_w, text_h), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    cv2.rectangle(frame, (x, y - text_h - 6), (x + text_w + 6, y), color, -1)
    cv2.putText(frame, text, (x + 3, y - 4), FONT, 0.5, (15, 15, 15), 1, cv2.LINE_AA)
