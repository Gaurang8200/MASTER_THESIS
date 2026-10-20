from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ObjectOrientation:
    angle_degrees: float


def _segment_object(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        raise ValueError("object crop is empty")
    saturation = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
    saturation = cv2.GaussianBlur(saturation, (5, 5), 0)
    _, mask = cv2.threshold(
        saturation,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("object contour is unavailable")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < crop.shape[0] * crop.shape[1] * 0.05:
        raise ValueError("object contour is too small")
    object_mask = np.zeros_like(mask)
    cv2.drawContours(object_mask, [contour], -1, 255, cv2.FILLED)
    return object_mask


def _geometry_from_mask(
    object_mask: np.ndarray,
) -> float:
    point_y, point_x = np.nonzero(object_mask)
    points = np.column_stack((point_x, point_y)).astype(np.float64)
    _, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean=None)
    if eigenvalues[1, 0] <= 0.0:
        raise ValueError("object PCA has no minor axis")
    if float(eigenvalues[0, 0] / eigenvalues[1, 0]) < 1.5:
        raise ValueError("object is not elongated enough for a stable 2D angle")
    axis_x, axis_y = eigenvectors[0]
    if axis_x < 0.0:
        axis_x, axis_y = -axis_x, -axis_y
    return math.degrees(math.atan2(-axis_y, axis_x))


def estimate_object_angle(crop: np.ndarray) -> float:
    return _geometry_from_mask(_segment_object(crop))


def measure_object_orientation(
    frame: np.ndarray,
    box: Sequence[float],
) -> ObjectOrientation:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("object orientation requires a BGR image")
    if len(box) != 4:
        raise ValueError("object orientation requires four box coordinates")
    height, width = frame.shape[:2]
    x1 = max(0, int(math.floor(float(box[0]))))
    y1 = max(0, int(math.floor(float(box[1]))))
    x2 = min(width, int(math.ceil(float(box[2]))))
    y2 = min(height, int(math.ceil(float(box[3]))))
    if x1 >= x2 or y1 >= y2:
        raise ValueError("object box is outside the image")
    crop = frame[y1:y2, x1:x2]
    return ObjectOrientation(_geometry_from_mask(_segment_object(crop)))


def stable_axis_angle_degrees(angles: Sequence[float]) -> float:
    values = tuple(float(angle) for angle in angles)
    if not values or not all(math.isfinite(angle) for angle in values):
        raise ValueError("object orientation samples are invalid")
    doubled = tuple(math.radians(2.0 * angle) for angle in values)
    sine_sum = sum(math.sin(angle) for angle in doubled)
    cosine_sum = sum(math.cos(angle) for angle in doubled)
    if math.hypot(sine_sum, cosine_sum) < 1e-9:
        raise ValueError("object orientation samples are inconsistent")
    return math.degrees(0.5 * math.atan2(sine_sum, cosine_sum))
