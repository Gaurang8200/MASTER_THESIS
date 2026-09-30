from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..camera_devices import depth_at_pixel


@dataclass(frozen=True)
class DepthMeasurement:
    object_depth_mm: float
    table_depth_mm: float
    height_mm: float
    object_sample_count: int
    table_sample_count: int
    table_depth_min_mm: float
    table_depth_max_mm: float


def measure_object_height(
    depth_path: Path,
    bbox: Sequence[float],
    table_ring_scale: float,
    minimum_valid_pixels: int,
    table_depth_tolerance_mm: float,
    maximum_height_mm: float,
) -> DepthMeasurement:
    if not depth_path.is_file():
        raise FileNotFoundError(f"OAK D depth capture does not exist: {depth_path}")
    depth = np.load(depth_path, allow_pickle=False)
    if depth.ndim != 2:
        raise ValueError("OAK D depth capture must be a two dimensional array")
    if len(bbox) != 4:
        raise ValueError("Selected object bounding box must contain four values")
    if table_ring_scale <= 1.0:
        raise ValueError("Depth table ring scale must be greater than one")
    if minimum_valid_pixels <= 0:
        raise ValueError("Depth minimum valid pixels must be positive")
    if table_depth_tolerance_mm <= 0.0:
        raise ValueError("Depth table tolerance must be positive")

    height, width = depth.shape
    x1, y1, x2, y2 = _clamped_box(bbox, width, height)
    box_width = x2 - x1
    box_height = y2 - y1
    object_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    object_depth_mm = depth_at_pixel(depth, object_center)

    expanded_width = box_width * table_ring_scale
    expanded_height = box_height * table_ring_scale
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    outer = _clamped_box(
        (
            center_x - expanded_width / 2.0,
            center_y - expanded_height / 2.0,
            center_x + expanded_width / 2.0,
            center_y + expanded_height / 2.0,
        ),
        width,
        height,
    )
    ox1, oy1, ox2, oy2 = outer
    ring = depth[oy1:oy2, ox1:ox2]
    ring_mask = np.ones(ring.shape, dtype=bool)
    ring_mask[y1 - oy1:y2 - oy1, x1 - ox1:x2 - ox1] = False
    table_values = _valid_depth_values(ring[ring_mask])

    if table_values.size < minimum_valid_pixels:
        raise ValueError(
            f"OAK D table ring has only {table_values.size} valid depth pixels"
        )

    table_candidates = _dominant_depth_values(
        table_values,
        table_depth_tolerance_mm,
    )
    if table_candidates.size < minimum_valid_pixels:
        raise ValueError(
            "OAK D largest surrounding depth group has only "
            f"{table_candidates.size} valid pixels"
        )
    table_depth_mm = float(np.median(table_candidates))
    table_depth_min_mm = float(np.min(table_candidates))
    table_depth_max_mm = float(np.max(table_candidates))
    height_mm = table_depth_mm - object_depth_mm
    if height_mm < 0.0:
        raise ValueError(
            "OAK D measured the selected object behind the table surface"
        )
    if height_mm > maximum_height_mm:
        raise ValueError(
            f"OAK D object height {height_mm:.1f} mm exceeds the configured limit"
        )
    measurement = DepthMeasurement(
        object_depth_mm=object_depth_mm,
        table_depth_mm=table_depth_mm,
        height_mm=height_mm,
        object_sample_count=1,
        table_sample_count=int(table_candidates.size),
        table_depth_min_mm=table_depth_min_mm,
        table_depth_max_mm=table_depth_max_mm,
    )
    _show_depth_measurement(
        depth,
        (x1, y1, x2, y2),
        object_center,
        outer,
        measurement,
    )
    return measurement


def _clamped_box(
    bbox: Sequence[float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1 = max(0, min(width - 1, int(round(float(bbox[0])))))
    y1 = max(0, min(height - 1, int(round(float(bbox[1])))))
    x2 = max(x1 + 1, min(width, int(round(float(bbox[2])))))
    y2 = max(y1 + 1, min(height, int(round(float(bbox[3])))))
    return x1, y1, x2, y2


def _valid_depth_values(values: np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64).reshape(-1)
    return numeric[(numeric >= 100.0) & (numeric <= 10000.0) & np.isfinite(numeric)]


def _dominant_depth_values(values: np.ndarray, tolerance_mm: float) -> np.ndarray:
    sorted_values = np.sort(values)
    start = 0
    best_start = 0
    best_stop = 1
    maximum_span_mm = 2.0 * tolerance_mm
    for stop, value in enumerate(sorted_values):
        while value - sorted_values[start] > maximum_span_mm:
            start += 1
        if stop + 1 - start > best_stop - best_start:
            best_start = start
            best_stop = stop + 1
    cluster = sorted_values[best_start:best_stop]
    center_mm = (cluster[0] + cluster[-1]) / 2.0
    return sorted_values[np.abs(sorted_values - center_mm) <= tolerance_mm]


def _show_depth_measurement(
    depth: np.ndarray,
    object_box: tuple[int, int, int, int],
    object_pixel: tuple[float, float],
    table_box: tuple[int, int, int, int],
    measurement: DepthMeasurement,
) -> None:
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        return

    import cv2

    valid = _valid_depth_values(depth)
    near_mm, far_mm = np.percentile(valid, (5.0, 95.0))
    span_mm = max(float(far_mm - near_mm), 1.0)
    scaled = np.clip((far_mm - depth) * 255.0 / span_mm, 0.0, 255.0)
    visual = cv2.applyColorMap(scaled.astype(np.uint8), cv2.COLORMAP_TURBO)
    visual[depth == 0] = 0

    x1, y1, x2, y2 = object_box
    object_x, object_y = int(round(object_pixel[0])), int(round(object_pixel[1]))
    ox1, oy1, ox2, oy2 = table_box
    table_mask = np.zeros(depth.shape, dtype=bool)
    table_mask[oy1:oy2, ox1:ox2] = True
    table_mask[y1:y2, x1:x2] = False
    table_mask &= (depth >= measurement.table_depth_min_mm) & (
        depth <= measurement.table_depth_max_mm
    )
    visual[table_mask] = (255, 0, 0)

    cv2.rectangle(visual, (ox1, oy1), (ox2 - 1, oy2 - 1), (255, 0, 0), 2)
    cv2.rectangle(visual, (x1, y1), (x2 - 1, y2 - 1), (255, 0, 0), 2)
    cv2.drawMarker(
        visual,
        (object_x, object_y),
        (0, 255, 0),
        cv2.MARKER_CROSS,
        16,
        2,
    )
    cv2.putText(
        visual,
        f"Green center pixel {measurement.object_depth_mm:.1f} mm",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        visual,
        f"Blue table {measurement.table_depth_mm:.1f} mm",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        visual,
        f"Object height {measurement.height_mm:.1f} mm",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )
    cv2.imshow("OAK D Depth Measurement 1280 x 720", visual)
    cv2.waitKey(1)
