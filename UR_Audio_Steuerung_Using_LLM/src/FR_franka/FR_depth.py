from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class DepthMeasurement:
    object_depth_mm: float
    support_depth_mm: float
    height_m: float
    object_sample_count: int
    support_sample_count: int


def measure_object_height(
    depth_path: Path,
    bbox: Sequence[float],
    inset_ratio: float,
    support_ring_scale: float,
    minimum_valid_pixels: int,
    maximum_height_m: float,
) -> DepthMeasurement:
    if not depth_path.is_file():
        raise FileNotFoundError(f"OAK D depth capture does not exist: {depth_path}")
    depth = np.load(depth_path, allow_pickle=False)
    if depth.ndim != 2:
        raise ValueError("OAK D depth capture must be a two dimensional array")
    if len(bbox) != 4:
        raise ValueError("Selected object bounding box must contain four values")
    if not 0.0 <= inset_ratio < 0.5:
        raise ValueError("Depth bounding box inset ratio must be below one half")
    if support_ring_scale <= 1.0:
        raise ValueError("Depth support ring scale must be greater than one")
    if minimum_valid_pixels <= 0:
        raise ValueError("Depth minimum valid pixels must be positive")

    height, width = depth.shape
    x1, y1, x2, y2 = _clamped_box(bbox, width, height)
    box_width = x2 - x1
    box_height = y2 - y1
    inset_x = int(round(box_width * inset_ratio))
    inset_y = int(round(box_height * inset_ratio))
    object_values = _valid_depth_values(
        depth[y1 + inset_y:y2 - inset_y, x1 + inset_x:x2 - inset_x]
    )

    expanded_width = box_width * support_ring_scale
    expanded_height = box_height * support_ring_scale
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
    support_values = _valid_depth_values(ring[ring_mask])

    if object_values.size < minimum_valid_pixels:
        raise ValueError(
            f"OAK D object area has only {object_values.size} valid depth pixels"
        )
    if support_values.size < minimum_valid_pixels:
        raise ValueError(
            f"OAK D support ring has only {support_values.size} valid depth pixels"
        )

    object_depth_mm = float(np.median(object_values))
    support_depth_mm = float(np.median(support_values))
    height_m = (support_depth_mm - object_depth_mm) / 1000.0
    if height_m < 0.0:
        raise ValueError(
            "OAK D measured the selected object behind the surrounding support surface"
        )
    if height_m > maximum_height_m:
        raise ValueError(
            f"OAK D object height {height_m:.4f} m exceeds the configured limit"
        )
    return DepthMeasurement(
        object_depth_mm=object_depth_mm,
        support_depth_mm=support_depth_mm,
        height_m=height_m,
        object_sample_count=int(object_values.size),
        support_sample_count=int(support_values.size),
    )


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
