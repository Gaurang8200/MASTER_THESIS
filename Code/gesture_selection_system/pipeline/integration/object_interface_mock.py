"""Mock object detection and segmentation source.

The gesture pipeline depends only on the ``ObjectSource`` protocol, so this mock
can be replaced by an adapter around the existing YOLOv5 detection and
segmentation repository without touching the selection logic. The mock is fully
deterministic so tests and demos behave the same on every run.

Masks and boxes are in image coordinates. The pick pose belongs to the object
system, the gesture module never computes it and never calibrates for it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

from schemas import BoundingBox, DetectedObject, RobotPose

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MockObjectSpec:
    """Object placed in normalized image coordinates so it follows the frame size."""

    object_id: str
    class_name: str
    confidence: float
    center: tuple[float, float]
    size: tuple[float, float]
    shape: Literal["ellipse", "rect"] = "ellipse"
    pick_pose_m: tuple[float, float, float] | None = None


DEFAULT_MOCK_OBJECTS: tuple[MockObjectSpec, ...] = (
    MockObjectSpec("obj_001", "red_cube", 0.93, (0.26, 0.52), (0.14, 0.20), "rect", (0.42, -0.18, 0.03)),
    MockObjectSpec("obj_002", "blue_cylinder", 0.88, (0.52, 0.60), (0.13, 0.22), "ellipse", (0.48, 0.01, 0.04)),
    MockObjectSpec("obj_003", "green_block", 0.81, (0.76, 0.50), (0.15, 0.18), "rect", (0.44, 0.19, 0.03)),
    # Kept below the object confidence threshold on purpose so the demo shows
    # that weak detections never take part in selection.
    MockObjectSpec("obj_004", "unknown_part", 0.55, (0.64, 0.82), (0.11, 0.13), "ellipse", None),
)


def _mask_for(spec: MockObjectSpec, width: int, height: int) -> tuple[np.ndarray, BoundingBox]:
    center_x = spec.center[0] * width
    center_y = spec.center[1] * height
    half_w = max(spec.size[0] * width / 2.0, 1.0)
    half_h = max(spec.size[1] * height / 2.0, 1.0)

    ys, xs = np.ogrid[0:height, 0:width]
    if spec.shape == "rect":
        mask = (np.abs(xs - center_x) <= half_w) & (np.abs(ys - center_y) <= half_h)
    else:
        mask = ((xs - center_x) / half_w) ** 2 + ((ys - center_y) / half_h) ** 2 <= 1.0

    box = BoundingBox(
        x1=center_x - half_w,
        y1=center_y - half_h,
        x2=center_x + half_w,
        y2=center_y + half_h,
    ).clipped(width, height)
    return np.ascontiguousarray(mask), box


class MockObjectDetector:
    """Returns a fixed scene of segmented objects for the given frame size.

    Masks are built once per frame shape and reused, which keeps the demo loop
    cheap and makes the frame to frame object identity stable. A stable
    ``object_id`` is required, otherwise the stable frame rule never confirms.
    """

    def __init__(self, specs: Sequence[MockObjectSpec] | None = None) -> None:
        self._specs = tuple(specs) if specs is not None else DEFAULT_MOCK_OBJECTS
        self._started = False
        self._cache_shape: tuple[int, int] | None = None
        self._cache: list[DetectedObject] = []

    def start(self) -> None:
        self._started = True

    def close(self) -> None:
        self._started = False
        self._cache_shape = None
        self._cache = []

    def get_objects(self, frame: np.ndarray) -> Sequence[DetectedObject]:
        if not self._started:
            raise RuntimeError("MockObjectDetector.start must be called before get_objects")
        if frame is None or frame.ndim < 2:
            raise ValueError("frame must be an image array")

        height, width = frame.shape[:2]
        if self._cache_shape == (height, width):
            return self._cache

        objects: list[DetectedObject] = []
        for spec in self._specs:
            mask, box = _mask_for(spec, width, height)
            objects.append(
                DetectedObject(
                    object_id=spec.object_id,
                    class_name=spec.class_name,
                    confidence=spec.confidence,
                    box=box,
                    mask=mask,
                    pick_pose=(
                        RobotPose(*spec.pick_pose_m, rx_deg=180.0)
                        if spec.pick_pose_m is not None
                        else None
                    ),
                )
            )

        self._cache_shape = (height, width)
        self._cache = objects
        LOGGER.debug("mock_objects_built count=%d shape=%dx%d", len(objects), width, height)
        return objects

    def health(self) -> dict[str, object]:
        return {
            "component": "object_source",
            "implementation": "mock",
            "started": self._started,
            "object_count": len(self._specs),
        }

    def __enter__(self) -> "MockObjectDetector":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
