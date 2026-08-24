# MO_Changes
"""Object detector of the existing pick and drop system.

Loads the YOLOv5 weights of that repository in process and returns detections in
image coordinates. The existing repository runs its detector as a subprocess for
one photo at a time, which takes seconds. A live gesture loop cannot wait for
that, so the model is loaded once and called per frame here.

That repository detects bounding boxes and produces no segmentation masks, so
object selection works on boxes.

YOLOv5 gives no identity across frames. The selection rule needs the same object
to stay the same object while the operator holds a point, so boxes are matched
frame to frame by overlap and carry a stable id.
"""

from __future__ import annotations

import logging
import os
import pathlib
import time
from dataclasses import dataclass

import numpy as np

from config import ObjectModelConfig
from schemas import BoundingBox, DetectedObject

LOGGER = logging.getLogger(__name__)


def box_iou(first: BoundingBox, second: BoundingBox) -> float:
    """Overlap ratio of two boxes, used to keep an object id across frames."""
    left = max(first.x1, second.x1)
    top = max(first.y1, second.y1)
    right = min(first.x2, second.x2)
    bottom = min(first.y2, second.y2)
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    union = first.area + second.area - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass
class _Tracked:
    object_id: str
    class_name: str
    box: BoundingBox


class ObjectIdTracker:
    """Gives a detection the id of the box it overlaps most in the last frame."""

    def __init__(self, min_iou: float) -> None:
        self._min_iou = min_iou
        self._previous: list[_Tracked] = []
        self._next_index = 1

    def reset(self) -> None:
        self._previous = []
        self._next_index = 1

    def assign(self, detections: list[tuple[str, float, BoundingBox]]) -> list[_Tracked]:
        current: list[_Tracked] = []
        taken: set[str] = set()

        for class_name, _confidence, box in detections:
            best: _Tracked | None = None
            best_iou = self._min_iou
            for candidate in self._previous:
                if candidate.object_id in taken or candidate.class_name != class_name:
                    continue
                overlap = box_iou(box, candidate.box)
                if overlap >= best_iou:
                    best = candidate
                    best_iou = overlap

            if best is None:
                object_id = f"obj_{self._next_index:03d}"
                self._next_index += 1
            else:
                object_id = best.object_id
            taken.add(object_id)
            current.append(_Tracked(object_id=object_id, class_name=class_name, box=box))

        self._previous = current
        return current


class Yolov5ObjectDetector:
    """Adapter around the YOLOv5 detector of the existing repository."""

    def __init__(self, config: ObjectModelConfig, device: str, confidence: float) -> None:
        self._config = config
        self._device = device
        self._confidence = confidence
        self._model = None
        self._tracker = ObjectIdTracker(config.track_min_iou)
        self._last_latency_ms = 0.0
        self._consecutive_failures = 0

    @property
    def is_started(self) -> bool:
        return self._model is not None

    def start(self) -> None:
        if self._model is not None:
            return

        weights = self._config.weights_path
        repo = self._config.repo_dir
        if not weights.is_file():
            raise FileNotFoundError(
                f"object detection weights not found at {weights}. "
                "Point object_model.weights at the trained YOLOv5 file."
            )
        if not (repo / "hubconf.py").is_file():
            raise FileNotFoundError(
                f"the YOLOv5 source was not found at {repo}. "
                "Clone ultralytics/yolov5 into that folder so the weights can be loaded."
            )

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for object detection") from exc

        started = time.perf_counter()
        original_windows_path = pathlib.WindowsPath
        if os.name != "nt":
            pathlib.WindowsPath = pathlib.PosixPath
        try:
            model = torch.hub.load(str(repo), "custom", path=str(weights), source="local")
        finally:
            pathlib.WindowsPath = original_windows_path
        model.conf = self._confidence
        model.to(self._device)
        self._model = model
        LOGGER.info(
            "object_model_loaded weights=%s device=%s load_ms=%.1f names=%s",
            weights.name,
            self._device,
            (time.perf_counter() - started) * 1000.0,
            getattr(model, "names", None),
        )

    def get_objects(self, frame: np.ndarray) -> list[DetectedObject]:
        """Detect objects in one BGR frame.

        A failed inference returns no objects rather than raising, so one bad
        frame cannot end the interaction. Returning nothing is safe because an
        empty scene simply selects nothing.
        """
        if self._model is None:
            raise RuntimeError("Yolov5ObjectDetector.start must be called before get_objects")
        if frame is None or frame.ndim != 3:
            raise ValueError("frame must be a three dimensional BGR image")

        started = time.perf_counter()
        try:
            results = self._model(frame[:, :, ::-1], size=self._config.imgsz)
        except Exception:
            self._consecutive_failures += 1
            self._last_latency_ms = (time.perf_counter() - started) * 1000.0
            LOGGER.exception("object_inference_failed")
            return []

        self._consecutive_failures = 0
        self._last_latency_ms = (time.perf_counter() - started) * 1000.0
        return self._to_objects(results, frame.shape[1], frame.shape[0])

    def _to_objects(self, results, width: int, height: int) -> list[DetectedObject]:
        names = getattr(self._model, "names", {})
        rows = results.xyxy[0].detach().cpu().numpy()

        raw: list[tuple[str, float, BoundingBox]] = []
        for x1, y1, x2, y2, confidence, class_index in rows:
            box = BoundingBox(float(x1), float(y1), float(x2), float(y2)).clipped(width, height)
            if box.area <= 0.0:
                continue
            class_name = str(names.get(int(class_index), int(class_index)))
            raw.append((class_name, float(confidence), box))

        tracked = self._tracker.assign(raw)
        return [
            DetectedObject(
                object_id=item.object_id,
                class_name=item.class_name,
                confidence=confidence,
                box=item.box,
            )
            for item, (_name, confidence, _box) in zip(tracked, raw)
        ]

    def close(self) -> None:
        self._model = None
        self._tracker.reset()

    def health(self) -> dict[str, object]:
        return {
            "component": "object_source",
            "implementation": "yolov5",
            "loaded": self._model is not None,
            "weights": str(self._config.weights_path),
            "device": self._device,
            "last_latency_ms": round(self._last_latency_ms, 2),
            "consecutive_failures": self._consecutive_failures,
        }

    def __enter__(self) -> "Yolov5ObjectDetector":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
