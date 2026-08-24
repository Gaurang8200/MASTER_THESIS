# MO_Changes
"""YOLO gesture detector.

The detector owns the model session, so it exposes explicit start, close and
health methods. Ultralytics and torch are imported inside ``start`` to keep
module import free of heavy side effects and to let the pure logic modules be
tested without the inference stack installed.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from config import ConfidenceConfig, ModelConfig, resolve_device
from schemas import BoundingBox, GestureDetection, GestureFrame, GestureName

LOGGER = logging.getLogger(__name__)


def verify_model_classes(
    model_names: object,
    expected_names: dict[int, GestureName],
) -> None:
    """Reject a checkpoint whose classes do not match the pipeline.

    The model is trained outside this repository, so the class order of the
    downloaded file is checked before the first frame. A silent mismatch would
    read an open palm as a fingertip.
    """
    if not isinstance(model_names, dict):
        LOGGER.warning("model_class_names_unavailable, skipping the class check")
        return

    found = {int(index): str(name) for index, name in model_names.items()}
    expected = {int(index): gesture.value for index, gesture in expected_names.items()}
    if found != expected:
        raise ValueError(
            "the checkpoint was trained on different classes.\n"
            f"  expected: {expected}\n"
            f"  found:    {found}\n"
            "Retrain with the class order above or fix the export."
        )


class GestureDetector:
    """Runs the trained detector on a frame and returns typed gesture detections.

    The same trained model covers bare hands and gloved hands, so no separate
    code path exists for gloves. Glove coverage is a property of the training
    data and is verified through the dataset rather than at runtime.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        confidence: ConfidenceConfig,
        class_ids_by_index: dict[int, GestureName],
    ) -> None:
        self._config = model_config
        self._confidence = confidence
        self._class_ids = dict(class_ids_by_index)
        self._model = None
        self._device = "cpu"
        self._last_latency_ms = 0.0
        self._consecutive_failures = 0
        self._unknown_class_hits = 0

    @property
    def is_started(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    def start(self) -> None:
        """Load the weights and record model identity."""
        if self._model is not None:
            return

        weights = self._config.weights_path
        if not weights.is_file():
            raise FileNotFoundError(
                f"gesture weights not found at {weights}. "
                "Place the trained gesture model there or point model.weights at it."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for gesture inference. Install it with "
                "pip install ultralytics"
            ) from exc

        self._device = resolve_device(self._config.device)
        if self._config.half and self._device == "cpu":
            LOGGER.warning("half precision is not used on cpu, continuing in float32")

        started = time.perf_counter()
        model = YOLO(str(weights))
        verify_model_classes(getattr(model, "names", None), self._class_ids)
        self._model = model
        load_ms = (time.perf_counter() - started) * 1000.0
        LOGGER.info(
            "gesture_model_loaded weights=%s device=%s imgsz=%d load_ms=%.1f names=%s",
            weights.name,
            self._device,
            self._config.imgsz,
            load_ms,
            getattr(self._model, "names", None),
        )

    def detect(self, frame: np.ndarray, frame_index: int) -> GestureFrame:
        """Run inference on one BGR frame.

        Inference failure returns a frame with ``ok`` false instead of raising,
        so a single bad frame degrades the pipeline rather than killing it.
        """
        if self._model is None:
            raise RuntimeError("GestureDetector.start must be called before detect")
        if frame is None or frame.ndim != 3:
            raise ValueError("frame must be a three dimensional BGR image")

        started = time.perf_counter()
        try:
            results = self._model.predict(
                source=frame,
                imgsz=self._config.imgsz,
                conf=self._confidence.detector_floor,
                device=self._device,
                half=self._config.half and self._device != "cpu",
                max_det=self._config.max_detections,
                verbose=False,
            )
        except Exception as exc:  # inference must not crash the interaction loop
            self._consecutive_failures += 1
            self._last_latency_ms = (time.perf_counter() - started) * 1000.0
            LOGGER.exception("gesture_inference_failed frame=%d", frame_index)
            return GestureFrame(
                frame_index=frame_index,
                latency_ms=self._last_latency_ms,
                ok=False,
                error=str(exc),
            )

        self._consecutive_failures = 0
        self._last_latency_ms = (time.perf_counter() - started) * 1000.0
        height, width = frame.shape[:2]
        detections = self._parse(results, width, height)

        if self._last_latency_ms > self._config.latency_budget_ms:
            LOGGER.debug(
                "gesture_latency_over_budget frame=%d latency_ms=%.1f budget_ms=%.1f",
                frame_index,
                self._last_latency_ms,
                self._config.latency_budget_ms,
            )

        return GestureFrame(
            frame_index=frame_index,
            detections=tuple(detections),
            latency_ms=self._last_latency_ms,
        )

    def _parse(self, results, width: int, height: int) -> list[GestureDetection]:
        detections: list[GestureDetection] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), score, class_index in zip(xyxy, confs, classes):
                gesture = self._class_ids.get(int(class_index))
                if gesture is None:
                    self._unknown_class_hits += 1
                    continue
                if float(score) < self._confidence.threshold_for(gesture):
                    continue
                box = BoundingBox(float(x1), float(y1), float(x2), float(y2)).clipped(width, height)
                if box.area <= 0.0:
                    continue
                detections.append(
                    GestureDetection(gesture=gesture, confidence=float(score), box=box)
                )
        return detections

    def close(self) -> None:
        """Release the model session."""
        self._model = None

    def health(self) -> dict[str, object]:
        return {
            "component": "gesture_detector",
            "loaded": self._model is not None,
            "weights": str(self._config.weights_path),
            "device": self._device,
            "last_latency_ms": round(self._last_latency_ms, 2),
            "consecutive_failures": self._consecutive_failures,
            "unknown_class_hits": self._unknown_class_hits,
        }

    def __enter__(self) -> "GestureDetector":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def weights_available(model_config: ModelConfig) -> bool:
    """Cheap check used by the demos to give a clear message before loading."""
    return Path(model_config.weights_path).is_file()
