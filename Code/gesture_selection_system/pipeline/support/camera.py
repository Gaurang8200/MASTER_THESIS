# MO_Changes
"""Webcam capture helper.

Capture is a resource, so it is owned by a small class with explicit start and
close and a context manager, which guarantees the device is released on error
and on keyboard interrupt.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
AUDIO_PROJECT_ROOT = REPOSITORY_ROOT / "UR_Audio_Steuerung_Using_LLM"
if str(AUDIO_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_PROJECT_ROOT))

from src.camera_devices import OAK_RGB_BACKEND, RgbCameraStream

from config import CameraConfig

LOGGER = logging.getLogger(__name__)


class CameraStream:
    """Opens one camera index and yields frames in the configured size."""

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._capture: RgbCameraStream | None = None
        self._consecutive_failures = 0

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.is_open

    @property
    def rotation_degrees(self) -> int:
        if self._capture is not None and self._capture.backend == OAK_RGB_BACKEND:
            return 0
        return self._config.rotation_degrees

    def start(self) -> None:
        if self.is_open:
            return
        capture = RgbCameraStream(self._config.width, self._config.height)
        capture.start()
        self._capture = capture
        LOGGER.info(
            "camera_opened requested=%dx%d",
            self._config.width,
            self._config.height,
        )

    def read(self) -> np.ndarray | None:
        """Return the next frame or None when the grab failed."""
        if self._capture is None:
            raise RuntimeError("CameraStream.start must be called before read")
        frame = self._capture.read()
        if frame is None:
            self._consecutive_failures += 1
            return None
        self._consecutive_failures = 0
        if self.rotation_degrees == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        if self._config.flip_horizontal:
            frame = cv2.flip(frame, 1)
        return frame

    def to_sensor_point(
        self,
        point: tuple[float, float],
        frame_shape: tuple[int, ...],
    ) -> tuple[float, float]:
        """Convert a displayed point back to the original camera coordinates."""
        height, width = frame_shape[:2]
        x_value, y_value = point
        if self._config.flip_horizontal:
            x_value = width - 1.0 - x_value
        if self.rotation_degrees == 180:
            x_value = width - 1.0 - x_value
            y_value = height - 1.0 - y_value
        return x_value, y_value

    def to_sensor_box(
        self,
        box: tuple[float, float, float, float],
        frame_shape: tuple[int, ...],
    ) -> tuple[float, float, float, float]:
        """Convert a displayed box back to the original camera coordinates."""
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = box
        if self._config.flip_horizontal:
            x1, x2 = width - x2, width - x1
        if self.rotation_degrees == 180:
            x1, x2 = width - x2, width - x1
            y1, y2 = height - y2, height - y1
        return x1, y1, x2, y2

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def close(self) -> None:
        if self._capture is not None:
            self._capture.close()
            self._capture = None
            LOGGER.info("camera_closed")

    def __enter__(self) -> "CameraStream":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
