"""Webcam capture helper.

Capture is a resource, so it is owned by a small class with explicit start and
close and a context manager, which guarantees the device is released on error
and on keyboard interrupt.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from config import CameraConfig

LOGGER = logging.getLogger(__name__)


class CameraStream:
    """Opens one camera index and yields frames in the configured size."""

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._capture: cv2.VideoCapture | None = None
        self._consecutive_failures = 0

    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def start(self) -> None:
        if self.is_open:
            return
        capture = cv2.VideoCapture(self._config.index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"could not open camera index {self._config.index}. "
                "Check that no other application holds the device."
            )
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        self._capture = capture
        actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        LOGGER.info(
            "camera_opened index=%d requested=%dx%d actual=%dx%d",
            self._config.index,
            self._config.width,
            self._config.height,
            actual_w,
            actual_h,
        )

    def read(self) -> np.ndarray | None:
        """Return the next frame or None when the grab failed."""
        if self._capture is None:
            raise RuntimeError("CameraStream.start must be called before read")
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self._consecutive_failures += 1
            return None
        self._consecutive_failures = 0
        if self._config.flip_horizontal:
            frame = cv2.flip(frame, 1)
        return frame

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            LOGGER.info("camera_closed index=%d", self._config.index)

    def __enter__(self) -> "CameraStream":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
