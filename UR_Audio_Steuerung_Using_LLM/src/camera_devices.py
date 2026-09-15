# MO_Changes
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2


CAMERA_BACKEND_ENV = "VISION_CAMERA_BACKEND"
CAMERA_DEVICE_ENV = "VISION_CAMERA_DEVICE"
OPENCV_BACKEND = "opencv"
OAK_RGB_BACKEND = "oak_rgb"


@dataclass(frozen=True)
class CameraOption:
    label: str
    backend: str
    device: str

    @property
    def value(self) -> str:
        return f"{self.backend}:{self.device}"


def discover_rgb_cameras(max_opencv_index: int = 9) -> list[CameraOption]:
    options = _discover_oak_cameras()
    for index in range(max_opencv_index + 1):
        capture = cv2.VideoCapture(index)
        try:
            if capture.isOpened():
                options.append(
                    CameraOption(_opencv_label(index), OPENCV_BACKEND, str(index))
                )
        finally:
            capture.release()
    return options


def parse_camera_option(value: str) -> tuple[str, str]:
    backend, separator, device = value.partition(":")
    if not separator or backend not in {OPENCV_BACKEND, OAK_RGB_BACKEND}:
        raise ValueError(f"Invalid camera selection {value}")
    return backend, device


def apply_camera_environment(value: str) -> None:
    backend, device = parse_camera_option(value)
    os.environ[CAMERA_BACKEND_ENV] = backend
    os.environ[CAMERA_DEVICE_ENV] = device


class RgbCameraStream:
    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Camera dimensions must be positive")
        self._width = width
        self._height = height
        self._backend = os.environ.get(CAMERA_BACKEND_ENV, OPENCV_BACKEND)
        self._device = os.environ.get(CAMERA_DEVICE_ENV, "0")
        self._capture: cv2.VideoCapture | None = None
        self._oak_device: Any = None
        self._oak_pipeline: Any = None
        self._oak_queue: Any = None

    @property
    def is_open(self) -> bool:
        if self._backend == OPENCV_BACKEND:
            return self._capture is not None and self._capture.isOpened()
        return self._oak_pipeline is not None

    @property
    def backend(self) -> str:
        return self._backend

    def start(self) -> None:
        if self.is_open:
            return
        if self._backend == OAK_RGB_BACKEND:
            self._start_oak_rgb()
            return
        self._start_opencv()

    def read(self) -> Any | None:
        if self._backend == OAK_RGB_BACKEND:
            if self._oak_queue is None:
                raise RuntimeError("OAK D RGB camera is not open")
            message = self._oak_queue.get()
            return message.getCvFrame() if message is not None else None
        if self._capture is None:
            raise RuntimeError("OpenCV camera is not open")
        success, frame = self._capture.read()
        return frame if success else None

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        pipeline = self._oak_pipeline
        device = self._oak_device
        self._oak_pipeline = None
        self._oak_device = None
        self._oak_queue = None
        try:
            if pipeline is not None:
                pipeline.stop()
                pipeline.wait()
        finally:
            if device is not None:
                device.close()

    def _start_opencv(self) -> None:
        try:
            index = int(self._device)
        except ValueError as error:
            raise ValueError(f"Invalid OpenCV camera index {self._device}") from error
        capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Could not open OpenCV camera {index}")
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._capture = capture

    def _start_oak_rgb(self) -> None:
        try:
            import depthai as dai
        except ImportError as error:
            raise RuntimeError("DepthAI is required for the OAK D RGB camera") from error

        device = dai.Device(dai.DeviceInfo(self._device)) if self._device else dai.Device()
        pipeline = None
        try:
            pipeline = dai.Pipeline(device)
            camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            output = camera.requestOutput(
                size=(self._width, self._height),
                type=dai.ImgFrame.Type.BGR888p,
                resizeMode=dai.ImgResizeMode.CROP,
                fps=15,
            )
            queue = output.createOutputQueue(maxSize=2, blocking=False)
            pipeline.start()
        except Exception:
            try:
                if pipeline is not None:
                    pipeline.stop()
            finally:
                device.close()
            raise
        self._oak_device = device
        self._oak_pipeline = pipeline
        self._oak_queue = queue


def _discover_oak_cameras() -> list[CameraOption]:
    try:
        import depthai as dai
    except ImportError:
        return []
    try:
        devices = dai.Device.getAllAvailableDevices()
    except Exception:
        return []
    options: list[CameraOption] = []
    for device in devices:
        identifier = _depthai_device_id(device)
        label = f"OAK D RGB {identifier}" if identifier else "OAK D RGB"
        options.append(CameraOption(label, OAK_RGB_BACKEND, identifier))
    return options


def _depthai_device_id(device: Any) -> str:
    for name in ("getDeviceId", "getMxId"):
        method = getattr(device, name, None)
        if callable(method):
            value = method()
            if value:
                return str(value)
    return str(getattr(device, "deviceId", ""))


def _opencv_label(index: int) -> str:
    if sys.platform.startswith("linux"):
        name_path = Path(f"/sys/class/video4linux/video{index}/name")
        if name_path.is_file():
            name = name_path.read_text(encoding="utf8").strip()
            if name:
                return f"{name} Camera {index}"
    return f"Camera {index}"
