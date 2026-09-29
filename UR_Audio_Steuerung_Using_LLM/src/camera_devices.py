# MO_Changes
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np


CAMERA_DEVICE_ENV = "VISION_CAMERA_DEVICE"
OAK_D_BACKEND = "oak_d"


@dataclass(frozen=True)
class CameraOption:
    label: str
    backend: str
    device: str

    @property
    def value(self) -> str:
        return f"{self.backend}:{self.device}"


@dataclass(frozen=True)
class OakDFrame:
    color: np.ndarray
    depth_mm: np.ndarray


def discover_oak_cameras() -> list[CameraOption]:
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
        label = f"OAK D {identifier}" if identifier else "OAK D"
        options.append(CameraOption(label, OAK_D_BACKEND, identifier))
    return options


def parse_camera_option(value: str) -> str:
    backend, separator, device = value.partition(":")
    if not separator or backend != OAK_D_BACKEND:
        raise ValueError(f"Invalid OAK D camera selection {value}")
    return device


def apply_camera_environment(value: str) -> None:
    os.environ[CAMERA_DEVICE_ENV] = parse_camera_option(value)


class RgbCameraStream:
    def __init__(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Camera dimensions must be positive")
        self._width = width
        self._height = height
        self._device = os.environ.get(CAMERA_DEVICE_ENV, "")
        self._oak_device: Any = None
        self._oak_pipeline: Any = None
        self._oak_queue: Any = None

    @property
    def is_open(self) -> bool:
        return self._oak_pipeline is not None

    @property
    def backend(self) -> str:
        return OAK_D_BACKEND

    def start(self) -> None:
        if self.is_open:
            return
        self._start_oak_rgbd()

    def read(self) -> np.ndarray | None:
        frame = self.read_rgbd()
        return frame.color if frame is not None else None

    def read_rgbd(self) -> OakDFrame | None:
        if self._oak_queue is None:
            raise RuntimeError("OAK D camera is not open")
        messages = self._oak_queue.get()
        if messages is None:
            return None
        color_message = messages["rgb"]
        depth_message = messages["depth_aligned"]
        if color_message is None or depth_message is None:
            return None
        color = color_message.getCvFrame()
        depth_mm = depth_message.getFrame()
        if color is None or depth_mm is None:
            return None
        if color.shape[:2] != depth_mm.shape[:2]:
            raise RuntimeError(
                "OAK D aligned depth dimensions do not match the RGB frame"
            )
        return OakDFrame(color=color, depth_mm=depth_mm)

    def close(self) -> None:
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

    def _start_oak_rgbd(self) -> None:
        try:
            import depthai as dai
        except ImportError as error:
            raise RuntimeError("DepthAI is required for the OAK D camera") from error

        device = dai.Device(dai.DeviceInfo(self._device)) if self._device else dai.Device()
        pipeline = None
        try:
            pipeline = dai.Pipeline(device)
            rgb = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            stereo = pipeline.create(dai.node.StereoDepth)
            align = pipeline.create(dai.node.ImageAlign)
            sync = pipeline.create(dai.node.Sync)
            sync.setSyncThreshold(timedelta(milliseconds=34))

            rgb_output = rgb.requestOutput(
                size=(self._width, self._height),
                type=dai.ImgFrame.Type.BGR888p,
                resizeMode=dai.ImgResizeMode.CROP,
                fps=15,
                enableUndistortion=False,
            )
            left_output = left.requestOutput(size=(640, 400), fps=15)
            right_output = right.requestOutput(size=(640, 400), fps=15)
            left_output.link(stereo.left)
            right_output.link(stereo.right)
            stereo.depth.link(align.input)
            rgb_output.link(align.inputAlignTo)
            rgb_output.link(sync.inputs["rgb"])
            align.outputAligned.link(sync.inputs["depth_aligned"])
            queue = sync.out.createOutputQueue(maxSize=2, blocking=False)
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


def _depthai_device_id(device: Any) -> str:
    for name in ("getDeviceId", "getMxId"):
        method = getattr(device, name, None)
        if callable(method):
            value = method()
            if value:
                return str(value)
    return str(getattr(device, "deviceId", ""))
