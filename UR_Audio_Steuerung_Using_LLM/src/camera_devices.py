# MO_Changes
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np


CAMERA_DEVICE_ENV = "VISION_CAMERA_DEVICE"
OAK_D_BACKEND = "oak_d"
OAK_MONO_RESOLUTION = (1280, 720)


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
    left_mono: np.ndarray
    rgb_transformation: Any
    left_transformation: Any

    def project_rgb_point_to_left(
        self,
        point: tuple[float, float],
        depth_mm: float,
    ) -> tuple[float, float]:
        if not np.isfinite(depth_mm) or depth_mm <= 0.0:
            raise ValueError("OAK D projection requires a valid positive depth")
        try:
            import depthai as dai
        except ImportError as error:
            raise RuntimeError("DepthAI is required for OAK D projection") from error
        projected = self.rgb_transformation.projectPointTo(
            self.left_transformation,
            dai.Point2f(float(point[0]), float(point[1])),
            float(depth_mm),
        )
        left_height, left_width = self.left_mono.shape[:2]
        if not 0.0 <= projected.x < left_width or not 0.0 <= projected.y < left_height:
            raise ValueError("Projected left mono pixel is outside the calibration frame")
        return float(projected.x), float(projected.y)


def median_depth_for_bbox(
    depth_mm: np.ndarray,
    bbox: tuple[float, float, float, float] | list[float],
    inset_ratio: float = 0.25,
    minimum_valid_pixels: int = 25,
) -> float:
    if depth_mm.ndim != 2:
        raise ValueError("OAK D depth map must be a two dimensional array")
    if len(bbox) != 4:
        raise ValueError("Object bounding box must contain four values")
    if not 0.0 <= inset_ratio < 0.5:
        raise ValueError("Depth bounding box inset ratio must be below one half")
    height, width = depth_mm.shape
    x1 = max(0, min(width - 1, int(round(float(bbox[0])))))
    y1 = max(0, min(height - 1, int(round(float(bbox[1])))))
    x2 = max(x1 + 1, min(width, int(round(float(bbox[2])))))
    y2 = max(y1 + 1, min(height, int(round(float(bbox[3])))))
    inset_x = int(round((x2 - x1) * inset_ratio))
    inset_y = int(round((y2 - y1) * inset_ratio))
    values = np.asarray(
        depth_mm[y1 + inset_y:y2 - inset_y, x1 + inset_x:x2 - inset_x],
        dtype=np.float64,
    ).reshape(-1)
    valid = values[(values >= 100.0) & (values <= 10000.0) & np.isfinite(values)]
    if valid.size < minimum_valid_pixels:
        raise ValueError(
            f"OAK D object area has only {valid.size} valid depth pixels"
        )
    return float(np.median(valid))


def map_detections_to_left_mono(
    detections: list[dict[str, Any]],
    frame: OakDFrame,
) -> list[dict[str, Any]]:
    mapped_detections: list[dict[str, Any]] = []
    left_height, left_width = frame.left_mono.shape[:2]
    for detection in detections:
        mapped = dict(detection)
        try:
            bbox = [float(value) for value in detection["bbox"]]
            center = [float(value) for value in detection["center"]]
            object_depth_mm = median_depth_for_bbox(frame.depth_mm, bbox)
            left_center = frame.project_rgb_point_to_left(
                (center[0], center[1]),
                object_depth_mm,
            )
            mapped["left_mono_center"] = [left_center[0], left_center[1]]
            mapped["left_mono_frame_size"] = [left_width, left_height]
            mapped["object_depth_mm"] = object_depth_mm
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            mapped["left_mono_mapping_error"] = str(error)
        mapped_detections.append(mapped)
    return mapped_detections


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
        left_message = messages["left_mono"]
        if color_message is None or depth_message is None or left_message is None:
            return None
        color = color_message.getCvFrame()
        depth_mm = depth_message.getFrame()
        left_mono = left_message.getCvFrame()
        if color is None or depth_mm is None or left_mono is None:
            return None
        if color.shape[:2] != depth_mm.shape[:2]:
            raise RuntimeError(
                "OAK D aligned depth dimensions do not match the RGB frame"
            )
        return OakDFrame(
            color=color,
            depth_mm=depth_mm,
            left_mono=left_mono,
            rgb_transformation=color_message.getTransformation(),
            left_transformation=left_message.getTransformation(),
        )

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
            sync = pipeline.create(dai.node.Sync)
            sync.setSyncThreshold(timedelta(milliseconds=34))

            rgb_output = rgb.requestOutput(
                size=(self._width, self._height),
                type=dai.ImgFrame.Type.BGR888p,
                resizeMode=dai.ImgResizeMode.CROP,
                fps=15,
                enableUndistortion=False,
            )
            left_output = left.requestOutput(size=OAK_MONO_RESOLUTION, fps=15)
            right_output = right.requestOutput(size=OAK_MONO_RESOLUTION, fps=15)
            left_output.link(stereo.left)
            right_output.link(stereo.right)
            rgb_output.link(sync.inputs["rgb"])
            left_output.link(sync.inputs["left_mono"])
            if device.getPlatform() == dai.Platform.RVC4:
                align = pipeline.create(dai.node.ImageAlign)
                stereo.depth.link(align.input)
                rgb_output.link(align.inputAlignTo)
                align.outputAligned.link(sync.inputs["depth_aligned"])
            else:
                rgb_output.link(stereo.inputAlignTo)
                stereo.depth.link(sync.inputs["depth_aligned"])
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
