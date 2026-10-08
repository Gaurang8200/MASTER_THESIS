from __future__ import annotations

import sys
from collections import deque
from datetime import timedelta
from pathlib import Path
from typing import Any

import cv2
import depthai as dai
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = REPOSITORY_ROOT / "UR_Audio_Steuerung_Using_LLM"
PIPELINE_ROOT = REPOSITORY_ROOT / "Code" / "gesture_selection_system" / "pipeline"
for folder in (AUDIO_ROOT, PIPELINE_ROOT / "support", PIPELINE_ROOT / "detection"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from config import ObjectModelConfig, resolve_device
from object_detector import Yolov5ObjectDetector


FRAME_SIZE = (1280, 720)
RGB_SENSOR_SIZE = (1920, 1080)
STEREO_SIZE = (640, 400)
YOLO_ROOT = AUDIO_ROOT / "Code-YOLOv5-Windows_llm" / "yolov5"
DEPTH_PATCH_RADIUS_PX = 3
DEPTH_HISTORY_FRAMES = 5
MIN_VALID_HISTORY_FRAMES = 3
MIN_VALID_DEPTH_FRACTION = 0.25
MIN_DEPTH_MM = 100.0
MAX_DEPTH_MM = 10000.0
MAX_TEMPORAL_DEVIATION_MM = 25.0


class ObjectDepthStabilizer:
    def __init__(self) -> None:
        self._history: dict[str, deque[float | None]] = {}

    def update(self, object_id: str, depth_mm: float | None) -> float:
        history = self._history.setdefault(
            object_id,
            deque(maxlen=DEPTH_HISTORY_FRAMES),
        )
        history.append(depth_mm)
        valid_depths = np.asarray(
            [value for value in history if value is not None],
            dtype=np.float32,
        )
        if valid_depths.size < MIN_VALID_HISTORY_FRAMES:
            raise ValueError("not enough recent valid depth frames")

        stable_depth_mm = float(np.median(valid_depths))
        median_deviation_mm = float(
            np.median(np.abs(valid_depths - stable_depth_mm))
        )
        if median_deviation_mm > MAX_TEMPORAL_DEVIATION_MM:
            raise ValueError("recent depth frames are inconsistent")
        return stable_depth_mm

    def retain(self, active_object_ids: set[str]) -> None:
        self._history = {
            object_id: history
            for object_id, history in self._history.items()
            if object_id in active_object_ids
        }


def median_depth_near_point(
    depth_mm: np.ndarray,
    point: tuple[float, float],
) -> float | None:
    if depth_mm.ndim != 2:
        raise ValueError("OAK D depth map must be a two dimensional array")
    height, width = depth_mm.shape
    center_x = int(round(point[0]))
    center_y = int(round(point[1]))
    if not 0 <= center_x < width or not 0 <= center_y < height:
        raise ValueError("OAK D object centre is outside the depth frame")

    x1 = max(0, center_x - DEPTH_PATCH_RADIUS_PX)
    x2 = min(width, center_x + DEPTH_PATCH_RADIUS_PX + 1)
    y1 = max(0, center_y - DEPTH_PATCH_RADIUS_PX)
    y2 = min(height, center_y + DEPTH_PATCH_RADIUS_PX + 1)
    patch = depth_mm[y1:y2, x1:x2]
    valid_mask = (
        np.isfinite(patch)
        & (patch >= MIN_DEPTH_MM)
        & (patch <= MAX_DEPTH_MM)
    )
    required_pixels = max(
        1,
        int(np.ceil(patch.size * MIN_VALID_DEPTH_FRACTION)),
    )
    valid_depths = patch[valid_mask]
    if valid_depths.size < required_pixels:
        return None
    return float(np.median(valid_depths))


class ShortRangeRgbdCamera:
    def __init__(self) -> None:
        self._device: Any = None
        self._pipeline: Any = None
        self._queue: Any = None

    def start(self) -> None:
        device = dai.Device()
        pipeline = None
        try:
            pipeline = dai.Pipeline(device)
            rgb = pipeline.create(dai.node.Camera).build(
                dai.CameraBoardSocket.CAM_A,
                sensorResolution=RGB_SENSOR_SIZE,
                sensorFps=15,
            )
            left = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
            right = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setSubpixel(False)
            stereo.setExtendedDisparity(True)
            sync = pipeline.create(dai.node.Sync)
            sync.setSyncThreshold(timedelta(milliseconds=34))

            rgb_output = rgb.requestOutput(
                size=FRAME_SIZE,
                type=dai.ImgFrame.Type.BGR888p,
                fps=15,
                enableUndistortion=False,
            )
            left_output = left.requestOutput(size=STEREO_SIZE, fps=15)
            right_output = right.requestOutput(size=STEREO_SIZE, fps=15)
            left_output.link(stereo.left)
            right_output.link(stereo.right)
            rgb_output.link(sync.inputs["rgb"])
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
        self._device = device
        self._pipeline = pipeline
        self._queue = queue

    def read(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self._queue is None:
            raise RuntimeError("Short range OAK D camera is not open")
        messages = self._queue.get()
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
            raise RuntimeError("Aligned depth dimensions do not match RGB")
        return color, depth_mm

    def close(self) -> None:
        pipeline = self._pipeline
        device = self._device
        self._pipeline = None
        self._device = None
        self._queue = None
        try:
            if pipeline is not None:
                pipeline.stop()
                pipeline.wait()
        finally:
            if device is not None:
                device.close()


def main() -> None:
    model_config = ObjectModelConfig(
        repo_path=str(YOLO_ROOT),
        weights=str(YOLO_ROOT / "my_model.pt"),
        imgsz=640,
        track_min_iou=0.5,
    )
    detector = Yolov5ObjectDetector(
        model_config,
        device=resolve_device("auto"),
        confidence=0.25,
    )
    camera = ShortRangeRgbdCamera()
    depth_stabilizer = ObjectDepthStabilizer()

    try:
        detector.start()
        camera.start()
        size_printed = False

        while True:
            rgbd_frame = camera.read()
            if rgbd_frame is None:
                continue
            frame, depth_mm = rgbd_frame
            height, width = frame.shape[:2]
            if (width, height) != FRAME_SIZE:
                raise RuntimeError(
                    f"OAK D returned {width} x {height}, expected 1280 x 720"
                )
            if not size_printed:
                print(
                    "OAK D test mode: RGB 1280 x 720, stereo 640 x 400, "
                    "Extended Disparity"
                )
                size_printed = True

            display = frame.copy()
            detected_objects = detector.get_objects(frame)
            active_object_ids = {
                detected_object.object_id for detected_object in detected_objects
            }
            depth_stabilizer.retain(active_object_ids)
            for detected_object in detected_objects:
                box = detected_object.box.as_int_tuple()
                center = detected_object.box.center
                try:
                    spatial_depth_mm = median_depth_near_point(depth_mm, center)
                    object_depth_mm = depth_stabilizer.update(
                        detected_object.object_id,
                        spatial_depth_mm,
                    )
                    depth_text = f"{object_depth_mm:.0f} mm"
                except ValueError:
                    depth_text = "depth unavailable"
                _draw_object(
                    display,
                    box,
                    detected_object.class_name,
                    detected_object.confidence,
                    depth_text,
                )

            cv2.imshow("OAK D RGB with object depth", display)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cv2.destroyAllWindows()
        detector.close()
        camera.close()


def _draw_object(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    class_name: str,
    confidence: float,
    depth_text: str,
) -> None:
    x1, y1, x2, y2 = box
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.drawMarker(
        frame,
        (int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))),
        (0, 255, 255),
        cv2.MARKER_CROSS,
        16,
        2,
    )
    cv2.putText(
        frame,
        f"{class_name} {confidence:.2f}  {depth_text}",
        (x1, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


if __name__ == "__main__":
    main()
