from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
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
DEPTH_PATCH_RADIUS_PX = 1
DEPTH_HISTORY_FRAMES = 5
MIN_VALID_HISTORY_FRAMES = 3
MIN_OBJECT_DEPTH_PIXELS = 3
MIN_DEPTH_MM = 100.0
TABLE_DEPTH_MIN_MM = 495.0
TABLE_DEPTH_MAX_MM = 505.0
OBJECT_DEPTH_CLUSTER_SPREAD_MM = 10.0
MAX_TEMPORAL_DEVIATION_MM = 25.0
MIN_STATIONARY_BOX_IOU = 0.80


@dataclass
class _ObjectDepthState:
    valid_depths: deque[float] = field(
        default_factory=lambda: deque(maxlen=DEPTH_HISTORY_FRAMES)
    )
    anchor_box: tuple[int, int, int, int] | None = None
    stable_depth_mm: float | None = None


class ObjectDepthStabilizer:
    def __init__(self) -> None:
        self._states: dict[str, _ObjectDepthState] = {}

    def update(
        self,
        object_id: str,
        box: tuple[int, int, int, int],
        depth_mm: float | None,
    ) -> float:
        state = self._states.setdefault(object_id, _ObjectDepthState())
        if (
            state.anchor_box is not None
            and _box_iou(state.anchor_box, box) < MIN_STATIONARY_BOX_IOU
        ):
            state = _ObjectDepthState(anchor_box=box)
            self._states[object_id] = state
        elif state.anchor_box is None:
            state.anchor_box = box

        if depth_mm is not None:
            state.valid_depths.append(depth_mm)
        valid_depths = np.asarray(state.valid_depths, dtype=np.float32)
        if valid_depths.size < MIN_VALID_HISTORY_FRAMES:
            if state.stable_depth_mm is None:
                raise ValueError("not enough valid depth measurements")
            return state.stable_depth_mm

        candidate_depth_mm = float(np.median(valid_depths))
        median_deviation_mm = float(
            np.median(np.abs(valid_depths - candidate_depth_mm))
        )
        if median_deviation_mm <= MAX_TEMPORAL_DEVIATION_MM:
            state.stable_depth_mm = candidate_depth_mm
        if state.stable_depth_mm is None:
            raise ValueError("valid depth measurements are inconsistent")
        return state.stable_depth_mm

    def retain(self, active_object_ids: set[str]) -> None:
        self._states = {
            object_id: state
            for object_id, state in self._states.items()
            if object_id in active_object_ids
        }


def _box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


def object_depth_from_box(
    depth_mm: np.ndarray,
    box: tuple[int, int, int, int],
) -> float | None:
    if depth_mm.ndim != 2:
        raise ValueError("OAK D depth map must be a two dimensional array")
    height, width = depth_mm.shape
    box_x1 = max(0, min(width, box[0]))
    box_y1 = max(0, min(height, box[1]))
    box_x2 = max(0, min(width, box[2]))
    box_y2 = max(0, min(height, box[3]))
    if box_x2 <= box_x1 or box_y2 <= box_y1:
        raise ValueError("OAK D object box is outside the depth frame")

    box_width = box_x2 - box_x1
    box_height = box_y2 - box_y1
    center_x = box_x1 + box_width / 2.0
    center_y = box_y1 + box_height / 2.0
    sample_centers = (
        (center_x, center_y),
        (box_x1 + box_width * 0.25, center_y),
        (box_x1 + box_width * 0.75, center_y),
        (center_x, box_y1 + box_height * 0.25),
        (center_x, box_y1 + box_height * 0.75),
    )

    sampled_pixels: set[tuple[int, int]] = set()
    object_depths: list[float] = []
    for sample_x, sample_y in sample_centers:
        pixel_x = int(round(sample_x))
        pixel_y = int(round(sample_y))
        patch_x1 = max(box_x1, pixel_x - DEPTH_PATCH_RADIUS_PX)
        patch_x2 = min(box_x2, pixel_x + DEPTH_PATCH_RADIUS_PX + 1)
        patch_y1 = max(box_y1, pixel_y - DEPTH_PATCH_RADIUS_PX)
        patch_y2 = min(box_y2, pixel_y + DEPTH_PATCH_RADIUS_PX + 1)
        for y in range(patch_y1, patch_y2):
            for x in range(patch_x1, patch_x2):
                if (x, y) in sampled_pixels:
                    continue
                sampled_pixels.add((x, y))
                value = float(depth_mm[y, x])
                if not np.isfinite(value) or value < MIN_DEPTH_MM:
                    continue
                if TABLE_DEPTH_MIN_MM <= value <= TABLE_DEPTH_MAX_MM:
                    continue
                if value > TABLE_DEPTH_MAX_MM:
                    continue
                object_depths.append(value)

    if len(object_depths) < MIN_OBJECT_DEPTH_PIXELS:
        return None
    return _nearest_depth_cluster_median(object_depths)


def _nearest_depth_cluster_median(depths_mm: list[float]) -> float | None:
    ordered_depths = sorted(depths_mm)
    for start, minimum_depth in enumerate(ordered_depths):
        cluster = [
            depth
            for depth in ordered_depths[start:]
            if depth - minimum_depth <= OBJECT_DEPTH_CLUSTER_SPREAD_MM
        ]
        if len(cluster) >= MIN_OBJECT_DEPTH_PIXELS:
            return float(np.median(cluster))
    return None


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
                try:
                    spatial_depth_mm = object_depth_from_box(depth_mm, box)
                    object_depth_mm = depth_stabilizer.update(
                        detected_object.object_id,
                        box,
                        spatial_depth_mm,
                    )
                    depth_text = f"{object_depth_mm:.0f} mm"
                except ValueError:
                    depth_text = "measuring depth"
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
