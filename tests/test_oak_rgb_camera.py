from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = REPOSITORY_ROOT / "UR_Audio_Steuerung_Using_LLM"
PIPELINE_ROOT = REPOSITORY_ROOT / "Code" / "gesture_selection_system" / "pipeline"
for folder in (AUDIO_ROOT, PIPELINE_ROOT / "support", PIPELINE_ROOT / "detection"):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from config import ObjectModelConfig, resolve_device
from object_detector import Yolov5ObjectDetector
from src.camera_devices import RgbCameraStream, median_depth_for_bbox


FRAME_SIZE = (1280, 720)
YOLO_ROOT = AUDIO_ROOT / "Code-YOLOv5-Windows_llm" / "yolov5"


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
    camera = RgbCameraStream(*FRAME_SIZE)

    try:
        detector.start()
        camera.start()
        size_printed = False

        while True:
            rgbd_frame = camera.read_rgbd()
            if rgbd_frame is None:
                continue
            frame = rgbd_frame.color
            height, width = frame.shape[:2]
            if (width, height) != FRAME_SIZE:
                raise RuntimeError(
                    f"OAK D returned {width} x {height}, expected 1280 x 720"
                )
            if not size_printed:
                print(f"OAK D RGB and aligned depth: {width} x {height}")
                size_printed = True

            display = frame.copy()
            for detected_object in detector.get_objects(frame):
                box = detected_object.box.as_int_tuple()
                try:
                    depth_mm = median_depth_for_bbox(rgbd_frame.depth_mm, box)
                    depth_text = f"{depth_mm:.0f} mm"
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
