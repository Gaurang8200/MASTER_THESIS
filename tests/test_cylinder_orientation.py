from __future__ import annotations

import math
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
from src.camera_devices import RgbCameraStream


FRAME_SIZE = (1280, 720)
YOLO_ROOT = AUDIO_ROOT / "Code-YOLOv5-Windows_llm" / "yolov5"


def estimate_cylinder_angle(crop: np.ndarray) -> tuple[float, tuple[float, float], float]:
    if crop.size == 0:
        raise ValueError("cylinder crop is empty")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("cylinder contour is unavailable")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < crop.shape[0] * crop.shape[1] * 0.05:
        raise ValueError("cylinder contour is too small")
    points = contour.reshape(-1, 2).astype(np.float64)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean=None)
    if eigenvalues[1, 0] <= 0.0:
        raise ValueError("cylinder PCA has no minor axis")
    elongation = float(eigenvalues[0, 0] / eigenvalues[1, 0])
    if elongation < 1.5:
        raise ValueError("cylinder is not elongated enough for a stable 2D angle")
    axis_x, axis_y = eigenvectors[0]
    angle_degrees = math.degrees(math.atan2(-axis_y, axis_x))
    angle_degrees = (angle_degrees + 90.0) % 180.0 - 90.0
    return angle_degrees, (float(mean[0, 0]), float(mean[0, 1])), elongation


def downward_gripper_quaternion(yaw_degrees: float) -> tuple[float, float, float, float]:
    half_yaw = math.radians(yaw_degrees) / 2.0
    return math.cos(half_yaw), math.sin(half_yaw), 0.0, 0.0


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
    camera = RgbCameraStream(*FRAME_SIZE, color_only=True)
    try:
        detector.start()
        camera.start()
        print("Cylinder orientation test: X, Y and Z stay fixed. Only gripper Z yaw changes.")
        while True:
            frame = camera.read()
            if frame is None:
                continue
            display = frame.copy()
            for detected in detector.get_objects(frame):
                if detected.class_name.lower() != "cylinder":
                    continue
                x1, y1, x2, y2 = detected.box.as_int_tuple()
                crop = frame[y1:y2, x1:x2]
                try:
                    angle, local_center, elongation = estimate_cylinder_angle(crop)
                except ValueError as error:
                    cv2.putText(
                        display,
                        str(error),
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    continue
                center = (
                    int(round(x1 + local_center[0])),
                    int(round(y1 + local_center[1])),
                )
                radians = math.radians(angle)
                axis_length = max(30, int(0.4 * max(x2 - x1, y2 - y1)))
                delta = (
                    int(round(axis_length * math.cos(radians))),
                    int(round(-axis_length * math.sin(radians))),
                )
                start = (center[0] - delta[0], center[1] - delta[1])
                end = (center[0] + delta[0], center[1] + delta[1])
                quaternion = downward_gripper_quaternion(angle)
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(display, start, end, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.circle(display, center, 5, (255, 0, 255), -1)
                cv2.putText(
                    display,
                    f"PCA {angle:+.1f} deg  Z yaw {angle:+.1f} deg  ratio {elongation:.1f}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                print(
                    f"PCA angle={angle:+.2f} deg, gripper Z yaw={angle:+.2f} deg, "
                    f"XYZ unchanged, quaternion={tuple(round(value, 5) for value in quaternion)}"
                )
            cv2.imshow("Cylinder 2D PCA orientation", display)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cv2.destroyAllWindows()
        detector.close()
        camera.close()


if __name__ == "__main__":
    main()
