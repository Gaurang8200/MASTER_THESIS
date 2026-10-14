from __future__ import annotations

import math
import sys
from collections import deque
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
from src.FR_franka.FR_config import load_franka_config
from src.FR_franka.FR_models import CartesianPose
from src.FR_franka.FR_robot import FrankaRobotArm
from src.camera_devices import RgbCameraStream


FRAME_SIZE = (1280, 720)
YOLO_ROOT = AUDIO_ROOT / "Code-YOLOv5-Windows_llm" / "yolov5"
STABLE_ANGLE_FRAMES = 7
SUPPORTED_CLASSES = ("cylinder", "box", "marker")


def estimate_object_angle(
    crop: np.ndarray,
) -> tuple[
    float,
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    float,
]:
    if crop.size == 0:
        raise ValueError("object crop is empty")
    saturation = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
    saturation = cv2.GaussianBlur(saturation, (5, 5), 0)
    _, mask = cv2.threshold(
        saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("object contour is unavailable")
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < crop.shape[0] * crop.shape[1] * 0.05:
        raise ValueError("object contour is too small")
    object_mask = np.zeros_like(mask)
    cv2.drawContours(object_mask, [contour], -1, 255, cv2.FILLED)
    point_y, point_x = np.nonzero(object_mask)
    points = np.column_stack((point_x, point_y)).astype(np.float64)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean=None)
    if eigenvalues[1, 0] <= 0.0:
        raise ValueError("object PCA has no minor axis")
    elongation = float(eigenvalues[0, 0] / eigenvalues[1, 0])
    if elongation < 1.5:
        raise ValueError("object is not elongated enough for a stable 2D angle")
    axis_x, axis_y = eigenvectors[0]
    if axis_x < 0.0:
        axis_x, axis_y = -axis_x, -axis_y
    angle_degrees = math.degrees(math.atan2(-axis_y, axis_x))
    center = mean[0]
    projections = (points - center) @ np.array((axis_x, axis_y))
    tail = center + projections.min() * np.array((axis_x, axis_y))
    nose = center + projections.max() * np.array((axis_x, axis_y))
    return (
        angle_degrees,
        (float(center[0]), float(center[1])),
        (float(tail[0]), float(tail[1])),
        (float(nose[0]), float(nose[1])),
        elongation,
    )


def stable_axis_angle_degrees(angles: deque[float]) -> float:
    doubled_angles = [math.radians(2.0 * angle) for angle in angles]
    return math.degrees(
        0.5
        * math.atan2(
            sum(math.sin(angle) for angle in doubled_angles),
            sum(math.cos(angle) for angle in doubled_angles),
        )
    )


def quaternion_multiply(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x1, y1, z1, w1 = first
    x2, y2, z2, w2 = second
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def z_rotation_quaternion(angle_degrees: float) -> tuple[float, float, float, float]:
    half_angle = math.radians(angle_degrees) / 2.0
    return 0.0, 0.0, math.sin(half_angle), math.cos(half_angle)


def quaternion_yaw_degrees(quaternion: tuple[float, float, float, float]) -> float:
    x, y, z, w = quaternion
    return math.degrees(
        math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    )


def create_robot() -> FrankaRobotArm:
    config = load_franka_config()
    robot = FrankaRobotArm(
        config.robot_ip,
        config.dynamics_factor,
        config.gripper_speed_mm_s,
        config.gripper_force,
    )
    robot.start()
    return robot


def rotate_gripper_about_z(
    robot: FrankaRobotArm,
    start_pose: CartesianPose,
    object_name: str,
    object_angle_degrees: float,
) -> None:
    current_pose = robot.current_pose()
    rotation_degrees = object_angle_degrees + 90.0
    target_quaternion = quaternion_multiply(
        z_rotation_quaternion(rotation_degrees),
        start_pose.quaternion,
    )
    target_pose = CartesianPose.create(
        current_pose.translation,
        target_quaternion,
    )
    print(
        f"ROBOT: {object_name}={object_angle_degrees:+.2f} deg, "
        f"perpendicular offset=+90.00 deg, "
        f"target Z yaw={quaternion_yaw_degrees(target_pose.quaternion):+.2f} deg at fixed "
        f"XYZ={tuple(round(value, 2) for value in current_pose.translation)} mm"
    )
    robot.move_pose(target_pose)
    print("ROBOT: Z-axis rotation complete")


def restore_start_orientation(robot: FrankaRobotArm, start_pose: CartesianPose) -> None:
    current_pose = robot.current_pose()
    robot.move_pose(CartesianPose.create(current_pose.translation, start_pose.quaternion))
    print(
        f"ROBOT: Original Z yaw restored to "
        f"{quaternion_yaw_degrees(start_pose.quaternion):+.2f} deg"
    )


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
    robot = None
    try:
        robot = create_robot()
        start_pose = robot.current_pose()
        print(
            f"ROBOT: Saved original Z yaw="
            f"{quaternion_yaw_degrees(start_pose.quaternion):+.2f} deg"
        )
        detector.start()
        camera.start()
        print("Object orientation test: R rotates, O restores the start angle, Q quits.")
        angle_histories = {
            name: deque(maxlen=STABLE_ANGLE_FRAMES) for name in SUPPORTED_CLASSES
        }
        geometry_histories = {
            name: deque(maxlen=STABLE_ANGLE_FRAMES) for name in SUPPORTED_CLASSES
        }
        while True:
            frame = camera.read()
            if frame is None:
                continue
            display = frame.copy()
            stable_targets = []
            objects = [
                detected
                for detected in detector.get_objects(frame)
                if detected.class_name.lower() in SUPPORTED_CLASSES
            ]
            class_counts = {
                name: sum(
                    detected.class_name.lower() == name for detected in objects
                )
                for name in SUPPORTED_CLASSES
            }
            for name in SUPPORTED_CLASSES:
                if class_counts[name] != 1:
                    angle_histories[name].clear()
                    geometry_histories[name].clear()
            for detected in objects:
                object_name = detected.class_name
                class_name = object_name.lower()
                x1, y1, x2, y2 = detected.box.as_int_tuple()
                if class_counts[class_name] != 1:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    continue
                crop = frame[y1:y2, x1:x2]
                try:
                    angle, local_center, local_tail, local_nose, elongation = (
                        estimate_object_angle(crop)
                    )
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
                    angle_histories[class_name].clear()
                    geometry_histories[class_name].clear()
                    continue
                angle_history = angle_histories[class_name]
                geometry_history = geometry_histories[class_name]
                angle_history.append(angle)
                geometry_history.append(
                    (
                        x1 + local_center[0],
                        y1 + local_center[1],
                        x1 + local_tail[0],
                        y1 + local_tail[1],
                        x1 + local_nose[0],
                        y1 + local_nose[1],
                    )
                )
                measured_angle = None
                if len(angle_history) == STABLE_ANGLE_FRAMES:
                    measured_angle = stable_axis_angle_degrees(angle_history)
                    stable_targets.append((object_name, measured_angle))
                geometry = np.mean(np.asarray(geometry_history), axis=0)
                center = (int(round(geometry[0])), int(round(geometry[1])))
                start = (int(round(geometry[2])), int(round(geometry[3])))
                end = (int(round(geometry[4])), int(round(geometry[5])))
                display_angle = measured_angle if measured_angle is not None else angle
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(display, start, end, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.arrowedLine(display, center, end, (255, 255, 0), 3, cv2.LINE_AA)
                cv2.circle(display, center, 5, (255, 0, 255), -1)
                cv2.putText(
                    display,
                    f"{object_name} PCA {display_angle:+.1f} deg  "
                    f"target {display_angle + 90.0:+.1f} deg  ratio {elongation:.1f}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            cv2.putText(
                display,
                "R: rotate gripper   O: original angle   Q: quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Three object 2D PCA orientation", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("r"), ord("R")):
                if len(stable_targets) != 1:
                    print(
                        "ROBOT: Rotation requires exactly one stable Cylinder, Box, or Marker"
                    )
                    continue
                try:
                    object_name, measured_angle = stable_targets[0]
                    rotate_gripper_about_z(
                        robot, start_pose, object_name, measured_angle
                    )
                except Exception as error:
                    print(f"ROBOT: Z-axis rotation failed: {error}")
            if key in (ord("o"), ord("O")):
                try:
                    restore_start_orientation(robot, start_pose)
                    for history in (*angle_histories.values(), *geometry_histories.values()):
                        history.clear()
                except Exception as error:
                    print(f"ROBOT: Original orientation restore failed: {error}")
    finally:
        cv2.destroyAllWindows()
        detector.close()
        camera.close()
        if robot is not None:
            robot.close()


if __name__ == "__main__":
    main()
