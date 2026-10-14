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
    cylinder_mask = np.zeros_like(mask)
    cv2.drawContours(cylinder_mask, [contour], -1, 255, cv2.FILLED)
    point_y, point_x = np.nonzero(cylinder_mask)
    points = np.column_stack((point_x, point_y)).astype(np.float64)
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean=None)
    if eigenvalues[1, 0] <= 0.0:
        raise ValueError("cylinder PCA has no minor axis")
    elongation = float(eigenvalues[0, 0] / eigenvalues[1, 0])
    if elongation < 1.5:
        raise ValueError("cylinder is not elongated enough for a stable 2D angle")
    axis_x, axis_y = eigenvectors[0]
    if axis_x < 0.0:
        axis_x, axis_y = -axis_x, -axis_y
    angle_degrees = math.degrees(math.atan2(-axis_y, axis_x))
    return angle_degrees, (float(mean[0, 0]), float(mean[0, 1])), elongation


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
    cylinder_angle_degrees: float,
) -> None:
    current_pose = robot.current_pose()
    rotation_degrees = cylinder_angle_degrees + 90.0
    target_quaternion = quaternion_multiply(
        z_rotation_quaternion(rotation_degrees),
        start_pose.quaternion,
    )
    target_pose = CartesianPose.create(
        current_pose.translation,
        target_quaternion,
    )
    print(
        f"ROBOT: Cylinder={cylinder_angle_degrees:+.2f} deg, perpendicular offset=+90.00 deg, "
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
        print("Cylinder orientation test: R rotates, O restores the start angle, Q quits.")
        angle_history: deque[float] = deque(maxlen=STABLE_ANGLE_FRAMES)
        while True:
            frame = camera.read()
            if frame is None:
                continue
            display = frame.copy()
            measured_angle = None
            cylinders = [
                detected
                for detected in detector.get_objects(frame)
                if detected.class_name.lower() == "cylinder"
            ]
            if len(cylinders) != 1:
                angle_history.clear()
            for detected in cylinders[:1]:
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
                    angle_history.clear()
                    continue
                angle_history.append(angle)
                if len(angle_history) == STABLE_ANGLE_FRAMES:
                    measured_angle = stable_axis_angle_degrees(angle_history)
                center = (
                    int(round(x1 + local_center[0])),
                    int(round(y1 + local_center[1])),
                )
                display_angle = measured_angle if measured_angle is not None else angle
                radians = math.radians(display_angle)
                axis_length = max(30, int(0.4 * max(x2 - x1, y2 - y1)))
                delta = (
                    int(round(axis_length * math.cos(radians))),
                    int(round(-axis_length * math.sin(radians))),
                )
                start = (center[0] - delta[0], center[1] - delta[1])
                end = (center[0] + delta[0], center[1] + delta[1])
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.line(display, start, end, (0, 255, 255), 3, cv2.LINE_AA)
                cv2.arrowedLine(display, center, end, (255, 255, 0), 3, cv2.LINE_AA)
                cv2.circle(display, center, 5, (255, 0, 255), -1)
                cv2.putText(
                    display,
                    f"PCA {display_angle:+.1f} deg  target offset {display_angle + 90.0:+.1f} deg  ratio {elongation:.1f}",
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
            cv2.imshow("Cylinder 2D PCA orientation", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("r"), ord("R")):
                if measured_angle is None:
                    print("ROBOT: Rotation skipped until seven valid angle frames are available")
                    continue
                try:
                    rotate_gripper_about_z(robot, start_pose, measured_angle)
                except Exception as error:
                    print(f"ROBOT: Z-axis rotation failed: {error}")
            if key in (ord("o"), ord("O")):
                try:
                    restore_start_orientation(robot, start_pose)
                    angle_history.clear()
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
