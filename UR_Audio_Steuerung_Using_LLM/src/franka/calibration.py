# MO_Changes
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .models import PixelPoint, RobotPoint


def _euler_zyx_matrix(angles: Sequence[float]) -> np.ndarray:
    alpha, beta, gamma = (float(value) for value in angles)
    rz = np.array(
        [
            [math.cos(alpha), -math.sin(alpha), 0.0],
            [math.sin(alpha), math.cos(alpha), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    ry = np.array(
        [
            [math.cos(beta), 0.0, math.sin(beta)],
            [0.0, 1.0, 0.0],
            [-math.sin(beta), 0.0, math.cos(beta)],
        ]
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(gamma), -math.sin(gamma)],
            [0.0, math.sin(gamma), math.cos(gamma)],
        ]
    )
    return rz @ ry @ rx


class FrankaPixelTransformer:
    def __init__(
        self,
        calibration_dir: Path,
        calibration_size: tuple[int, int],
        mirror_x: bool,
    ) -> None:
        self._calibration_dir = calibration_dir
        self._calibration_size = calibration_size
        self._mirror_x = mirror_x
        self._camera_matrix: np.ndarray
        self._distortion: np.ndarray
        self._flange_to_camera: np.ndarray
        self._plane_point_mm: np.ndarray
        self._plane_normal: np.ndarray
        self._load()

    def transform(
        self,
        pixel: PixelPoint,
        source_size: tuple[int, int],
        base_to_end_effector_m: np.ndarray,
    ) -> RobotPoint:
        width, height = source_size
        if width <= 0 or height <= 0:
            raise ValueError("source image dimensions must be positive")
        if base_to_end_effector_m.shape != (4, 4):
            raise ValueError("base to end effector transform must be four by four")
        scaled_x = pixel.x * self._calibration_size[0] / width
        scaled_y = pixel.y * self._calibration_size[1] / height
        if self._mirror_x:
            scaled_x = self._calibration_size[0] - scaled_x
        if not 0.0 <= scaled_x < self._calibration_size[0]:
            raise ValueError("pixel x is outside the calibrated image")
        if not 0.0 <= scaled_y < self._calibration_size[1]:
            raise ValueError("pixel y is outside the calibrated image")

        undistorted = cv2.undistortPoints(
            np.array([[[scaled_x, scaled_y]]], dtype=np.float64),
            self._camera_matrix,
            self._distortion,
        )[0, 0]
        ray_camera = np.array([undistorted[0], undistorted[1], 1.0], dtype=float)

        base_to_end_effector_mm = np.asarray(base_to_end_effector_m, dtype=float).copy()
        base_to_end_effector_mm[:3, 3] *= 1000.0
        base_to_camera = base_to_end_effector_mm @ self._flange_to_camera
        ray_origin = base_to_camera[:3, 3]
        ray_direction = base_to_camera[:3, :3] @ ray_camera
        denominator = float(np.dot(self._plane_normal, ray_direction))
        if abs(denominator) < 1e-9:
            raise ValueError("camera ray is parallel to the calibrated table")
        distance = float(
            np.dot(self._plane_normal, self._plane_point_mm - ray_origin) / denominator
        )
        if distance <= 0.0:
            raise ValueError("calibrated table is behind the camera")
        point_mm = ray_origin + distance * ray_direction
        return RobotPoint(*(point_mm / 1000.0))

    def _load(self) -> None:
        camera_path = self._calibration_dir / "output_wp2camera.json"
        camera_to_flange_path = self._calibration_dir / "output_c2f.json"
        poses_path = self._calibration_dir / "robot_poses.json"
        for path in (camera_path, camera_to_flange_path, poses_path):
            if not path.is_file():
                raise FileNotFoundError(f"Missing Franka calibration file {path}")

        camera_data = json.loads(camera_path.read_text(encoding="utf-8"))
        flange_data = json.loads(camera_to_flange_path.read_text(encoding="utf-8"))
        pose_data = json.loads(poses_path.read_text(encoding="utf-8"))["Posen"]
        self._camera_matrix = np.asarray(camera_data["camera_matrix"], dtype=float)
        self._distortion = np.asarray(camera_data["dist_coefs"], dtype=float)
        self._flange_to_camera = np.asarray(flange_data["fTc"], dtype=float)
        if self._camera_matrix.shape != (3, 3):
            raise ValueError("Franka camera matrix must be three by three")
        if self._flange_to_camera.shape != (4, 4):
            raise ValueError("Franka flange to camera matrix must be four by four")

        plane_transforms: list[np.ndarray] = []
        for index in range(len(pose_data)):
            pose = pose_data[f"p{index}"]
            base_to_flange = np.eye(4, dtype=float)
            base_to_flange[:3, :3] = _euler_zyx_matrix(
                (pose["a"], pose["b"], pose["c"])
            )
            base_to_flange[:3, 3] = (pose["x"], pose["y"], pose["z"])
            rotation_vector = np.asarray(
                camera_data["rotational_vectors"][f"image{index}"], dtype=float
            )
            translation_vector = np.asarray(
                camera_data["translational_vectors"][f"image{index}"], dtype=float
            ).reshape(3)
            camera_to_plane = np.eye(4, dtype=float)
            camera_to_plane[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
            camera_to_plane[:3, 3] = translation_vector
            plane_transforms.append(
                base_to_flange @ self._flange_to_camera @ camera_to_plane
            )

        origins = np.asarray([transform[:3, 3] for transform in plane_transforms])
        normals = np.asarray([transform[:3, 2] for transform in plane_transforms])
        reference = normals[0]
        normals = np.asarray(
            [normal if np.dot(normal, reference) >= 0.0 else -normal for normal in normals]
        )
        mean_normal = normals.mean(axis=0)
        self._plane_normal = mean_normal / np.linalg.norm(mean_normal)
        self._plane_point_mm = origins.mean(axis=0)
        if float(np.max(np.std(origins, axis=0))) > 10.0:
            raise ValueError("Franka table calibration observations are inconsistent")
        if float(np.max(np.std(normals, axis=0))) > 0.05:
            raise ValueError("Franka table normal observations are inconsistent")
