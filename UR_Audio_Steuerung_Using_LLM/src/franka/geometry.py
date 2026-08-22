# MO_Changes
from __future__ import annotations

import json
import math
from typing import Sequence

import numpy as np

from .calibration import _euler_zyx_matrix
from .config import FrankaConfig
from .models import CartesianPose


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> tuple[float, float, float, float]:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            w = (matrix[2, 1] - matrix[1, 2]) / scale
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            w = (matrix[0, 2] - matrix[2, 0]) / scale
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            w = (matrix[1, 0] - matrix[0, 1]) / scale
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
    return CartesianPose.create((0.0, 0.0, 0.0), (x, y, z, w)).quaternion


def rotation_vector_to_quaternion(vector: Sequence[float]) -> tuple[float, float, float, float]:
    values = np.asarray(tuple(float(value) for value in vector), dtype=float)
    if values.shape != (3,):
        raise ValueError("robot rotation vector requires three values")
    angle = float(np.linalg.norm(values))
    if angle < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    axis = values / angle
    sine = math.sin(angle / 2.0)
    return CartesianPose.create(
        (0.0, 0.0, 0.0),
        (axis[0] * sine, axis[1] * sine, axis[2] * sine, math.cos(angle / 2.0)),
    ).quaternion


def calibration_pose_15(config: FrankaConfig) -> CartesianPose:
    pose_data = json.loads(
        (config.calibration_dir / "robot_poses.json").read_text(encoding="utf-8")
    )["Posen"]["p15"]
    rotation = _euler_zyx_matrix((pose_data["a"], pose_data["b"], pose_data["c"]))
    return CartesianPose.create(
        (
            float(pose_data["x"]) / 1000.0,
            float(pose_data["y"]) / 1000.0,
            float(pose_data["z"]) / 1000.0,
        ),
        rotation_matrix_to_quaternion(rotation),
    )
