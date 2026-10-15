# MO_Changes
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .FR_models import CartesianPose


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


def rotate_quaternion_about_base_z(
    quaternion: Sequence[float],
    angle_degrees: float,
) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in quaternion)
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError("robot orientation requires four finite quaternion values")
    if not math.isfinite(angle_degrees):
        raise ValueError("object angle must be finite")
    x, y, z, w = values
    half_angle = math.radians(angle_degrees) / 2.0
    rotation_z = math.sin(half_angle)
    rotation_w = math.cos(half_angle)
    rotated = (
        rotation_w * x - rotation_z * y,
        rotation_w * y + rotation_z * x,
        rotation_w * z + rotation_z * w,
        rotation_w * w - rotation_z * z,
    )
    return CartesianPose.create((0.0, 0.0, 0.0), rotated).quaternion
