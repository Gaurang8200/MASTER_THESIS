# MO_Changes
from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np

from .FR_geometry import rotation_matrix_to_quaternion
from .FR_models import CartesianPose, PixelPoint, RobotPoint
from .FR_original_calibration import FR_function_pool as function_pool, pixel2robot


_CALIBRATION_LOCK = threading.Lock()
REQUIRED_CALIBRATION_FILES = (
    "output_wp2camera.json",
    "output_c2f.json",
    "robot_poses.json",
)


@contextmanager
def _calibration_working_directory(calibration_directory: Path) -> Iterator[None]:
    previous_directory = Path.cwd()
    os.chdir(calibration_directory)
    try:
        yield
    finally:
        os.chdir(previous_directory)


class OriginalFrankaPixelTransformer:
    def __init__(
        self,
        calibration_size: tuple[int, int],
        mirror_x: bool,
        calibration_directory: Path,
        pose_index: int = 14,
    ) -> None:
        width, height = calibration_size
        if width <= 0 or height <= 0:
            raise ValueError("calibration image dimensions must be positive")
        if pose_index < 0:
            raise ValueError("calibration pose index must not be negative")
        self._calibration_size = calibration_size
        self._mirror_x = mirror_x
        self._pose_index = pose_index
        self._calibration_directory = calibration_directory.resolve()
        self._validate_calibration_files()

    def transform(
        self,
        pixel: PixelPoint,
        source_size: tuple[int, int],
    ) -> RobotPoint:
        width, height = source_size
        if width <= 0 or height <= 0:
            raise ValueError("source image dimensions must be positive")
        scaled_x = pixel.x * self._calibration_size[0] / width
        scaled_y = pixel.y * self._calibration_size[1] / height
        if self._mirror_x:
            scaled_x = self._calibration_size[0] - scaled_x
        if not 0.0 <= scaled_x < self._calibration_size[0]:
            raise ValueError("pixel x is outside the calibrated image")
        if not 0.0 <= scaled_y < self._calibration_size[1]:
            raise ValueError("pixel y is outside the calibrated image")
        with _CALIBRATION_LOCK, _calibration_working_directory(
            self._calibration_directory
        ):
            x_robot, y_robot, z_robot = pixel2robot(
                scaled_x,
                scaled_y,
                self._pose_index,
            )
        return RobotPoint(float(x_robot), float(y_robot), float(z_robot))

    def calibration_pose(self) -> CartesianPose:
        with _CALIBRATION_LOCK, _calibration_working_directory(
            self._calibration_directory
        ):
            transforms, _ = function_pool.read_bTf("robot_poses.json")
        if self._pose_index >= len(transforms):
            raise ValueError(f"calibration pose {self._pose_index} does not exist")
        transform = np.asarray(transforms[self._pose_index], dtype=float)
        return CartesianPose.create(
            transform[:3, 3],
            rotation_matrix_to_quaternion(transform[:3, :3]),
        )

    def _validate_calibration_files(self) -> None:
        missing = [
            name
            for name in REQUIRED_CALIBRATION_FILES
            if not (self._calibration_directory / name).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Dedicated OAK D calibration files are missing from "
                f"{self._calibration_directory}: {missing}"
            )
