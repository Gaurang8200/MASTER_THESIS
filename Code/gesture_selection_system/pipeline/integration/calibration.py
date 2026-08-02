"""Pixel to robot conversion of the existing pick and drop repository.

Calibration is used in one place only, turning a fingertip pixel on the table
into a robot place coordinate. Gesture classification and object selection are
pure image space operations and never reach this module.

No calibration math is implemented here. This calls
``function_pool.calc_pixel2robot`` of the existing repository, which is the same
function ``pixel2robot.py`` uses. That script cannot be imported because it reads
and writes text files at import time, so the function is called directly.

Only x and y come out of the transform, which is what ``pixel2robot.py`` writes
to ``robot_coordinates.txt``. The release height and the tool orientation are set
by the existing ``final_position`` and are configured here, not derived.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys

import numpy as np

from config import PlaceCalibrationConfig
from schemas import RobotPose

LOGGER = logging.getLogger(__name__)


class PlaceCalibrationError(RuntimeError):
    """Raised when a place pixel cannot be converted into a robot pose."""


class ExistingRepoCalibration:
    """Calls the pixel to robot conversion of the existing repository.

    The calibration was recorded at a fixed camera resolution, so a pixel from a
    different frame size is scaled into that resolution first. Loading happens in
    ``start`` because it touches the file system and imports third party code.
    """

    def __init__(self, config: PlaceCalibrationConfig) -> None:
        self._config = config
        self._function_pool = None
        self._tvec = None
        self._rvec = None
        self._camera_matrix = None
        self._flange_to_camera = None
        self._base_to_flange = None

    @property
    def is_started(self) -> bool:
        return self._function_pool is not None

    def start(self) -> None:
        if self._function_pool is not None:
            return

        repo_dir = self._config.repo_dir
        if not repo_dir.is_dir():
            raise PlaceCalibrationError(f"calibration repository not found at {repo_dir}")

        files = {
            "wp2camera": repo_dir / self._config.wp2camera_json,
            "c2f": repo_dir / self._config.c2f_json,
            "robot_poses": repo_dir / self._config.robot_poses_json,
        }
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise PlaceCalibrationError(f"missing calibration files: {missing}")

        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        try:
            import function_pool
        except ImportError as exc:
            # function_pool pulls in scipy and sympy, so a missing dependency of
            # the existing repository surfaces here and not as a wrong pose.
            raise PlaceCalibrationError(
                f"could not import function_pool from {repo_dir}: {exc}. "
                "Install its dependencies with pip install scipy sympy"
            ) from exc

        try:
            tvec, rvec, camera_matrix, _dist = function_pool.read_wp2c(str(files["wp2camera"]))
            flange_to_camera = function_pool.read_c2f(str(files["c2f"]))
            base_to_flange, _ = function_pool.read_bTf(str(files["robot_poses"]))
        except Exception as exc:
            raise PlaceCalibrationError(f"could not read calibration data: {exc}") from exc

        available = min(len(tvec), len(rvec), len(base_to_flange))
        if self._config.pose_index >= available:
            raise PlaceCalibrationError(
                f"pose_index {self._config.pose_index} is out of range, "
                f"the calibration holds {available} poses"
            )

        self._function_pool = function_pool
        self._tvec = tvec
        self._rvec = rvec
        self._camera_matrix = camera_matrix
        self._flange_to_camera = flange_to_camera
        self._base_to_flange = base_to_flange
        LOGGER.info(
            "place_calibration_loaded repo poses index=%d resolution=%dx%d",
            available,
            self._config.pose_index,
            *self._config.calibration_resolution,
        )

    def convert_place_pixel_to_robot_pose(
        self,
        fingertip_pixel: tuple[float, float],
        frame_shape: tuple[int, int] | None = None,
    ) -> RobotPose:
        if self._function_pool is None:
            raise PlaceCalibrationError("ExistingRepoCalibration.start must be called first")

        pixel_coords = self._to_calibration_pixel(fingertip_pixel, frame_shape)

        try:
            # The vendored function prints its intermediate results, which would
            # flood the interaction loop.
            with contextlib.redirect_stdout(io.StringIO()):
                result, *_ = self._function_pool.calc_pixel2robot(
                    self._tvec,
                    self._rvec,
                    self._camera_matrix,
                    self._base_to_flange,
                    self._flange_to_camera,
                    pixel_coords,
                    self._config.pose_index,
                    None,
                    False,
                )
        except Exception as exc:
            raise PlaceCalibrationError(f"pixel to robot transform failed: {exc}") from exc

        # The existing transform works in millimeters and only x and y are used,
        # which is exactly what pixel2robot.py writes to robot_coordinates.txt.
        # The release height and the tool orientation are not part of the
        # transform. The existing final_position sets both itself, so they come
        # from the configuration here and are never derived from the pixel.
        rx, ry, rz = self._config.place_orientation
        return RobotPose(
            x_m=float(result[0, 0]) / 1000.0,
            y_m=float(result[1, 0]) / 1000.0,
            z_m=self._config.place_z_m,
            rx=rx,
            ry=ry,
            rz=rz,
        )

    def _to_calibration_pixel(
        self, fingertip_pixel: tuple[float, float], frame_shape: tuple[int, int] | None
    ) -> np.ndarray:
        x_px = float(fingertip_pixel[0])
        y_px = float(fingertip_pixel[1])
        if frame_shape is not None:
            height, width = frame_shape[0], frame_shape[1]
            if width <= 0 or height <= 0:
                raise PlaceCalibrationError(f"invalid frame shape {frame_shape}")
            target_width, target_height = self._config.calibration_resolution
            x_px = x_px * target_width / float(width)
            y_px = y_px * target_height / float(height)
        return np.array([[x_px], [y_px], [1.0]], dtype=np.float64)

    def health(self) -> dict[str, object]:
        return {
            "component": "place_calibration",
            "loaded": self._function_pool is not None,
            "repo": str(self._config.repo_dir),
            "pose_index": self._config.pose_index,
            "calibration_resolution": list(self._config.calibration_resolution),
        }


def calibration_available(config: PlaceCalibrationConfig) -> bool:
    """Cheap probe used by the runner to report the calibration state at startup."""
    required = (
        config.repo_dir / config.wp2camera_json,
        config.repo_dir / config.c2f_json,
        config.repo_dir / config.robot_poses_json,
    )
    return all(path.is_file() for path in required)
