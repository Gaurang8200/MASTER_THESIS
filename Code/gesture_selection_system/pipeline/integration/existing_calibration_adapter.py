"""Adapter for the pixel to robot conversion of the existing pick and drop repository.

Calibration is only used when converting a fingertip pixel on the table into a
robot place coordinate. Gesture classification and fingertip to mask object
selection are pure image space operations and never reach this module.

This package deliberately does not implement its own camera calibration. The
``existing_repo`` adapter calls ``function_pool.calc_pixel2robot`` of the
existing repository, which is the same math ``pixel2robot.py`` uses. The
``mock`` adapter is a placeholder for development until that repository is
reachable from the runtime environment.
"""

from __future__ import annotations

import contextlib
import io
import logging
import sys

import numpy as np

from config import ExistingRepoCalibrationConfig, MockPlaceCalibrationConfig, PlaceCalibrationConfig
from schemas import RobotPose

LOGGER = logging.getLogger(__name__)


class PlaceCalibrationError(RuntimeError):
    """Raised when a place pixel cannot be converted into a robot pose."""


class MockPlaceCalibration:
    """Placeholder mapping used until the existing repository is connected.

    It is a plain linear pixel to table mapping. It is good enough to exercise
    the interaction and the JSON contract, and it is never a fallback during a
    real robot session, which is why the mode is reported in every result.
    """

    def __init__(self, config: MockPlaceCalibrationConfig) -> None:
        self._config = config

    @property
    def mode(self) -> str:
        return "mock"

    def start(self) -> None:
        LOGGER.warning(
            "place_calibration_is_mock: place poses are placeholders, "
            "switch place_calibration.mode to existing_repo for real coordinates"
        )

    def convert_place_pixel_to_robot_pose(
        self,
        fingertip_pixel: tuple[float, float],
        frame_shape: tuple[int, int] | None = None,
    ) -> RobotPose:
        scale_x, scale_y = self._config.meters_per_pixel
        origin_x, origin_y = self._config.origin_m
        roll, pitch, yaw = self._config.orientation_deg
        return RobotPose(
            x_m=origin_x + float(fingertip_pixel[0]) * scale_x,
            y_m=origin_y + float(fingertip_pixel[1]) * scale_y,
            z_m=self._config.place_z_m,
            rx_deg=roll,
            ry_deg=pitch,
            rz_deg=yaw,
        )

    def health(self) -> dict[str, object]:
        return {"component": "place_calibration", "mode": "mock", "placeholder": True}


class ExistingRepoCalibration:
    """Calls the pixel to robot conversion of the existing repository.

    The calibration was recorded at a fixed camera resolution, so a pixel from a
    different frame size is scaled into that resolution first. Loading happens in
    ``start`` because it touches the file system and imports third party code.
    """

    def __init__(self, config: ExistingRepoCalibrationConfig) -> None:
        self._config = config
        self._function_pool = None
        self._tvec = None
        self._rvec = None
        self._camera_matrix = None
        self._flange_to_camera = None
        self._base_to_flange = None

    @property
    def mode(self) -> str:
        return "existing_repo"

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

        # This is the connection point to the existing pick and drop repository.
        # No calibration math is reimplemented here.
        try:
            import function_pool
        except ImportError as exc:
            # function_pool pulls in scipy and sympy, so a missing dependency of
            # the existing repository surfaces here and not as a wrong pose.
            raise PlaceCalibrationError(
                f"could not import function_pool from {repo_dir}: {exc}"
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
            "place_calibration_loaded repo=%s poses=%d index=%d",
            repo_dir.name,
            available,
            self._config.pose_index,
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
                result, _transform, _rot, _trans, base_to_pixel, tool_offset = (
                    self._function_pool.calc_pixel2robot(
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
                )
        except Exception as exc:
            raise PlaceCalibrationError(f"pixel to robot transform failed: {exc}") from exc

        # The existing transform works in millimeters.
        x_m = float(result[0, 0]) / 1000.0
        y_m = float(result[1, 0]) / 1000.0
        z_m = float(result[2, 0]) / 1000.0 + self._config.place_z_offset_m
        roll, pitch, yaw = self._orientation(base_to_pixel, tool_offset)
        return RobotPose(x_m=x_m, y_m=y_m, z_m=z_m, rx_deg=roll, ry_deg=pitch, rz_deg=yaw)

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

    def _orientation(self, base_to_pixel, tool_offset) -> tuple[float, float, float]:
        """Tool orientation for the place point, taken from the existing transform.

        The existing repository derives the flange pose by removing the tool
        offset from the base to pixel transform and converting the rotation with
        its own helper. The same returned values are reused here so that no
        calibration math is duplicated.
        """
        try:
            flange = np.dot(np.asarray(base_to_pixel), np.linalg.inv(np.asarray(tool_offset)))
            roll, pitch, yaw = self._function_pool.rot2euler(flange[0:3, 0:3], degrees=True)
            return float(roll), float(pitch), float(yaw)
        except Exception:
            LOGGER.exception("place_orientation_failed, falling back to the configured orientation")
            return tuple(float(value) for value in self._config.fallback_orientation_deg)

    def health(self) -> dict[str, object]:
        return {
            "component": "place_calibration",
            "mode": "existing_repo",
            "loaded": self._function_pool is not None,
            "repo": str(self._config.repo_dir),
            "pose_index": self._config.pose_index,
        }


def build_place_calibration(config: PlaceCalibrationConfig):
    """Create the configured adapter without starting it."""
    if config.mode == "mock":
        return MockPlaceCalibration(config.mock)
    if config.mode == "existing_repo":
        return ExistingRepoCalibration(config.existing_repo)
    raise PlaceCalibrationError(f"unknown place calibration mode {config.mode}")


def convert_place_pixel_to_robot_pose(
    fingertip_pixel: tuple[float, float],
    calibration,
    frame_shape: tuple[int, int] | None = None,
) -> RobotPose:
    """Convert one place pixel into a robot pose through the configured adapter."""
    return calibration.convert_place_pixel_to_robot_pose(fingertip_pixel, frame_shape)


def calibration_available(config: PlaceCalibrationConfig) -> bool:
    """Cheap probe used by the runner to report the calibration state at startup."""
    if config.mode == "mock":
        return True
    repo = config.existing_repo
    required = (
        repo.repo_dir / repo.wp2camera_json,
        repo.repo_dir / repo.c2f_json,
        repo.repo_dir / repo.robot_poses_json,
    )
    return all(path.is_file() for path in required)
