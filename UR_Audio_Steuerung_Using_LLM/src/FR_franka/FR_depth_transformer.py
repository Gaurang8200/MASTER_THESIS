from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..camera_devices import MIN_DEPTH_MM, MIN_OBJECT_DEPTH_PIXELS
from .FR_models import PixelPoint


@dataclass(frozen=True)
class CalibratedDepth:
    object_depth_mm: float
    camera_point_mm: tuple[float, float, float]
    robot_base_z_mm: float
    sample_count: int

class FrankaDepthTransformer:
    def __init__(
        self,
        calibration_size: tuple[int, int],
        mirror_x: bool,
        calibration_directory: Path,
    ) -> None:
        width, height = calibration_size
        if width <= 0 or height <= 0:
            raise ValueError("calibration image dimensions must be positive")
        self._calibration_size = calibration_size
        self._mirror_x = mirror_x
        self._camera_matrix, self._distortion = _read_camera_calibration(
            calibration_directory / "output_wp2camera.json"
        )
        self._flange_from_camera = _read_transform(
            calibration_directory / "output_c2f.json",
            "fTc",
        )

    def transform(
        self,
        pixel: PixelPoint,
        source_size: tuple[int, int],
        depth_mm: float,
        base_from_flange: np.ndarray,
        sample_count: int,
    ) -> CalibratedDepth:
        if not np.isfinite(depth_mm) or depth_mm < MIN_DEPTH_MM:
            raise ValueError(f"OAK D object depth {depth_mm:.1f} mm is invalid")
        if sample_count < MIN_OBJECT_DEPTH_PIXELS:
            raise ValueError("OAK D object depth has too few valid samples")
        checked_base_from_flange = np.asarray(base_from_flange, dtype=float)
        if checked_base_from_flange.shape != (4, 4):
            raise ValueError("Robot base to flange transform must be 4 by 4")
        if not np.all(np.isfinite(checked_base_from_flange)):
            raise ValueError("Robot base to flange transform contains invalid values")

        calibration_pixel = self._scale_pixel(pixel, source_size)
        normalized = cv2.undistortPoints(
            np.array([[[calibration_pixel.x, calibration_pixel.y]]], dtype=np.float64),
            self._camera_matrix,
            self._distortion,
        )[0, 0]
        camera_point = np.array(
            [normalized[0] * depth_mm, normalized[1] * depth_mm, depth_mm, 1.0],
            dtype=float,
        )
        robot_point = checked_base_from_flange @ self._flange_from_camera @ camera_point
        if not np.all(np.isfinite(robot_point[:3])):
            raise ValueError("Calibrated robot point contains invalid values")
        return CalibratedDepth(
            object_depth_mm=float(depth_mm),
            camera_point_mm=tuple(float(value) for value in camera_point[:3]),
            robot_base_z_mm=float(robot_point[2]),
            sample_count=sample_count,
        )

    def _scale_pixel(
        self,
        pixel: PixelPoint,
        source_size: tuple[int, int],
    ) -> PixelPoint:
        width, height = source_size
        if width <= 0 or height <= 0:
            raise ValueError("source image dimensions must be positive")
        x = pixel.x * self._calibration_size[0] / width
        y = pixel.y * self._calibration_size[1] / height
        if self._mirror_x:
            x = self._calibration_size[0] - x
        if not 0.0 <= x < self._calibration_size[0]:
            raise ValueError("left mono pixel x is outside the calibrated image")
        if not 0.0 <= y < self._calibration_size[1]:
            raise ValueError("left mono pixel y is outside the calibrated image")
        return PixelPoint(x, y)

def _read_camera_calibration(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = _read_json(path)
    camera_matrix = np.asarray(data.get("camera_matrix"), dtype=float)
    distortion = np.asarray(data.get("dist_coefs"), dtype=float)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"Camera matrix in {path} must be 3 by 3")
    if distortion.size < 4:
        raise ValueError(f"Distortion coefficients in {path} are incomplete")
    return camera_matrix, distortion


def _read_transform(path: Path, key: str) -> np.ndarray:
    transform = np.asarray(_read_json(path).get(key), dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"Transform {key} in {path} must be a valid 4 by 4 matrix")
    return transform


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Calibration file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Calibration file {path} must contain a JSON object")
    return data
