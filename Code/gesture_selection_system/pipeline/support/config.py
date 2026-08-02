"""Typed configuration for the gesture selection system.

The YAML file is the single source of truth for thresholds, stability windows,
calibration and camera settings. Loading it through Pydantic means a wrong
threshold fails at startup instead of during a live robot session.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
_COMMON_DIR = PIPELINE_ROOT.parent / "common"
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))

from device import resolve_device  # noqa: E402,F401  re-exported for the detector
from gesture_classes import CLASS_IDS, GestureName  # noqa: E402
from paths import resolve_under  # noqa: E402

DEFAULT_CONFIG_PATH = PIPELINE_ROOT / "configs" / "gesture_config.yaml"


def resolve_path(value: str | Path) -> Path:
    """Resolve a configured path against the pipeline folder when it is relative."""
    return resolve_under(PIPELINE_ROOT, value)


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    weights: str = "models/best.pt"
    device: str = "auto"
    imgsz: int = Field(default=640, ge=64, le=4096)
    half: bool = False
    max_detections: int = Field(default=20, ge=1, le=300)
    latency_budget_ms: float = Field(default=120.0, gt=0.0)

    @property
    def weights_path(self) -> Path:
        return resolve_path(self.weights)


class ConfidenceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gesture: float = Field(default=0.75, gt=0.0, le=1.0)
    fingertip: float = Field(default=0.65, gt=0.0, le=1.0)
    object: float = Field(default=0.70, gt=0.0, le=1.0)

    def threshold_for(self, gesture: GestureName) -> float:
        if gesture is GestureName.INDEX_FINGERTIP:
            return self.fingertip
        return self.gesture

    @property
    def detector_floor(self) -> float:
        """Lowest score worth asking the model for, filtered per class later."""
        return min(self.gesture, self.fingertip)


class StabilityConfig(BaseModel):
    """How long a gesture has to be held before it counts.

    Seconds rather than frame counts, so the behaviour is the same on a fast
    machine and on a slow one.
    """

    model_config = ConfigDict(extra="forbid")

    activate_seconds: float = Field(default=5.0, gt=0.0, le=60.0)
    deactivate_seconds: float = Field(default=5.0, gt=0.0, le=60.0)
    select_seconds: float = Field(default=5.0, gt=0.0, le=60.0)
    place_seconds: float = Field(default=5.0, gt=0.0, le=60.0)
    lost_gesture_timeout_seconds: float = Field(default=3.0, ge=0.0, le=120.0)


class SelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fingertip_radius_px: int = Field(default=8, ge=0, le=200)
    min_mask_overlap_px: int = Field(default=1, ge=1)
    require_pointing_finger: bool = True
    hold_selection_until_mode_off: bool = True
    require_selection_before_place: bool = True
    # A place point is any spot inside the table area. Whether an object already
    # lies there does not block it.
    ignore_objects_for_place: bool = True


class WorkspaceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_polygon: list[tuple[float, float]] = Field(min_length=3)

    @field_validator("normalized_polygon")
    @classmethod
    def _within_unit_square(cls, value: list[tuple[float, float]]) -> list[tuple[float, float]]:
        for x, y in value:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"workspace point {x}, {y} is outside the normalized image")
        return value

    def to_pixels(self, width: int, height: int) -> np.ndarray:
        """Return the calibrated table polygon in pixels for a given frame size."""
        return np.array(
            [(x * width, y * height) for x, y in self.normalized_polygon],
            dtype=np.float64,
        )


class MockPlaceCalibrationConfig(BaseModel):
    """Placeholder mapping used until the existing repository is connected."""

    model_config = ConfigDict(extra="forbid")

    origin_m: tuple[float, float] = (0.30, -0.25)
    meters_per_pixel: tuple[float, float] = (0.0009, 0.0009)
    place_z_m: float = 0.05
    orientation_deg: tuple[float, float, float] = (180.0, 0.0, 0.0)

    @field_validator("meters_per_pixel")
    @classmethod
    def _positive_scale(cls, value: tuple[float, float]) -> tuple[float, float]:
        if value[0] <= 0.0 or value[1] <= 0.0:
            raise ValueError("meters_per_pixel must be positive on both axes")
        return value


class ExistingRepoCalibrationConfig(BaseModel):
    """Location of the calibration data of the existing pick and drop repository."""

    model_config = ConfigDict(extra="forbid")

    repo_path: str = "../Object-Detection-Using-YOLO-v5-main 2"
    wp2camera_json: str = "output_wp2camera.json"
    c2f_json: str = "output_c2f.json"
    robot_poses_json: str = "robot_poses.json"
    pose_index: int = Field(default=15, ge=0)
    calibration_resolution: tuple[int, int] = (2560, 1472)
    place_z_offset_m: float = 0.0
    fallback_orientation_deg: tuple[float, float, float] = (180.0, 0.0, 0.0)

    @property
    def repo_dir(self) -> Path:
        return resolve_path(self.repo_path)


class PlaceCalibrationConfig(BaseModel):
    """Calibration is used for the place point only, never for gesture logic."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["mock", "existing_repo"] = "mock"
    mock: MockPlaceCalibrationConfig = Field(default_factory=MockPlaceCalibrationConfig)
    existing_repo: ExistingRepoCalibrationConfig = Field(
        default_factory=ExistingRepoCalibrationConfig
    )


class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(default=0, ge=0)
    width: int = Field(default=1280, ge=64)
    height: int = Field(default=720, ge=64)
    flip_horizontal: bool = True


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    show_masks: bool = True
    mask_alpha: float = Field(default=0.35, ge=0.0, le=1.0)
    show_workspace: bool = True
    show_hud: bool = True
    fingertip_marker_radius_px: int = Field(default=10, ge=1)


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"


class GestureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: ModelConfig = Field(default_factory=ModelConfig)
    class_ids: dict[str, int]
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    stability: StabilityConfig = Field(default_factory=StabilityConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    workspace: WorkspaceConfig
    place_calibration: PlaceCalibrationConfig = Field(default_factory=PlaceCalibrationConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _check_class_ids(self) -> "GestureConfig":
        # The runtime mapping has to match the order the model was trained on,
        # otherwise a detected open palm would be read as a fingertip.
        if self.class_ids != CLASS_IDS:
            raise ValueError(
                f"class_ids {self.class_ids} do not match the training classes {CLASS_IDS}"
            )
        return self

    def gesture_by_class_id(self) -> dict[int, GestureName]:
        return {index: GestureName(name) for name, index in self.class_ids.items()}


def load_config(path: str | Path | None = None) -> GestureConfig:
    """Load and validate the runtime configuration."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"gesture configuration not found at {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return GestureConfig.model_validate(payload)
