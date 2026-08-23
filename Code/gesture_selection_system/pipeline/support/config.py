# MO_Changes
"""Typed configuration for the gesture selection system.

The YAML file is the single source of truth for thresholds, stability windows,
calibration and camera settings. Loading it through Pydantic means a wrong
threshold fails at startup instead of during a live robot session.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gesture_classes import GestureName

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PIPELINE_ROOT / "configs" / "gesture_config.yaml"


def resolve_path(value: str | Path) -> Path:
    """Resolve a configured path against the pipeline folder when it is relative."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (PIPELINE_ROOT / path).resolve()


def resolve_device(requested: str) -> str:
    """Pick an inference device, falling back to cpu when nothing else is usable."""
    if requested != "auto":
        return requested
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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

    # The existing detector produces boxes and no masks, so a fingertip selects
    # an object by sitting inside its box. Where boxes overlap the smallest one
    # wins, which picks the object the operator is actually on.
    # Setting a ratio also requires the fingertip to be that close to the box
    # centre, measured in half box widths. Leave it empty to accept the box.
    max_center_distance_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
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


class PlaceCalibrationConfig(BaseModel):
    """Calibration data of the existing pick and drop repository.

    Used for the place point only. Gesture classification and object selection
    stay in image coordinates and never read this section.
    """

    model_config = ConfigDict(extra="forbid")

    repo_path: str = "../../Object-Detection-Using-YOLO-v5-main 2"
    wp2camera_json: str = "output_wp2camera.json"
    c2f_json: str = "output_c2f.json"
    robot_poses_json: str = "robot_poses.json"
    pose_index: int = Field(default=15, ge=0)
    calibration_resolution: tuple[int, int] = (2560, 1472)
    # Height the object is released at. The existing program picks this per
    # object type and the values it uses sit between 0.040 and 0.067.
    place_z_m: float = 0.05
    # Fixed tool orientation for a place, a rotation vector in radians. Same
    # value the existing final_position uses.
    place_orientation: tuple[float, float, float] = (2.221, 2.221, 0.0)

    @property
    def repo_dir(self) -> Path:
        return resolve_path(self.repo_path)


class ObjectModelConfig(BaseModel):
    """YOLOv5 detector of the existing pick and drop repository."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    repo_path: str = "../../Object-Detection-Using-YOLO-v5-main 2/yolov5"
    weights: str = "../../Object-Detection-Using-YOLO-v5-main 2/yolov5/my_model.pt"
    imgsz: int = Field(default=640, ge=64, le=4096)
    # Overlap needed to keep an object id from one frame to the next.
    track_min_iou: float = Field(default=0.5, gt=0.0, le=1.0)

    @property
    def repo_dir(self) -> Path:
        return resolve_path(self.repo_path)

    @property
    def weights_path(self) -> Path:
        return resolve_path(self.weights)


class RobotConfig(BaseModel):
    """Connection to the UR controller of the existing pick and drop system."""

    model_config = ConfigDict(extra="forbid")

    host: str = "10.84.59.207"
    port: int = Field(default=30002, ge=1, le=65535)
    # Motion stays off until it is switched on deliberately.
    dispatch: bool = False
    acceleration: float = Field(default=0.1, gt=0.0, le=2.0)
    velocity: float = Field(default=0.1, gt=0.0, le=2.0)
    timeout_s: float = Field(default=3.0, gt=0.0, le=60.0)
    # File the existing pipeline reads the picked pixel from. Its detection.py
    # writes the same file, so a pick handed over this way replaces that step.
    center_point_file: str = (
        "../../Object-Detection-Using-YOLO-v5-main 2/txt_file/center_point.txt"
    )

    # File the confirmed place coordinate is written to, in the same shape
    # pixel2robot.py writes robot_coordinates.txt.
    place_coordinates_file: str = (
        "../../Object-Detection-Using-YOLO-v5-main 2/txt_file/place_coordinates.txt"
    )

    @property
    def center_point_path(self) -> Path:
        return resolve_path(self.center_point_file)

    @property
    def place_coordinates_path(self) -> Path:
        return resolve_path(self.place_coordinates_file)


class CameraConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(default=0, ge=0)
    width: int = Field(default=1280, ge=64)
    height: int = Field(default=720, ge=64)
    flip_horizontal: bool = True


class VisualizationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    show_workspace: bool = True
    show_hud: bool = True
    fingertip_marker_radius_px: int = Field(default=10, ge=1)
    show_object_boxes: bool = True


class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: str = "INFO"


class GestureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: ModelConfig = Field(default_factory=ModelConfig)
    object_model: ObjectModelConfig = Field(default_factory=ObjectModelConfig)
    robot: RobotConfig = Field(default_factory=RobotConfig)
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
        known = {gesture.value for gesture in GestureName}
        configured = set(self.class_ids)
        unknown = configured.difference(known)
        required = {
            GestureName.POINTING_FINGER.value,
            GestureName.INDEX_FINGERTIP.value,
        }
        missing = required.difference(configured)
        indexes = list(self.class_ids.values())
        if unknown:
            raise ValueError(f"unknown gesture classes: {sorted(unknown)}")
        if missing:
            raise ValueError(f"required gesture classes are missing: {sorted(missing)}")
        if len(indexes) != len(set(indexes)) or any(index < 0 for index in indexes):
            raise ValueError("class_ids must use unique nonnegative indexes")
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
