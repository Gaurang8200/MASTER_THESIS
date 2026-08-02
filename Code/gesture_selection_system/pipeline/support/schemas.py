"""Shared contracts for the gesture selection system.

Internal state uses dataclasses because it is cheap and never validated twice.
Everything that leaves the pipeline as JSON uses Pydantic so that a malformed
result fails at the boundary instead of inside the robot pipeline.

Coordinate rule of this package. Gesture detection and object selection stay in
image coordinates. Robot coordinates appear in exactly one place, the place
pose, which is produced by the calibration of the existing pick and drop
repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gesture_classes import GestureName

SCHEMA_VERSION = "1.1"


class SelectionMode(str, Enum):
    OFF = "off"
    ON = "on"


class ModeTransition(str, Enum):
    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"


class InteractionMode(str, Enum):
    """What the operator has achieved in the current frame."""

    IDLE = "idle"
    OBJECT_SELECTION = "object_selection"
    PLACE_SELECTION = "place_selection"


@dataclass(frozen=True)
class BoundingBox:
    """Axis aligned box in pixel coordinates of the processed frame."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"invalid bounding box ordering: {self}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def clipped(self, width: int, height: int) -> "BoundingBox":
        return BoundingBox(
            x1=float(min(max(self.x1, 0.0), width)),
            y1=float(min(max(self.y1, 0.0), height)),
            x2=float(min(max(self.x2, 0.0), width)),
            y2=float(min(max(self.y2, 0.0), height)),
        )

    def as_int_tuple(self) -> tuple[int, int, int, int]:
        return int(round(self.x1)), int(round(self.y1)), int(round(self.x2)), int(round(self.y2))


@dataclass(frozen=True)
class GestureDetection:
    gesture: GestureName
    confidence: float
    box: BoundingBox


@dataclass(frozen=True)
class GestureFrame:
    """Result of one detector call, entirely in image coordinates.

    An unhealthy frame keeps ``ok`` false so that downstream policy can fail
    closed instead of acting on an empty detection list that looks valid.
    """

    frame_index: int
    detections: tuple[GestureDetection, ...] = ()
    latency_ms: float = 0.0
    ok: bool = True
    error: str | None = None

    def all_of(self, gesture: GestureName) -> list[GestureDetection]:
        return [d for d in self.detections if d.gesture is gesture]

    def best(self, gesture: GestureName) -> GestureDetection | None:
        candidates = self.all_of(gesture)
        if not candidates:
            return None
        return max(candidates, key=lambda d: d.confidence)

    def has(self, gesture: GestureName) -> bool:
        return any(d.gesture is gesture for d in self.detections)


@dataclass(frozen=True)
class RobotPose:
    """Cartesian pose in the robot base frame, in the form UR script expects.

    Position is in meters and orientation is a rotation vector in radians, which
    is what ``movel(p[...])`` takes and what the existing repository already
    writes in ``final_position``.
    """

    x_m: float
    y_m: float
    z_m: float
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    frame: str = "robot_base"

    def as_list(self) -> list[float]:
        return [self.x_m, self.y_m, self.z_m, self.rx, self.ry, self.rz]


@dataclass
class DetectedObject:
    """One object reported by the detector of the existing pick and drop system.

    The box is in image coordinates. That detector produces boxes and no
    segmentation masks, so selection works on box containment.
    """

    object_id: str
    class_name: str
    confidence: float
    box: BoundingBox


@runtime_checkable
class GestureSource(Protocol):
    """Any gesture detector the pipeline can run on, real or faked in tests."""

    def start(self) -> None: ...

    def detect(self, frame: np.ndarray, frame_index: int) -> GestureFrame: ...

    def close(self) -> None: ...

    def health(self) -> dict[str, object]: ...


@runtime_checkable
class ObjectSource(Protocol):
    """Object detection boundary.

    The YOLOv5 adapter of the existing repository satisfies this, and it is the
    only object contract the selection logic depends on.
    """

    def start(self) -> None: ...

    def get_objects(self, frame: np.ndarray) -> Sequence[DetectedObject]: ...

    def close(self) -> None: ...

    def health(self) -> dict[str, object]: ...


@runtime_checkable
class PlaceCalibration(Protocol):
    """Pixel to robot conversion, used for the place point and nothing else."""

    def start(self) -> None: ...

    def convert_place_pixel_to_robot_pose(
        self, fingertip_pixel: tuple[float, float], frame_shape: tuple[int, int]
    ) -> RobotPose: ...

    def health(self) -> dict[str, object]: ...


class PointModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float
    y: float


class BoxModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_box(cls, box: BoundingBox) -> "BoxModel":
        return cls(x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2)


class PoseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x_m: float
    y_m: float
    z_m: float
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    frame: str = "robot_base"

    @classmethod
    def from_pose(cls, pose: RobotPose) -> "PoseModel":
        return cls(
            x_m=pose.x_m,
            y_m=pose.y_m,
            z_m=pose.z_m,
            rx=pose.rx,
            ry=pose.ry,
            rz=pose.rz,
            frame=pose.frame,
        )


class FingertipModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    center_px: PointModel
    confidence: float
    inside_workspace: bool
    pointing_finger_present: bool


class SelectedObjectModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    class_name: str
    confidence: float
    bbox: BoxModel
    centroid_px: PointModel
    held_s: float


class PlacePointModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pixel: PointModel
    pose: PoseModel
    held_s: float


class LatencyModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gesture_ms: float
    objects_ms: float
    total_ms: float
    budget_ms: float
    within_budget: bool


class PipelineOutput(BaseModel):
    """The structured result of one processed frame.

    ``calibration_used`` tells the caller whether a robot coordinate is part of
    this result. It is false for every object selection, because that decision
    is made purely in image coordinates.

    This module never sends a command, so ``safe_to_execute`` is advice for the
    caller and not a promise that a motion was started.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    mode: InteractionMode = InteractionMode.IDLE
    frame_index: int
    frame_width: int
    frame_height: int
    timestamp_monotonic_s: float
    selection_mode: SelectionMode
    mode_transition: ModeTransition | None = None
    calibration_used: bool = False
    fingertip_pixel: list[float] | None = None
    selected_object_id: str | None = None
    place_pixel: list[float] | None = None
    place_robot_pose: list[float] | None = None
    fingertip: FingertipModel | None = None
    selected_object: SelectedObjectModel | None = None
    place_point: PlacePointModel | None = None
    candidate_object_id: str | None = None
    latency: LatencyModel
    degraded: bool = False
    safe_to_execute: bool = False
    robot_command_dispatched: bool = False
    notes: list[str] = Field(default_factory=list)

    def to_json(self, indent: int | None = 2) -> str:
        return self.model_dump_json(indent=indent)
