# MO_Changes
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .FR_models import CartesianPose


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "src" / "FR_config" / "franka_robot.json"


@dataclass(frozen=True)
class FrankaConfig:
    robot_ip: str
    dynamics_factor: float
    gripper_speed: float
    gripper_force: float
    home_joints: tuple[float, ...]
    intermediate_joints: tuple[float, ...]
    approach_height: float
    camera_offset: tuple[float, float, float]
    lift_height: float
    pick_heights: dict[int, float]
    place_heights: dict[int, float]
    default_orientation: tuple[float, float, float, float]
    calibration_width: int
    calibration_height: int
    mirror_x: bool
    workspace_x: tuple[float, float]
    workspace_y: tuple[float, float]
    workspace_z: tuple[float, float]
    zones: dict[str, CartesianPose]

    def pick_height(self, object_class: int) -> float:
        if object_class not in self.pick_heights:
            raise ValueError(f"No Franka pick height exists for object class {object_class}")
        return self.pick_heights[object_class]

    def place_height(self, object_class: int) -> float:
        if object_class not in self.place_heights:
            raise ValueError(f"No Franka place height exists for object class {object_class}")
        return self.place_heights[object_class]

    def zone(self, name: str) -> CartesianPose:
        if name not in self.zones:
            raise ValueError(
                f"Franka zone {name} has not been taught in {DEFAULT_CONFIG_PATH}"
            )
        return self.zones[name]


def _float_tuple(value: Any, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{name} requires {size} values")
    return tuple(float(item) for item in value)


def load_franka_config(path: Path = DEFAULT_CONFIG_PATH) -> FrankaConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    zones = {
        name: CartesianPose.create(value["translation"], value["quaternion"])
        for name, value in data.get("zones", {}).items()
    }
    config = FrankaConfig(
        robot_ip=str(data["robot_ip"]),
        dynamics_factor=float(data["dynamics_factor"]),
        gripper_speed=float(data["gripper_speed"]),
        gripper_force=float(data["gripper_force"]),
        home_joints=_float_tuple(data["home_joints"], 7, "home_joints"),
        intermediate_joints=_float_tuple(
            data["intermediate_joints"], 7, "intermediate_joints"
        ),
        approach_height=float(data["approach_height"]),
        camera_offset=_float_tuple(data["camera_offset"], 3, "camera_offset"),
        lift_height=float(data["lift_height"]),
        pick_heights={int(key): float(value) for key, value in data["pick_heights"].items()},
        place_heights={
            int(key): float(value) for key, value in data["place_heights"].items()
        },
        default_orientation=_float_tuple(
            data["default_orientation"], 4, "default_orientation"
        ),
        calibration_width=int(data["calibration_image_size"][0]),
        calibration_height=int(data["calibration_image_size"][1]),
        mirror_x=bool(data["mirror_x"]),
        workspace_x=_float_tuple(data["workspace"]["x"], 2, "workspace.x"),
        workspace_y=_float_tuple(data["workspace"]["y"], 2, "workspace.y"),
        workspace_z=_float_tuple(data["workspace"]["z"], 2, "workspace.z"),
        zones=zones,
    )
    _validate_config(config)
    return config


def _validate_config(config: FrankaConfig) -> None:
    if not config.robot_ip.strip():
        raise ValueError("robot_ip must not be empty")
    if not 0.0 < config.dynamics_factor <= 1.0:
        raise ValueError("dynamics_factor must be between zero and one")
    if config.calibration_width <= 0 or config.calibration_height <= 0:
        raise ValueError("calibration image dimensions must be positive")
    for lower, upper in (config.workspace_x, config.workspace_y, config.workspace_z):
        if lower >= upper:
            raise ValueError("workspace lower limit must be below its upper limit")
    for name, pose in config.zones.items():
        x, y, z = pose.translation
        if not config.workspace_x[0] <= x <= config.workspace_x[1]:
            raise ValueError(f"Franka zone {name} x is outside the workspace")
        if not config.workspace_y[0] <= y <= config.workspace_y[1]:
            raise ValueError(f"Franka zone {name} y is outside the workspace")
        if not config.workspace_z[0] <= z <= config.workspace_z[1]:
            raise ValueError(f"Franka zone {name} z is outside the workspace")
    for object_class, height in config.place_heights.items():
        if not config.workspace_z[0] <= height <= config.workspace_z[1]:
            raise ValueError(
                f"Franka place height for class {object_class} is outside the workspace"
            )
