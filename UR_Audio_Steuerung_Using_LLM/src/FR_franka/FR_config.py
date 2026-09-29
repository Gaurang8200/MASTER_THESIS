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
    gripper_speed_mm_s: float
    gripper_force: float
    home_joints: tuple[float, ...]
    intermediate_joints: tuple[float, ...]
    approach_height_mm: float
    camera_offset_mm: tuple[float, float, float]
    lift_height_mm: float
    table_surface_z_mm: float
    grip_offset_below_surface_mm: float
    place_heights_mm: dict[int, float]
    default_orientation: tuple[float, float, float, float]
    calibration_directory: Path
    calibration_width: int
    calibration_height: int
    mirror_x: bool
    workspace_x_mm: tuple[float, float]
    workspace_y_mm: tuple[float, float]
    workspace_z_mm: tuple[float, float]
    depth_bbox_inset_ratio: float
    depth_table_ring_scale: float
    depth_minimum_valid_pixels: int
    depth_table_tolerance_mm: float
    depth_maximum_height_mm: float
    zones: dict[str, CartesianPose]

    def place_height_mm(self, object_class: int) -> float:
        if object_class not in self.place_heights_mm:
            raise ValueError(f"No Franka place height exists for object class {object_class}")
        return self.place_heights_mm[object_class]

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
    if data.get("linear_unit") != "mm":
        raise ValueError("Franka configuration linear_unit must be mm")
    calibration_directory = Path(str(data["calibration_directory"]))
    if not calibration_directory.is_absolute():
        calibration_directory = PROJECT_ROOT / "src" / "FR_franka" / calibration_directory
    zones = {
        name: CartesianPose.create(value["translation_mm"], value["quaternion"])
        for name, value in data.get("zones", {}).items()
    }
    config = FrankaConfig(
        robot_ip=str(data["robot_ip"]),
        dynamics_factor=float(data["dynamics_factor"]),
        gripper_speed_mm_s=float(data["gripper_speed_mm_s"]),
        gripper_force=float(data["gripper_force"]),
        home_joints=_float_tuple(data["home_joints"], 7, "home_joints"),
        intermediate_joints=_float_tuple(
            data["intermediate_joints"], 7, "intermediate_joints"
        ),
        approach_height_mm=float(data["approach_height_mm"]),
        camera_offset_mm=_float_tuple(
            data["camera_offset_mm"], 3, "camera_offset_mm"
        ),
        lift_height_mm=float(data["lift_height_mm"]),
        table_surface_z_mm=float(data["table_surface_z_mm"]),
        grip_offset_below_surface_mm=float(data["grip_offset_below_surface_mm"]),
        place_heights_mm={
            int(key): float(value) for key, value in data["place_heights_mm"].items()
        },
        default_orientation=_float_tuple(
            data["default_orientation"], 4, "default_orientation"
        ),
        calibration_directory=calibration_directory,
        calibration_width=int(data["calibration_image_size"][0]),
        calibration_height=int(data["calibration_image_size"][1]),
        mirror_x=bool(data["mirror_x"]),
        workspace_x_mm=_float_tuple(
            data["workspace_mm"]["x"], 2, "workspace_mm.x"
        ),
        workspace_y_mm=_float_tuple(
            data["workspace_mm"]["y"], 2, "workspace_mm.y"
        ),
        workspace_z_mm=_float_tuple(
            data["workspace_mm"]["z"], 2, "workspace_mm.z"
        ),
        depth_bbox_inset_ratio=float(data["depth"]["bbox_inset_ratio"]),
        depth_table_ring_scale=float(data["depth"]["table_ring_scale"]),
        depth_minimum_valid_pixels=int(data["depth"]["minimum_valid_pixels"]),
        depth_table_tolerance_mm=float(data["depth"]["table_depth_tolerance_mm"]),
        depth_maximum_height_mm=float(data["depth"]["maximum_height_mm"]),
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
    if not 0.0 <= config.depth_bbox_inset_ratio < 0.5:
        raise ValueError("depth bbox inset ratio must be below one half")
    if config.depth_table_ring_scale <= 1.0:
        raise ValueError("depth table ring scale must be greater than one")
    if config.depth_minimum_valid_pixels <= 0:
        raise ValueError("depth minimum valid pixels must be positive")
    if config.depth_table_tolerance_mm <= 0.0:
        raise ValueError("depth table tolerance must be positive")
    if config.gripper_speed_mm_s <= 0.0:
        raise ValueError("gripper speed must be positive")
    if config.grip_offset_below_surface_mm <= 0.0:
        raise ValueError("grip offset below surface must be positive")
    if config.depth_maximum_height_mm <= 0.0:
        raise ValueError("depth maximum height must be positive")
    for lower, upper in (
        config.workspace_x_mm,
        config.workspace_y_mm,
        config.workspace_z_mm,
    ):
        if lower >= upper:
            raise ValueError("workspace lower limit must be below its upper limit")
    if not (
        config.workspace_z_mm[0]
        <= config.table_surface_z_mm
        <= config.workspace_z_mm[1]
    ):
        raise ValueError("table surface Z is outside the workspace")
    for name, pose in config.zones.items():
        x, y, z = pose.translation
        if not config.workspace_x_mm[0] <= x <= config.workspace_x_mm[1]:
            raise ValueError(f"Franka zone {name} x is outside the workspace")
        if not config.workspace_y_mm[0] <= y <= config.workspace_y_mm[1]:
            raise ValueError(f"Franka zone {name} y is outside the workspace")
        if not config.workspace_z_mm[0] <= z <= config.workspace_z_mm[1]:
            raise ValueError(f"Franka zone {name} z is outside the workspace")
    for object_class, height in config.place_heights_mm.items():
        if not config.workspace_z_mm[0] <= height <= config.workspace_z_mm[1]:
            raise ValueError(
                f"Franka place height for class {object_class} is outside the workspace"
            )
