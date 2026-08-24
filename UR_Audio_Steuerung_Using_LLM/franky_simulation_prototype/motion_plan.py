# MO_Changes
from __future__ import annotations

from dataclasses import dataclass

from src.franka.config import FrankaConfig
from src.franka.models import CartesianPose

from .scene import SceneSpec


@dataclass(frozen=True)
class PrototypeMotionPlan:
    home_joints: tuple[float, ...]
    camera_approach: CartesianPose
    pick_pose: CartesianPose
    lift_pose: CartesianPose
    intermediate_joints: tuple[float, ...]
    target_pose: CartesianPose


def build_motion_plan(
    config: FrankaConfig,
    scene: SceneSpec,
) -> PrototypeMotionPlan:
    zone = config.zone(scene.target_zone)
    offset_x, offset_y, offset_z = config.camera_offset
    orientation = config.default_orientation

    camera_approach = CartesianPose.create(
        (
            scene.object_x + offset_x,
            scene.object_y + offset_y,
            config.approach_height + offset_z,
        ),
        orientation,
    )
    pick_pose = CartesianPose.create(
        (
            scene.object_x,
            scene.object_y,
            config.pick_height(scene.object_class),
        ),
        orientation,
    )
    lift_pose = CartesianPose.create(
        (scene.object_x, scene.object_y, config.lift_height),
        orientation,
    )
    target_pose = CartesianPose.create(
        (
            zone.translation[0],
            zone.translation[1],
            config.place_height(scene.object_class),
        ),
        zone.quaternion,
    )
    return PrototypeMotionPlan(
        home_joints=config.home_joints,
        camera_approach=camera_approach,
        pick_pose=pick_pose,
        lift_pose=lift_pose,
        intermediate_joints=config.intermediate_joints,
        target_pose=target_pose,
    )
