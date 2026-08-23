# MO_Changes
from __future__ import annotations

from collections.abc import Callable

from src.franka.config import load_franka_config
from src.franka.robot import RobotArm

from .motion_plan import PrototypeMotionPlan, build_motion_plan
from .scene import SceneSpec
from .simulation_arm import FrankySimulationArm


OutputCallback = Callable[[str], None]


def execute_motion_plan(
    arm: RobotArm,
    plan: PrototypeMotionPlan,
    output: OutputCallback = print,
) -> None:
    arm.start()
    if not arm.health():
        arm.close()
        raise RuntimeError("Franky simulation health check failed")
    try:
        output("FRANKY SIMULATION STEP 1: Move to the home joints")
        arm.move_joints(plan.home_joints)

        output("FRANKY SIMULATION STEP 2: Move to the camera approach pose")
        arm.move_pose(plan.camera_approach)

        output("FRANKY SIMULATION STEP 3: Move to the cylinder")
        arm.move_pose(plan.pick_pose)

        output("FRANKY SIMULATION STEP 4: Close the gripper")
        arm.grip()

        output("FRANKY SIMULATION STEP 5: Lift the cylinder")
        arm.move_pose(plan.lift_pose)

        output("FRANKY SIMULATION STEP 6: Move through the intermediate joints")
        arm.move_joints(plan.intermediate_joints)

        output("FRANKY SIMULATION STEP 7: Move to the target zone")
        arm.move_pose(plan.target_pose)

        output("FRANKY SIMULATION STEP 8: Open the gripper")
        arm.release()

        output("FRANKY SIMULATION STEP 9: Return to the home joints")
        arm.move_joints(plan.home_joints)
    finally:
        arm.close()


def run_demo(render: bool = True) -> None:
    config = load_franka_config()
    scene = SceneSpec()
    plan = build_motion_plan(config, scene)
    arm = FrankySimulationArm(config, scene, render=render)
    execute_motion_plan(arm, plan)
