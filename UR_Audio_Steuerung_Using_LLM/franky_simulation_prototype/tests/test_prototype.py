# MO_Changes
from __future__ import annotations

import unittest
from collections.abc import Sequence
from importlib.util import find_spec
from xml.etree import ElementTree as ET

from src.franka.config import load_franka_config
from src.franka.models import CartesianPose
from src.franka.robot import FrankaRobotArm, RobotArm

from franky_simulation_prototype.demo import execute_motion_plan
from franky_simulation_prototype.motion_plan import build_motion_plan
from franky_simulation_prototype.scene import SceneSpec, add_workcell
from franky_simulation_prototype.scene_preview import run_scene_preview
from franky_simulation_prototype.simulation_arm import FrankySimulationArm


class RecordingArm(RobotArm):
    def __init__(self) -> None:
        self.events: list[tuple[str, tuple[float, ...]]] = []
        self.started = False

    def start(self) -> None:
        self.started = True
        self.events.append(("start", ()))

    def health(self) -> bool:
        return self.started

    def move_joints(self, joints: Sequence[float]) -> None:
        self.events.append(("joints", tuple(joints)))

    def move_pose(self, pose: CartesianPose) -> None:
        self.events.append(("pose", pose.translation + pose.quaternion))

    def current_pose(self) -> CartesianPose:
        return CartesianPose.create((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

    def grip(self) -> None:
        self.events.append(("grip", ()))

    def release(self) -> None:
        self.events.append(("release", ()))

    def close(self) -> None:
        self.started = False
        self.events.append(("close", ()))


class FrankySimulationPrototypeTest(unittest.TestCase):
    def test_motion_plan_reuses_current_franka_values(self) -> None:
        config = load_franka_config()
        plan = build_motion_plan(config, SceneSpec())

        self.assertEqual(plan.home_joints, config.home_joints)
        for actual, expected in zip(
            plan.camera_approach.translation,
            (0.484, 0.2, 0.5),
        ):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(plan.pick_pose.translation, (0.4, 0.2, 0.05))
        self.assertEqual(plan.lift_pose.translation, (0.4, 0.2, 0.2))
        self.assertEqual(plan.target_pose.translation, (0.366, -0.146, 0.067))

    def test_scene_contains_a_dynamic_cylinder_and_target_zone(self) -> None:
        worldbody = ET.Element("worldbody")
        add_workcell(worldbody, SceneSpec(), 0.366, -0.146)

        cylinder = worldbody.find("./body[@name='prototype_object']")
        self.assertIsNotNone(cylinder)
        self.assertIsNotNone(cylinder.find("./freejoint"))
        self.assertEqual(cylinder.find("./geom").attrib["type"], "cylinder")

        zone = worldbody.find("./body[@name='prototype_target_zone']")
        self.assertIsNotNone(zone)
        self.assertIsNone(zone.find("./freejoint"))

    def test_simulation_adapter_reuses_existing_franka_movements(self) -> None:
        self.assertTrue(issubclass(FrankySimulationArm, FrankaRobotArm))

    @unittest.skipUnless(find_spec("franky_sim"), "franky_sim is not installed")
    def test_mujoco_builds_the_robot_and_isolated_workcell(self) -> None:
        import mujoco
        from franky_sim import SimulationServer
        from franky_sim.mujoco_simulator import MujocoSimulator

        config = load_franka_config()
        scene = SceneSpec()
        zone = config.zone(scene.target_zone)

        with MujocoSimulator(enable_visualization=False) as simulator:
            add_workcell(
                simulator.worldbody,
                scene,
                zone.translation[0],
                zone.translation[1],
            )
            simulator.add_robot(initial_q=config.home_joints)
            with SimulationServer(simulator) as server:
                server.run_once(realtime=False)
                object_id = mujoco.mj_name2id(
                    simulator.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    "prototype_object",
                )
                robot_id = mujoco.mj_name2id(
                    simulator.model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    "robot0_fr3_link0",
                )

        self.assertGreaterEqual(object_id, 0)
        self.assertGreaterEqual(robot_id, 0)

    @unittest.skipUnless(find_spec("franky_sim"), "franky_sim is not installed")
    def test_headless_scene_preview_advances_physics(self) -> None:
        run_scene_preview(render=False, steps=1)

    def test_demo_executes_the_complete_isolated_sequence(self) -> None:
        arm = RecordingArm()
        plan = build_motion_plan(load_franka_config(), SceneSpec())

        execute_motion_plan(arm, plan, output=lambda message: None)

        self.assertEqual(
            [event[0] for event in arm.events],
            [
                "start",
                "joints",
                "pose",
                "pose",
                "grip",
                "pose",
                "joints",
                "pose",
                "release",
                "joints",
                "close",
            ],
        )


if __name__ == "__main__":
    unittest.main()
