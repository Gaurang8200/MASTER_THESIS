# MO_Changes
from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from src.franka.models import CartesianPose
from src.franka.robot import FrankaRobotArm


class _FakeEndEffectorPose:
    translation = np.array([0.4, 0.2, 0.1])
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])


class _FakeRobot:
    def __init__(self, host: str) -> None:
        self.host = host
        self.relative_dynamics_factor = None
        self.state = object()
        self.current_cartesian_state = types.SimpleNamespace(
            pose=types.SimpleNamespace(end_effector_pose=_FakeEndEffectorPose())
        )
        self.motions: list[object] = []

    def move(self, motion: object) -> None:
        self.motions.append(motion)


class _FakeGripper:
    def __init__(self, host: str) -> None:
        self.host = host
        self.grasped = False
        self.opened = False

    def grasp(self, width: float, speed: float, force: float, epsilon_outer: float) -> bool:
        self.grasped = True
        return True

    def open(self, speed: float) -> None:
        self.opened = True


class _FakeAffine:
    def __init__(self, translation: np.ndarray, quaternion: np.ndarray) -> None:
        self.translation = translation
        self.quaternion = quaternion


class _FakeMotion:
    def __init__(self, *values: object) -> None:
        self.values = values


class FrankaRobotTest(unittest.TestCase):
    def test_real_adapter_uses_franky_classes(self) -> None:
        fake_franky = types.SimpleNamespace(
            Robot=_FakeRobot,
            Gripper=_FakeGripper,
            Affine=_FakeAffine,
            JointMotion=_FakeMotion,
            CartesianMotion=_FakeMotion,
            ReferenceType=types.SimpleNamespace(Absolute="absolute"),
        )
        with patch.dict(sys.modules, {"franky": fake_franky}):
            arm = FrankaRobotArm("172.16.0.2", 0.05, 0.02, 20.0)
            arm.start()
            arm.move_joints((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7))
            arm.move_pose(
                CartesianPose.create((0.4, 0.2, 0.1), (1.0, 0.0, 0.0, 0.0))
            )
            arm.grip()
            arm.release()

            self.assertTrue(arm.health())
            self.assertEqual(arm.current_pose().translation, (0.4, 0.2, 0.1))
            self.assertEqual(len(arm._robot.motions), 2)
            self.assertTrue(arm._gripper.grasped)
            self.assertTrue(arm._gripper.opened)


if __name__ == "__main__":
    unittest.main()
