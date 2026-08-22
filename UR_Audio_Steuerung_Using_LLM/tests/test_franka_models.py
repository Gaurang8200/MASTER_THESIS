# MO_Changes
from __future__ import annotations

import unittest

from src.franka.models import CartesianPose
from src.franka.robot import SimulatedFrankaRobotArm


class FrankaModelTest(unittest.TestCase):
    def test_pose_normalizes_quaternion(self) -> None:
        pose = CartesianPose.create((0.4, 0.2, 0.1), (2.0, 0.0, 0.0, 0.0))
        self.assertEqual(pose.quaternion, (1.0, 0.0, 0.0, 0.0))

    def test_simulated_arm_records_motion_and_gripper(self) -> None:
        initial = CartesianPose.create((0.36, 0.32, 0.45), (1.0, 0.0, 0.0, 0.0))
        target = CartesianPose.create((0.40, 0.20, 0.10), (1.0, 0.0, 0.0, 0.0))
        arm = SimulatedFrankaRobotArm(initial)

        arm.start()
        arm.move_pose(target)
        arm.grip()
        arm.release()

        self.assertEqual(arm.current_pose(), target)
        self.assertEqual([event[0] for event in arm.movements], ["pose", "grip", "release"])

    def test_franka_rejects_six_joint_ur_pose(self) -> None:
        initial = CartesianPose.create((0.36, 0.32, 0.45), (1.0, 0.0, 0.0, 0.0))
        arm = SimulatedFrankaRobotArm(initial)
        arm.start()

        with self.assertRaisesRegex(ValueError, "seven joint values"):
            arm.move_joints((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
