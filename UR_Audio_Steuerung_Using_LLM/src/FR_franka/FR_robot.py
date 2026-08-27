# MO_Changes
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np

from .FR_models import CartesianPose


LOGGER = logging.getLogger(__name__)


def quaternion_to_rotation_matrix(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = CartesianPose.create((0.0, 0.0, 0.0), quaternion).quaternion
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


class RobotArm(ABC):
    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def health(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def move_joints(self, joints: Sequence[float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def move_pose(self, pose: CartesianPose) -> None:
        raise NotImplementedError

    @abstractmethod
    def current_pose(self) -> CartesianPose:
        raise NotImplementedError

    @abstractmethod
    def grip(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def release(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def base_to_end_effector(self) -> np.ndarray:
        pose = self.current_pose()
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = quaternion_to_rotation_matrix(pose.quaternion)
        transform[:3, 3] = np.asarray(pose.translation, dtype=float)
        return transform


class FrankaRobotArm(RobotArm):
    def __init__(
        self,
        host: str,
        dynamics_factor: float,
        gripper_speed: float,
        gripper_force: float,
    ) -> None:
        if not host.strip():
            raise ValueError("Franka host must not be empty")
        if not 0.0 < dynamics_factor <= 1.0:
            raise ValueError("dynamics factor must be between zero and one")
        self._host = host
        self._dynamics_factor = float(dynamics_factor)
        self._gripper_speed = float(gripper_speed)
        self._gripper_force = float(gripper_force)
        self._api: Any = None
        self._robot: Any = None
        self._gripper: Any = None

    def start(self) -> None:
        if self._robot is not None:
            return
        try:
            import franky
        except ImportError as error:
            raise RuntimeError(
                "Real Franka mode requires the franky-control package"
            ) from error
        LOGGER.info("FRANKA_CONNECT_START host=%s", self._host)
        self._api = franky
        self._robot = franky.Robot(self._host)
        self._robot.relative_dynamics_factor = self._dynamics_factor
        self._gripper = franky.Gripper(self._host)
        LOGGER.info("FRANKA_CONNECT_READY host=%s", self._host)

    def health(self) -> bool:
        if self._robot is None:
            return False
        try:
            return self._robot.state is not None
        except Exception:
            LOGGER.exception("FRANKA_HEALTH_FAILED")
            return False

    def move_joints(self, joints: Sequence[float]) -> None:
        self._require_started()
        checked = tuple(float(value) for value in joints)
        if len(checked) != 7:
            raise ValueError("Franka joint motion requires seven joint values")
        LOGGER.info("FRANKA_MOVE_JOINTS target=%s", checked)
        self._robot.move(self._api.JointMotion(list(checked)))

    def move_pose(self, pose: CartesianPose) -> None:
        self._require_started()
        LOGGER.info("FRANKA_MOVE_POSE target=%s", pose.translation)
        affine = self._api.Affine(
            np.asarray(pose.translation, dtype=float),
            np.asarray(pose.quaternion, dtype=float),
        )
        motion = self._api.CartesianMotion(affine, self._api.ReferenceType.Absolute)
        self._robot.move(motion)

    def current_pose(self) -> CartesianPose:
        self._require_started()
        state = self._robot.current_cartesian_state
        end_effector_pose = state.pose.end_effector_pose
        return CartesianPose.create(
            np.asarray(end_effector_pose.translation, dtype=float).reshape(3),
            np.asarray(end_effector_pose.quaternion, dtype=float).reshape(4),
        )

    def grip(self) -> None:
        self._require_started()
        if self._gripper is None:
            raise RuntimeError("Franka gripper is unavailable")
        LOGGER.info("FRANKA_GRIP_START")
        success = self._gripper.grasp(
            0.0,
            self._gripper_speed,
            self._gripper_force,
            epsilon_outer=1.0,
        )
        if not success:
            raise RuntimeError("Franka gripper did not confirm the grasp")
        LOGGER.info("FRANKA_GRIP_READY")

    def release(self) -> None:
        self._require_started()
        if self._gripper is None:
            raise RuntimeError("Franka gripper is unavailable")
        LOGGER.info("FRANKA_RELEASE_START")
        self._gripper.open(self._gripper_speed)
        LOGGER.info("FRANKA_RELEASE_READY")

    def close(self) -> None:
        LOGGER.info("FRANKA_CONNECTION_CLOSE host=%s", self._host)
        self._gripper = None
        self._robot = None
        self._api = None

    def _require_started(self) -> None:
        if self._robot is None:
            raise RuntimeError("Franka robot is not connected")


class SimulatedFrankaRobotArm(RobotArm):
    def __init__(self, initial_pose: CartesianPose) -> None:
        self._pose = initial_pose
        self._started = False
        self._gripping = False
        self.movements: list[tuple[str, tuple[float, ...]]] = []

    def start(self) -> None:
        self._started = True

    def health(self) -> bool:
        return self._started

    def move_joints(self, joints: Sequence[float]) -> None:
        self._require_started()
        checked = tuple(float(value) for value in joints)
        if len(checked) != 7:
            raise ValueError("Franka joint motion requires seven joint values")
        self.movements.append(("joints", checked))

    def move_pose(self, pose: CartesianPose) -> None:
        self._require_started()
        self._pose = pose
        self.movements.append(("pose", pose.translation + pose.quaternion))

    def current_pose(self) -> CartesianPose:
        self._require_started()
        return self._pose

    def grip(self) -> None:
        self._require_started()
        self._gripping = True
        self.movements.append(("grip", ()))

    def release(self) -> None:
        self._require_started()
        self._gripping = False
        self.movements.append(("release", ()))

    def close(self) -> None:
        self._started = False

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("Simulated Franka robot is not started")
