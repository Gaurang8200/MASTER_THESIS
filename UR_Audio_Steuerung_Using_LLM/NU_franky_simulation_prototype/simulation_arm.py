# MO_Changes
from __future__ import annotations

import logging
from contextlib import ExitStack
from typing import Any

from src.FR_franka.FR_config import FrankaConfig
from src.FR_franka.FR_robot import FrankaRobotArm

from .scene import SceneSpec, add_workcell


LOGGER = logging.getLogger(__name__)


class FrankySimulationArm(FrankaRobotArm):
    def __init__(
        self,
        config: FrankaConfig,
        scene: SceneSpec,
        render: bool = True,
    ) -> None:
        super().__init__(
            host="127.0.0.1",
            dynamics_factor=config.dynamics_factor,
            gripper_speed=config.gripper_speed,
            gripper_force=config.gripper_force,
        )
        self._config = config
        self._scene_spec = scene
        self._render = bool(render)
        self._resources: ExitStack | None = None
        self._simulation_server: Any = None
        self._simulator: Any = None
        self._simulation_hostname: str | None = None

    @property
    def simulation_hostname(self) -> str:
        if self._simulation_hostname is None:
            raise RuntimeError("Franky simulation has not started")
        return self._simulation_hostname

    def start(self) -> None:
        if self._robot is not None:
            return
        try:
            import franky
            from franky_sim import SimulationServer
            from franky_sim.mujoco_simulator import MujocoSimulator
        except ImportError as error:
            raise RuntimeError(
                "Install franky_control and franky_sim before running the prototype"
            ) from error

        resources = ExitStack()
        try:
            simulator = resources.enter_context(
                MujocoSimulator(enable_visualization=self._render)
            )
            zone = self._config.zone(self._scene_spec.target_zone)
            add_workcell(
                simulator.worldbody,
                self._scene_spec,
                zone.translation[0],
                zone.translation[1],
            )
            simulation_robot = simulator.add_robot(
                initial_q=self._config.home_joints,
            )
            server = resources.enter_context(SimulationServer(simulator))
            server.run_async()

            robot = franky.Robot(
                simulation_robot.hostname,
                realtime_config=franky.RealtimeConfig.Ignore,
            )
            robot.relative_dynamics_factor = self._dynamics_factor
            gripper = franky.Gripper(simulation_robot.hostname)
        except Exception:
            resources.close()
            raise

        self._resources = resources
        self._simulator = simulator
        self._simulation_server = server
        self._simulation_hostname = simulation_robot.hostname
        self._host = simulation_robot.hostname
        self._api = franky
        self._robot = robot
        self._gripper = gripper
        LOGGER.info("FRANKY_SIM_READY host=%s", self._host)

    def health(self) -> bool:
        if self._robot is None or self._simulation_server is None:
            return False
        try:
            return bool(
                self._simulation_server.running
                and self._robot.current_joint_state is not None
            )
        except Exception:
            LOGGER.exception("FRANKY_SIM_HEALTH_FAILED")
            return False

    def close(self) -> None:
        resources = self._resources
        super().close()
        self._resources = None
        self._simulation_server = None
        self._simulator = None
        self._simulation_hostname = None
        if resources is not None:
            resources.close()
        LOGGER.info("FRANKY_SIM_CLOSED")
