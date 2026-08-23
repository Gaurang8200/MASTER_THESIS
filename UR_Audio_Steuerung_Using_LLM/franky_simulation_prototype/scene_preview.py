# MO_Changes
from __future__ import annotations

import argparse

from src.franka.config import load_franka_config

from .scene import SceneSpec, add_workcell


def run_scene_preview(render: bool = True, steps: int = 1) -> None:
    try:
        from franky_sim import SimulationServer
        from franky_sim.mujoco_simulator import MujocoSimulator
    except ImportError as error:
        raise RuntimeError(
            "Install franky_sim before opening the prototype scene"
        ) from error

    if steps < 1:
        raise ValueError("Preview steps must be positive")

    config = load_franka_config()
    scene = SceneSpec()
    zone = config.zone(scene.target_zone)

    with MujocoSimulator(enable_visualization=render) as simulator:
        add_workcell(
            simulator.worldbody,
            scene,
            zone.translation[0],
            zone.translation[1],
        )
        simulator.add_robot(initial_q=config.home_joints)
        with SimulationServer(simulator) as server:
            if render:
                server.run_forever()
                return
            for _ in range(steps):
                server.run_once(realtime=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open the isolated Franka workcell without commanding motion"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Build and validate the scene without opening the viewer",
    )
    arguments = parser.parse_args()
    run_scene_preview(render=not arguments.headless)


if __name__ == "__main__":
    main()
