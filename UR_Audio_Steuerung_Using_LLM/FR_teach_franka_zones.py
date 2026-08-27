# MO_Changes
from __future__ import annotations

import json
from pathlib import Path

from src.FR_franka.FR_config import DEFAULT_CONFIG_PATH, load_franka_config
from src.FR_franka.FR_robot import FrankaRobotArm


ZONE_NAMES = ("Zone_1", "Zone_2", "Zone_3")


def teach_zones(config_path: Path = DEFAULT_CONFIG_PATH) -> None:
    config = load_franka_config(config_path)
    arm = FrankaRobotArm(
        config.robot_ip,
        config.dynamics_factor,
        config.gripper_speed,
        config.gripper_force,
    )
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    zones = dict(raw_config.get("zones", {}))
    arm.start()
    try:
        if not arm.health():
            raise RuntimeError("Franka robot health check failed")
        for zone_name in ZONE_NAMES:
            input(
                f"Move the Franka tool to the release pose for {zone_name}, then press Enter: "
            )
            pose = arm.current_pose()
            zones[zone_name] = {
                "translation": list(pose.translation),
                "quaternion": list(pose.quaternion),
            }
            print(f"Recorded {zone_name}: {pose.translation}")
    finally:
        arm.close()
    raw_config["zones"] = zones
    config_path.write_text(
        json.dumps(raw_config, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved Franka zones to {config_path}")


if __name__ == "__main__":
    teach_zones()
