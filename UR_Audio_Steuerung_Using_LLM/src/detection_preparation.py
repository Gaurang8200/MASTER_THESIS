# MO_Changes
from __future__ import annotations

from src.franka import prepare_franka_for_detection
from src.robot_control import move_to_main_position


def prepare_robot_for_detection(robot_type: str, robot_ip: str) -> None:
    selected_robot = robot_type.strip().lower()
    selected_ip = robot_ip.strip()
    if not selected_ip:
        raise ValueError("Robot IP must not be empty")
    if selected_robot == "franka":
        prepare_franka_for_detection(selected_ip)
        return
    if selected_robot == "universal":
        move_to_main_position(selected_ip)
        return
    raise ValueError(f"Unsupported robot type {robot_type}")
