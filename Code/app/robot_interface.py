from .schemas import RobotCommand
from loguru import logger

class RobotInterface:
    def __init__(self, robot_ip: str):
        self.robot_ip = robot_ip

    def send_command(self, cmd: RobotCommand):
        logger.info(f"Sending command to robot: {cmd}")
        # Placeholder: Implement ROS2, socket, or REST API call here
        pass