from .schemas import RobotCommand, SafetyStatus
from loguru import logger

def validate_command(cmd: RobotCommand, stop_detected: bool, threshold: float) -> SafetyStatus:
    if stop_detected:
        logger.warning("Stop gesture/command detected!")
        return SafetyStatus(safe=False, reason="Stop detected")
    if cmd.action == "unknown" or (cmd.confidence is not None and cmd.confidence < threshold):
        logger.info("Low confidence or unknown command")
        return SafetyStatus(safe=False, reason="Low confidence or unknown command")
    return SafetyStatus(safe=True)