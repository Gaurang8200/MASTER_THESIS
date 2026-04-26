from app.schemas import RobotCommand
from app.safety import validate_command

def test_safety_stop():
    cmd = RobotCommand(action="pick", object_id="obj", target_location="A", confidence=0.9)
    status = validate_command(cmd, stop_detected=True, threshold=0.85)
    assert not status.safe

def test_safety_low_confidence():
    cmd = RobotCommand(action="pick", object_id="obj", target_location="A", confidence=0.5)
    status = validate_command(cmd, stop_detected=False, threshold=0.85)
    assert not status.safe