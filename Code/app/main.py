import asyncio
from .config import load_config
from .logging_config import setup_logging
from .schemas import RobotCommand
from .fusion import fuse_modalities
from .safety import validate_command
from .robot_interface import RobotInterface

from gesture.mediapipe_detector import GestureDetector
from voice.asr import ASR
from nlp.intent_parser import IntentParser

async def main():
    config = load_config()
    setup_logging(config.log_level)
    robot = RobotInterface(config.robot_ip)
    gesture_detector = GestureDetector(config)
    asr = ASR(config)
    nlp = IntentParser(config)

    while True:
        gesture_result = await gesture_detector.detect()
        asr_result = await asr.listen()
        nlp_result = nlp.parse(asr_result.text)
        command = fuse_modalities(gesture_result, nlp_result)
        stop_detected = (nlp_result.action == "stop") or gesture_result.object_id == "stop"
        safety = validate_command(command, stop_detected, config.confidence_threshold)
        if safety.safe:
            robot.send_command(command)
        else:
            # Optionally trigger confirmation via UI
            pass
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())