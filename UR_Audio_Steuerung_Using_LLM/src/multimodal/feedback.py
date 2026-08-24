# MO_Changes
"""Terminal and spoken feedback for multimodal selection."""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable


REJECTION_MESSAGES = {
    "pointed_object_not_in_detection_list": (
        "Fingertip and pointing finger were detected, but the pointed object does not match the detected object list."
    ),
    "fingertip_outside_object_boxes": (
        "Object is not detected. Point inside a detected object box and try again."
    ),
    "object_not_in_detection_list": (
        "The pointed object is not in the detected object list. Detect the objects again."
    ),
    "pointing_is_ambiguous": (
        "More than one object is under your fingertip. Point clearly at one object."
    ),
    "overview_match_ambiguous": (
        "The pointed object cannot be matched safely. Point clearly at one object."
    ),
    "fingertip_not_detected": "Your fingertip was not detected. Point at an object and try again.",
    "pointing_finger_not_detected": "A pointing gesture was not detected. Point and try again.",
    "pointing_not_stable": (
        "The pointed object was detected, but the pointing position was not held continuously for five seconds."
    ),
    "selection_timed_out": "No object was selected in time. Point at an object and try again.",
    "gesture_process_not_started": "Gesture selection could not start. Check the camera and model.",
    "camera_unavailable": "Gesture selection could not open the camera. Check macOS camera permission.",
    "gesture_result_missing": "Gesture selection did not return a result. Try again.",
    "service_failed": "Gesture selection failed. Check the terminal for details.",
}


class OperatorFeedback:
    """Publishes one message to the terminal, user interface, and speaker."""

    def __init__(self, ui_writer: Callable[[str], None] | None = None) -> None:
        self._ui_writer = ui_writer

    def rejection(self, reason: str) -> str:
        message = REJECTION_MESSAGES.get(
            reason,
            "Object is not detected or is not in the detected object list. Try again.",
        )
        self.publish(message)
        return message

    def publish(self, message: str, wait_for_speech: bool = False) -> None:
        terminal_message = f"MULTIMODAL: {message}"
        print(terminal_message)
        if self._ui_writer is not None:
            self._ui_writer(terminal_message + "\n")
        self._speak(message, wait_for_speech)

    @staticmethod
    def _speak(message: str, wait_for_speech: bool = False) -> bool:
        command: list[str] | None = None
        if sys.platform == "darwin" and shutil.which("say"):
            command = ["say", message]
        elif shutil.which("spd-say"):
            command = ["spd-say", message]
        if command is None:
            print("MULTIMODAL: No system speech command is available")
            return False
        runner = subprocess.run if wait_for_speech else subprocess.Popen
        runner(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
