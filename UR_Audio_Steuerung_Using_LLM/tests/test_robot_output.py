# MO_Changes
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import robot_control
from src.robot_output import emit_method_execution


class RobotOutputTest(unittest.TestCase):
    def test_method_message_is_identical_for_both_robot_workflows(self) -> None:
        callback_messages: list[str] = []

        with patch("builtins.print") as terminal_print:
            message = emit_method_execution(
                "convert_pixel_to_robot",
                callback_messages.append,
            )

        expected = "DEBUG METHOD: Executing convert_pixel_to_robot"
        self.assertEqual(message, expected)
        terminal_print.assert_called_once_with(expected)
        self.assertEqual(callback_messages, [expected])

    def test_ur_simulation_coordinates_are_unchanged_by_output(self) -> None:
        messages: list[str] = []

        with tempfile.TemporaryDirectory() as directory:
            predecessor = Path(directory)
            txt_directory = predecessor / "txt_file"
            txt_directory.mkdir()
            with patch.object(robot_control, "PREDECESSOR_DIR", str(predecessor)):
                robot_control.set_simulation_mode(True, messages.append)
                try:
                    result = robot_control.convert_pixel_to_robot_multi(1280, 720)
                finally:
                    robot_control.set_simulation_mode(False, None)

            coordinates = (txt_directory / "robot_coordinates.txt").read_text(
                encoding="utf-8"
            )

        self.assertTrue(result)
        self.assertEqual(coordinates, "0.300000\n-0.050000\n")
        self.assertTrue(
            any("SIMULATION COORDINATE RESULT" in message for message in messages)
        )


if __name__ == "__main__":
    unittest.main()
