# MO_Changes
from __future__ import annotations

import unittest
from unittest.mock import patch

from src.detection_preparation import get_default_robot_ip, prepare_robot_for_detection


class DetectionPreparationTest(unittest.TestCase):
    def test_each_robot_type_has_its_own_default_ip(self) -> None:
        self.assertEqual(get_default_robot_ip("franka"), "172.16.0.2")
        self.assertEqual(get_default_robot_ip("universal"), "192.168.2.180")

    def test_unknown_robot_type_has_no_default_ip(self) -> None:
        with self.assertRaises(ValueError):
            get_default_robot_ip("unknown")

    @patch("src.detection_preparation.move_to_main_position")
    @patch("src.detection_preparation.prepare_franka_for_detection")
    def test_franka_selection_uses_only_franka_preparation(
        self,
        franka_prepare,
        universal_prepare,
    ) -> None:
        prepare_robot_for_detection("franka", "172.16.0.2")

        franka_prepare.assert_called_once_with("172.16.0.2")
        universal_prepare.assert_not_called()

    @patch("src.detection_preparation.move_to_main_position")
    @patch("src.detection_preparation.prepare_franka_for_detection")
    def test_universal_selection_uses_only_universal_preparation(
        self,
        franka_prepare,
        universal_prepare,
    ) -> None:
        prepare_robot_for_detection("universal", "192.168.2.180")

        universal_prepare.assert_called_once_with("192.168.2.180")
        franka_prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
