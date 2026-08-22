# MO_Changes
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from src.franka.config import load_franka_config
from src.franka.models import CartesianPose, RobotPoint
from src.franka.robot import SimulatedFrankaRobotArm
from src.franka.workflow import FrankaAudioWorkflow, prepare_franka_for_detection
from src.zone_coordinates import OBJECT_PLACE_HEIGHTS, ZONE_COORDINATES


class _FakeTransformer:
    def __init__(self) -> None:
        self.calls = 0

    def transform(self, pixel, source_size):
        self.calls += 1
        return RobotPoint(0.40, 0.20, 0.3)


class FrankaWorkflowTest(unittest.TestCase):
    def test_franka_uses_ur_zones_and_object_place_heights(self) -> None:
        config = load_franka_config()

        for name in ("Zone_1", "Zone_2", "Zone_3"):
            with self.subTest(name=name):
                ur_zone = ZONE_COORDINATES[name]
                franka_zone = config.zone(name)
                self.assertEqual(
                    franka_zone.translation[:2],
                    (ur_zone["x"], ur_zone["y"]),
                )
        for object_class, height in OBJECT_PLACE_HEIGHTS.items():
            with self.subTest(object_class=object_class):
                self.assertEqual(config.place_height(object_class), height)
        self.assertLessEqual(config.workspace_y[0], ZONE_COORDINATES["Zone_1"]["y"])

    @patch("src.franka.workflow.FrankaRobotArm")
    def test_detection_preparation_uses_original_main_joints(
        self,
        arm_type,
    ) -> None:
        arm = arm_type.return_value
        arm.health.return_value = True
        expected_joints = load_franka_config().home_joints

        prepare_franka_for_detection("172.16.0.2")

        arm.start.assert_called_once_with()
        self.assertEqual(arm_type.call_args.args[0], "172.16.0.2")
        arm.move_joints.assert_called_once_with(expected_joints)
        arm.move_pose.assert_not_called()
        arm.close.assert_called_once_with()

    def test_complete_precision_workflow_uses_franka_arm(self) -> None:
        config = load_franka_config()
        initial = CartesianPose.create((0.36, 0.32, 0.45), config.default_orientation)
        arm = SimulatedFrankaRobotArm(initial)
        transformer = _FakeTransformer()
        methods = [
            "move_to_main_position",
            "detect_object",
            "convert_pixel_to_robot",
            "move_to_selected_object",
            "precision_detection",
            "filter_and_prepare_selected_object_after_precision_detection",
            "precision_pca_calculation",
            "precision_direction_object",
            "pick_the_object",
            "suction_on",
            "pick_up_object",
            "intermediate_position",
            "move_to_target(Zone_2)",
            "final_position",
            "suction_off",
            "intermediate_position",
            "move_to_main_position",
            "delet_txt_file",
        ]

        with tempfile.TemporaryDirectory() as directory:
            txt_dir = Path(directory)
            (txt_dir / "selection_data.json").write_text(
                json.dumps(
                    {
                        "original_center_x": 1200.0,
                        "original_center_y": 700.0,
                        "selected_object_class": "Cylinder",
                    }
                ),
                encoding="utf-8",
            )
            (txt_dir / "detected_objects.json").write_text(
                json.dumps({"objects": [{"class": 0}]}),
                encoding="utf-8",
            )
            (txt_dir / "label.txt").write_text("0 1 2 3 4 0.95", encoding="utf-8")
            (txt_dir / "final_object_center_point.txt").write_text(
                "1210 705", encoding="utf-8"
            )
            (txt_dir / "final_object_label.txt").write_text(
                "0 1 2 3 4 0.96", encoding="utf-8"
            )
            (txt_dir / "robot_RPY.txt").write_text(
                "[3.142 0.0 0.0]", encoding="utf-8"
            )
            with (
                patch("src.franka.workflow.TXT_DIR", txt_dir),
                patch("src.franka.workflow._read_detection_image_size", return_value=(2560, 1472)),
                patch("src.franka.workflow.perception_steps.detect_object"),
                patch("src.franka.workflow.perception_steps.precision_detection"),
                patch(
                    "src.franka.workflow.perception_steps."
                    "filter_and_prepare_selected_object_after_precision_detection",
                    return_value=True,
                ),
                patch("src.franka.workflow.perception_steps.precision_pca_calculation"),
                patch("src.franka.workflow.perception_steps.precision_direction_object"),
                patch("src.franka.workflow.perception_steps.delet_txt_file"),
            ):
                FrankaAudioWorkflow(
                    arm,
                    config,
                    transformer,
                    False,
                    lambda message: None,
                ).execute(methods)

        self.assertEqual(transformer.calls, 1)
        self.assertEqual(
            [event[0] for event in arm.movements],
            [
                "joints",
                "pose",
                "pose",
                "grip",
                "pose",
                "joints",
                "pose",
                "release",
                "joints",
                "joints",
            ],
        )
        self.assertEqual(arm.movements[6][1][:3], (0.366, -0.008, 0.067))

    def test_missing_real_zone_is_rejected_before_robot_starts(self) -> None:
        config = replace(load_franka_config(), zones={})
        initial = CartesianPose.create((0.36, 0.32, 0.45), config.default_orientation)
        arm = SimulatedFrankaRobotArm(initial)
        transformer = _FakeTransformer()

        with tempfile.TemporaryDirectory() as directory:
            txt_dir = Path(directory)
            (txt_dir / "selection_data.json").write_text("{}", encoding="utf-8")
            (txt_dir / "detected_objects.json").write_text("{}", encoding="utf-8")
            with patch("src.franka.workflow.TXT_DIR", txt_dir):
                workflow = FrankaAudioWorkflow(
                    arm,
                    config,
                    transformer,
                    False,
                    lambda message: None,
                )
                with self.assertRaisesRegex(ValueError, "has not been taught"):
                    workflow.execute(
                        ["move_to_main_position", "move_to_target(Zone_2)"]
                    )

        self.assertFalse(arm.health())
        self.assertEqual(arm.movements, [])


if __name__ == "__main__":
    unittest.main()
