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
from src.franka.workflow import FrankaAudioWorkflow


class _FakeTransformer:
    def __init__(self) -> None:
        self.calls = 0

    def transform(self, pixel, source_size, base_to_end_effector):
        self.calls += 1
        if self.calls == 1:
            return RobotPoint(0.40, 0.20, 0.0)
        return RobotPoint(0.405, 0.195, 0.0)


class FrankaWorkflowTest(unittest.TestCase):
    def test_complete_precision_workflow_uses_franka_arm(self) -> None:
        config = load_franka_config()
        zone = CartesianPose.create((0.50, 0.10, 0.10), config.default_orientation)
        config = replace(config, zones={"Zone_2": zone})
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

        self.assertEqual(transformer.calls, 2)
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
