# MO_Changes
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from src.franka.models import PixelPoint
from src.franka.original_transformer import (
    CALIBRATION_DIR,
    OriginalFrankaPixelTransformer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SOURCE_CALIBRATION_DIR = REPOSITORY_ROOT / "Handgesture_FrankaEmika"


class FrankaCalibrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.transformer = OriginalFrankaPixelTransformer((1280, 720), False)

    def test_original_pixel2robot_calculation_is_called_with_pose_15(self) -> None:
        with patch(
            "src.franka.original_transformer.pixel2robot",
            return_value=(0.385, 0.338, 0.3),
        ) as original_calculation:
            point = self.transformer.transform(
                PixelPoint(1280.0, 736.0),
                (2560, 1472),
            )

        original_calculation.assert_called_once_with(640.0, 360.0, 15)
        self.assertEqual((point.x, point.y, point.z), (0.385, 0.338, 0.3))

    def test_original_calculation_returns_known_calibrated_coordinates(self) -> None:
        point = self.transformer.transform(
            PixelPoint(640.0, 360.0),
            (1280, 720),
        )

        self.assertAlmostEqual(point.x, 0.3851089091, places=9)
        self.assertAlmostEqual(point.y, 0.3380565786, places=9)
        self.assertEqual(point.z, 0.3)

    def test_pose_15_comes_from_copied_robot_poses(self) -> None:
        pose = self.transformer.calibration_pose()

        self.assertEqual(pose.translation, (0.3627558, 0.3237964, 0.4580297))

    def test_copied_calibration_json_files_equal_the_sources(self) -> None:
        names = (
            "input_params.json",
            "output_b2p.json",
            "output_c2f.json",
            "output_wp2camera.json",
            "robot_poses.json",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(
                    (CALIBRATION_DIR / name).read_bytes(),
                    (SOURCE_CALIBRATION_DIR / name).read_bytes(),
                )

    def test_copied_function_pool_keeps_original_calculation(self) -> None:
        copied = (CALIBRATION_DIR / "function_pool.py").read_text(encoding="utf-8")
        source = (SOURCE_CALIBRATION_DIR / "function_pool.py").read_text(encoding="utf-8")

        self.assertEqual(copied.removeprefix("# MO_Changes\n"), source)

    def test_copied_pixel2robot_keeps_original_calculation(self) -> None:
        copied = (CALIBRATION_DIR / "pixel2robot.py").read_text(encoding="utf-8")
        source = (SOURCE_CALIBRATION_DIR / "pixel2robot.py").read_text(encoding="utf-8")
        normalized = copied.removeprefix("# MO_Changes\n").replace(
            "from . import function_pool as fp",
            "import function_pool as fp",
        )

        self.assertEqual(normalized, source)


if __name__ == "__main__":
    unittest.main()
