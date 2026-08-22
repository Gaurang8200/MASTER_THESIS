# MO_Changes
from __future__ import annotations

import unittest

from src.franka.calibration import FrankaPixelTransformer
from src.franka.config import load_franka_config
from src.franka.geometry import calibration_pose_15
from src.franka.models import PixelPoint
from src.franka.robot import SimulatedFrankaRobotArm


class FrankaCalibrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_franka_config()
        cls.transformer = FrankaPixelTransformer(
            cls.config.calibration_dir,
            (cls.config.calibration_width, cls.config.calibration_height),
            cls.config.mirror_x,
        )

    def test_pixel_maps_to_calibrated_table_in_franka_base(self) -> None:
        arm = SimulatedFrankaRobotArm(calibration_pose_15(self.config))
        arm.start()

        point = self.transformer.transform(
            PixelPoint(640.0, 360.0),
            (1280, 720),
            arm.base_to_end_effector(),
        )

        self.assertGreater(point.x, 0.1)
        self.assertLess(point.x, 0.75)
        self.assertGreater(point.y, -0.1)
        self.assertLess(point.y, 0.65)
        self.assertAlmostEqual(point.z, -0.011, delta=0.01)

    def test_detection_pixel_is_scaled_to_calibration_resolution(self) -> None:
        arm = SimulatedFrankaRobotArm(calibration_pose_15(self.config))
        arm.start()
        full = self.transformer.transform(
            PixelPoint(1280.0, 736.0),
            (2560, 1472),
            arm.base_to_end_effector(),
        )
        calibrated = self.transformer.transform(
            PixelPoint(640.0, 360.0),
            (1280, 720),
            arm.base_to_end_effector(),
        )

        self.assertAlmostEqual(full.x, calibrated.x, places=9)
        self.assertAlmostEqual(full.y, calibrated.y, places=9)
        self.assertAlmostEqual(full.z, calibrated.z, places=9)


if __name__ == "__main__":
    unittest.main()
