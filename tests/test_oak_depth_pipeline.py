from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = REPOSITORY_ROOT / "UR_Audio_Steuerung_Using_LLM"
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

from src.camera_devices import parse_camera_option
from src.FR_franka.FR_depth import measure_object_height
from src.FR_franka.FR_models import PixelPoint
from src.FR_franka.FR_original_transformer import OriginalFrankaPixelTransformer


class OakDepthPipelineTests(unittest.TestCase):
    def test_height_uses_object_and_support_medians(self) -> None:
        depth = np.full((100, 100), 700, dtype=np.uint16)
        depth[30:70, 30:70] = 620
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.npy"
            np.save(path, depth, allow_pickle=False)
            measurement = measure_object_height(
                path,
                (30, 30, 70, 70),
                inset_ratio=0.25,
                support_ring_scale=1.6,
                minimum_valid_pixels=25,
                maximum_height_m=0.25,
            )
        self.assertEqual(measurement.object_depth_mm, 620.0)
        self.assertEqual(measurement.support_depth_mm, 700.0)
        self.assertAlmostEqual(measurement.height_m, 0.08)

    def test_invalid_depth_fails_closed(self) -> None:
        depth = np.zeros((40, 40), dtype=np.uint16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.npy"
            np.save(path, depth, allow_pickle=False)
            with self.assertRaisesRegex(ValueError, "valid depth pixels"):
                measure_object_height(path, (10, 10, 30, 30), 0.2, 1.5, 5, 0.25)

    def test_only_oak_camera_options_are_accepted(self) -> None:
        self.assertEqual(parse_camera_option("oak_d:device_1"), "device_1")
        with self.assertRaisesRegex(ValueError, "OAK D"):
            parse_camera_option("opencv:0")

    def test_missing_oak_calibration_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "OAK D calibration"):
                OriginalFrankaPixelTransformer(
                    (640, 400),
                    True,
                    Path(directory),
                )

    def test_existing_pixel_conversion_algorithm_is_unchanged(self) -> None:
        calibration_directory = (
            AUDIO_ROOT / "src" / "FR_franka" / "FR_original_calibration"
        )
        transformer = OriginalFrankaPixelTransformer(
            (640, 400),
            True,
            calibration_directory,
        )
        point = transformer.transform(PixelPoint(372.0, 200.0), (640, 400))
        self.assertAlmostEqual(point.x, 0.41553, places=5)
        self.assertAlmostEqual(point.y, 0.37818, places=5)
        self.assertEqual(point.z, 0.3)


if __name__ == "__main__":
    unittest.main()
