from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = REPOSITORY_ROOT / "UR_Audio_Steuerung_Using_LLM"
if str(AUDIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIO_ROOT))

from src.camera_devices import (
    OAK_MONO_RESOLUTION,
    OakDFrame,
    map_detections_to_left_mono,
    parse_camera_option,
)
from src.FR_franka.FR_depth import measure_object_height
from src.FR_franka.FR_models import PixelPoint
from src.FR_franka.FR_original_transformer import OriginalFrankaPixelTransformer


class OakDepthPipelineTests(unittest.TestCase):
    def test_rgb_point_maps_to_left_mono_with_same_depth(self) -> None:
        class RgbTransformation:
            def projectPointTo(
                self,
                target: object,
                point: Any,
                depth: float,
            ) -> SimpleNamespace:
                self.target = target
                self.point = point
                self.depth = depth
                return SimpleNamespace(x=372.0, y=200.0)

        rgb_transformation = RgbTransformation()
        left_transformation = object()
        frame = OakDFrame(
            color=np.zeros((720, 1280, 3), dtype=np.uint8),
            depth_mm=np.full((720, 1280), 620, dtype=np.uint16),
            left_mono=np.zeros((720, 1280), dtype=np.uint8),
            rgb_transformation=rgb_transformation,
            left_transformation=left_transformation,
        )
        mapped = map_detections_to_left_mono(
            [{"center": [800.0, 420.0], "bbox": [700.0, 320.0, 900.0, 520.0]}],
            frame,
        )
        self.assertEqual(mapped[0]["left_mono_center"], [372.0, 200.0])
        self.assertEqual(mapped[0]["left_mono_frame_size"], [1280, 720])
        self.assertEqual(mapped[0]["object_depth_mm"], 620.0)
        self.assertEqual(rgb_transformation.depth, 620.0)
        self.assertEqual(
            (rgb_transformation.point.x, rgb_transformation.point.y),
            (800.0, 420.0),
        )
        self.assertIs(rgb_transformation.target, left_transformation)

    def test_active_mono_resolution_matches_calibration(self) -> None:
        config_path = AUDIO_ROOT / "src" / "FR_config" / "franka_robot.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(OAK_MONO_RESOLUTION, (1280, 720))
        self.assertEqual(config["calibration_image_size"], [1280, 720])

    def test_franka_capture_and_gesture_use_calibration_resolution(self) -> None:
        detector_root = AUDIO_ROOT / "Code-YOLOv5-Windows_llm"
        for name in ("detection_multi.py", "detection_multi_precision_run.py"):
            source = (detector_root / name).read_text(encoding="utf-8")
            self.assertIn("FRANKA_CAPTURE_RESOLUTION = (1280, 720)", source)
        gesture_config = (
            REPOSITORY_ROOT
            / "Code"
            / "gesture_selection_system"
            / "pipeline"
            / "configs"
            / "gesture_config.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("  width: 1280\n  height: 720", gesture_config)
        self.assertIn("  imgsz: 640", gesture_config)

    def test_rgb_mapping_rejects_missing_object_depth(self) -> None:
        frame = OakDFrame(
            color=np.zeros((80, 128, 3), dtype=np.uint8),
            depth_mm=np.zeros((80, 128), dtype=np.uint16),
            left_mono=np.zeros((40, 64), dtype=np.uint8),
            rgb_transformation=object(),
            left_transformation=object(),
        )
        mapped = map_detections_to_left_mono(
            [{"center": [80.0, 42.0], "bbox": [70.0, 32.0, 90.0, 52.0]}],
            frame,
        )
        self.assertNotIn("left_mono_center", mapped[0])
        self.assertIn("valid depth pixels", mapped[0]["left_mono_mapping_error"])

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
