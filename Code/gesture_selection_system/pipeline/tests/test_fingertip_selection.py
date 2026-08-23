# MO_Changes
from __future__ import annotations

import sys
import unittest
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
for folder in ("support", "logic"):
    path = str(PIPELINE_ROOT / folder)
    if path not in sys.path:
        sys.path.insert(0, path)

run_path = str(PIPELINE_ROOT / "run")
if run_path not in sys.path:
    sys.path.insert(0, run_path)

from fingertip_selection import bbox_center, find_touched_object, point_in_box
from schemas import BoundingBox, DetectedObject
from speech_selection_service import parse_args


def object_at(object_id: str, box: BoundingBox) -> DetectedObject:
    return DetectedObject(object_id, "Cylinder", 0.95, box)


class FingertipSelectionTest(unittest.TestCase):
    def test_speech_selection_default_hold_is_three_seconds(self) -> None:
        arguments = parse_args(
            [
                "--session-id",
                "test_session",
                "--result-file",
                "result.json",
                "--request-file",
                "request.json",
                "--ready-file",
                "ready.json",
            ]
        )
        self.assertEqual(arguments.hold_seconds, 3.0)

    def test_fingertip_uses_the_box_center(self) -> None:
        self.assertEqual(bbox_center(BoundingBox(10, 20, 30, 50)), (20.0, 35.0))

    def test_point_on_box_boundary_is_inside(self) -> None:
        box = BoundingBox(100, 100, 200, 200)
        self.assertTrue(point_in_box((100, 150), box))

    def test_point_one_pixel_outside_is_rejected(self) -> None:
        box = BoundingBox(100, 100, 200, 200)
        result = find_touched_object([object_at("obj_1", box)], (99, 150), 0.70)
        self.assertIsNone(result.touched)
        self.assertEqual(result.inside_count, 0)

    def test_overlapping_boxes_are_reported_as_ambiguous(self) -> None:
        objects = [
            object_at("obj_1", BoundingBox(100, 100, 250, 250)),
            object_at("obj_2", BoundingBox(150, 150, 220, 220)),
        ]
        result = find_touched_object(objects, (180, 180), 0.70)
        self.assertEqual(result.inside_count, 2)
        self.assertEqual(result.touched.object_id, "obj_2")

    def test_low_confidence_object_cannot_be_selected(self) -> None:
        item = DetectedObject("obj_1", "Cylinder", 0.50, BoundingBox(0, 0, 100, 100))
        result = find_touched_object([item], (50, 50), 0.70)
        self.assertIsNone(result.touched)
        self.assertEqual(result.skipped_low_confidence, 1)


if __name__ == "__main__":
    unittest.main()
