# MO_Changes
from __future__ import annotations

import unittest

from src.multimodal.selection import is_pointing_reference, resolve_multimodal_selection


def selected_result(box: list[float], class_name: str = "Cylinder") -> dict[str, object]:
    return {
        "status": "selected",
        "reason": "selected",
        "safe_to_use": True,
        "frame_width": 1000,
        "frame_height": 1000,
        "selected_object": {
            "class_name": class_name,
            "bbox": box,
        },
    }


class MultimodalSelectionTest(unittest.TestCase):
    def test_this_object_requires_gesture(self) -> None:
        self.assertTrue(is_pointing_reference("Pick up this object", {}))

    def test_standalone_pick_up_requires_gesture(self) -> None:
        self.assertTrue(is_pointing_reference("Pick up", {}))

    def test_named_object_remains_speech_selection(self) -> None:
        info = {"object": "Cylinder", "selection_mode": "speech"}
        self.assertFalse(is_pointing_reference("Pick up the second cylinder", info))

    def test_rejected_gesture_blocks_selection(self) -> None:
        resolution = resolve_multimodal_selection(
            "Pick up this object",
            {},
            {
                "status": "rejected",
                "reason": "fingertip_outside_object_boxes",
                "safe_to_use": False,
            },
            {"objects": []},
        )
        self.assertTrue(resolution.required)
        self.assertFalse(resolution.accepted)
        self.assertEqual(resolution.reason, "fingertip_outside_object_boxes")

    def test_single_matching_class_is_selected(self) -> None:
        cylinder = {
            "id": 7,
            "class_name": "Cylinder",
            "bbox": [200, 200, 400, 400],
        }
        resolution = resolve_multimodal_selection(
            "Pick up this object",
            {},
            selected_result([100, 100, 200, 200]),
            {
                "objects": [cylinder],
                "metadata": {"image_width": 2000, "image_height": 2000},
            },
        )
        self.assertTrue(resolution.accepted)
        self.assertIs(resolution.selected_object, cylinder)
        self.assertEqual(resolution.object_index, 0)

    def test_normalized_geometry_selects_the_correct_duplicate(self) -> None:
        first = {
            "id": 1,
            "class_name": "Cylinder",
            "bbox": [200, 200, 400, 400],
        }
        second = {
            "id": 2,
            "class_name": "Cylinder",
            "bbox": [1400, 1400, 1700, 1700],
        }
        detection = {
            "objects": [first, second],
            "metadata": {"image_width": 2000, "image_height": 2000},
        }
        resolution = resolve_multimodal_selection(
            "Pick up this object",
            {},
            selected_result([100, 100, 200, 200]),
            detection,
        )
        self.assertTrue(resolution.accepted)
        self.assertIs(resolution.selected_object, first)
        self.assertEqual(resolution.object_index, 0)

    def test_equally_good_duplicate_matches_are_rejected(self) -> None:
        detection = {
            "objects": [
                {"id": 1, "class_name": "Cylinder", "bbox": [200, 200, 400, 400]},
                {"id": 2, "class_name": "Cylinder", "bbox": [200, 200, 400, 400]},
            ],
            "metadata": {"image_width": 2000, "image_height": 2000},
        }
        resolution = resolve_multimodal_selection(
            "Pick up this object",
            {},
            selected_result([100, 100, 200, 200]),
            detection,
        )
        self.assertFalse(resolution.accepted)
        self.assertEqual(resolution.reason, "overview_match_ambiguous")


if __name__ == "__main__":
    unittest.main()
