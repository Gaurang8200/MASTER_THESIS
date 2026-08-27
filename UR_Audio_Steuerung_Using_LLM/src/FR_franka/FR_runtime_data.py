# MO_Changes
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2

from .FR_models import PixelPoint


def read_detection_image_size(txt_dir: Path, yolo_root: Path) -> tuple[int, int]:
    detection_data_path = txt_dir / "detected_objects.json"
    detection_data = json.loads(detection_data_path.read_text(encoding="utf-8"))
    image_path_value = detection_data.get("metadata", {}).get("image_path")
    if not image_path_value:
        raise ValueError("Detection metadata does not contain the source image path")
    image_path = Path(str(image_path_value))
    if not image_path.is_absolute():
        image_path = yolo_root / image_path
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read detection image {image_path}")
    height, width = image.shape[:2]
    return int(width), int(height)


def read_pixel_file(path: Path) -> PixelPoint:
    values = read_numeric_values(path)
    if len(values) < 2:
        raise ValueError(f"Pixel file {path} requires two values")
    return PixelPoint(values[0], values[1])


def read_class_file(path: Path) -> int:
    values = read_numeric_values(path)
    if not values:
        raise ValueError(f"Class file {path} is empty")
    return int(values[0])


def read_numeric_values(path: Path) -> list[float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    content = path.read_text(encoding="utf-8")
    pattern = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    return [float(value) for value in re.findall(pattern, content)]
