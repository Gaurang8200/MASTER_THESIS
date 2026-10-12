from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


YOLO_ROOT = (
    Path(__file__).resolve().parents[1]
    / "Code-YOLOv5-Windows_llm"
    / "yolov5"
)
DETECTION_SCRIPT = YOLO_ROOT / "detect_multi_objects.py"
_MODULE_NAME = "master_thesis_yolo_detection"
_runtime: ModuleType | None = None


def run_yolo_detection(**options: Any) -> list[dict[str, object]]:
    return _load_runtime().run(**options)


def warm_yolo_detection(weights: str | Path) -> None:
    _load_runtime().warmup_model(weights=str(weights))


def _load_runtime() -> ModuleType:
    global _runtime
    if _runtime is not None:
        return _runtime
    if not DETECTION_SCRIPT.is_file():
        raise FileNotFoundError(f"YOLO detection script not found at {DETECTION_SCRIPT}")
    if str(YOLO_ROOT) not in sys.path:
        sys.path.insert(0, str(YOLO_ROOT))
    specification = importlib.util.spec_from_file_location(
        _MODULE_NAME,
        DETECTION_SCRIPT,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load YOLO detection from {DETECTION_SCRIPT}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[_MODULE_NAME] = module
    specification.loader.exec_module(module)
    _runtime = module
    return module
