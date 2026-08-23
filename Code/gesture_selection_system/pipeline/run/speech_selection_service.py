# MO_Changes
"""One speech session to one safe fingertip selection result."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
for _folder in ("support", "detection", "logic"):
    _path = str(PIPELINE_ROOT / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from camera import CameraStream
from config import GestureConfig, load_config, resolve_device
from fingertip_selection import HoldTimer, bbox_center, find_touched_object
from gesture_classes import GestureName
from gesture_detector import GestureDetector
from object_detector import Yolov5ObjectDetector
from schemas import DetectedObject, GestureFrame, SelectionMode

LOGGER = logging.getLogger(__name__)
CONTRACT_VERSION = "1.0"


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf8")
    temporary.replace(path)


def _stop_requested(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("command") == "stop"


def _object_payload(item: DetectedObject) -> dict[str, object]:
    center_x, center_y = item.box.center
    return {
        "live_object_id": item.object_id,
        "class_name": item.class_name,
        "confidence": item.confidence,
        "bbox": [item.box.x1, item.box.y1, item.box.x2, item.box.y2],
        "center": [center_x, center_y],
    }


def _base_result(session_id: str, status: str, reason: str) -> dict[str, object]:
    return {
        "schema_version": CONTRACT_VERSION,
        "session_id": session_id,
        "status": status,
        "reason": reason,
        "safe_to_use": False,
        "selected_at_unix_s": None,
        "frame_index": None,
        "frame_width": None,
        "frame_height": None,
        "fingertip_pixel": None,
        "fingertip_confidence": None,
        "pointing_finger_present": False,
        "objects_considered": 0,
        "selected_object": None,
        "latency_ms": None,
    }


def _reason_for_frame(
    gesture_frame: GestureFrame,
    pointing_present: bool,
    has_fingertip: bool,
    object_count: int,
    inside_count: int,
) -> str:
    if not gesture_frame.ok:
        return "gesture_inference_failed"
    if not has_fingertip:
        return "fingertip_not_detected"
    if not pointing_present:
        return "pointing_finger_not_detected"
    if object_count == 0:
        return "object_not_in_detection_list"
    if inside_count == 0:
        return "fingertip_outside_object_boxes"
    if inside_count > 1:
        return "pointing_is_ambiguous"
    return "pointing_not_stable"


def _render(
    frame,
    gesture_frame: GestureFrame,
    objects: Sequence[DetectedObject],
    fingertip_center: tuple[float, float] | None,
    selected_id: str | None,
    reason: str,
) -> int:
    import cv2

    from visualization import draw_fingertip, draw_gestures, draw_hud, draw_objects

    canvas = frame.copy()
    draw_objects(canvas, objects, selected_id)
    draw_gestures(canvas, gesture_frame)
    if fingertip_center is not None:
        draw_fingertip(canvas, fingertip_center, 10, True)
    draw_hud(canvas, ("speech gesture selection", reason, "q cancel"), SelectionMode.ON)
    cv2.imshow("multimodal fingertip selection", canvas)
    return cv2.waitKey(1) & 0xFF


def run_session(
    config: GestureConfig,
    session_id: str,
    result_file: Path,
    request_file: Path,
    ready_file: Path,
    timeout_seconds: float,
    hold_seconds: float,
    display: bool,
) -> int:
    gesture = GestureDetector(
        model_config=config.model,
        confidence=config.confidence,
        class_ids_by_index=config.gesture_by_class_id(),
    )
    objects_source = Yolov5ObjectDetector(
        config=config.object_model,
        device=resolve_device(config.model.device),
        confidence=config.confidence.object,
    )
    timer = HoldTimer(hold_seconds)
    result = _base_result(session_id, "rejected", "no_usable_frame")
    selected: DetectedObject | None = None
    camera = CameraStream(config.camera)
    frame_index = 0

    try:
        gesture.start()
        objects_source.start()
        camera.start()
        _write_json(
            ready_file,
            {
                "schema_version": CONTRACT_VERSION,
                "session_id": session_id,
                "status": "ready",
            },
        )
        started = time.monotonic()

        while time.monotonic() - started < timeout_seconds:
            if _stop_requested(request_file):
                break

            frame = camera.read()
            if frame is None:
                if camera.consecutive_failures >= 10:
                    result = _base_result(session_id, "error", "camera_read_failed")
                    break
                continue

            frame_started = time.perf_counter()
            gesture_frame = gesture.detect(frame, frame_index)
            objects = objects_source.get_objects(frame) if gesture_frame.ok else []
            fingertip = gesture_frame.best(GestureName.INDEX_FINGERTIP)
            pointing_present = gesture_frame.has(GestureName.POINTING_FINGER)
            center = bbox_center(fingertip.box) if fingertip is not None else None

            touch = None
            if center is not None and pointing_present:
                touch = find_touched_object(
                    objects,
                    center,
                    config.confidence.object,
                    config.selection.max_center_distance_ratio,
                )

            inside_count = touch.inside_count if touch is not None else 0
            candidate = touch.touched if touch is not None and inside_count == 1 else None
            hold = timer.update(candidate.object_id if candidate is not None else None, time.monotonic())
            if hold.just_confirmed and candidate is not None:
                selected = candidate
                result = {
                    **_base_result(session_id, "selected", "selected"),
                    "safe_to_use": True,
                    "selected_at_unix_s": time.time(),
                    "frame_index": frame_index,
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                    "fingertip_pixel": [center[0], center[1]],
                    "fingertip_confidence": fingertip.confidence,
                    "pointing_finger_present": True,
                    "objects_considered": touch.considered,
                    "selected_object": _object_payload(candidate),
                    "hold_seconds": hold.held_s,
                    "latency_ms": round((time.perf_counter() - frame_started) * 1000.0, 3),
                }
            elif selected is None:
                reason = _reason_for_frame(
                    gesture_frame,
                    pointing_present,
                    fingertip is not None,
                    len(objects),
                    inside_count,
                )
                result = {
                    **_base_result(session_id, "rejected", reason),
                    "frame_index": frame_index,
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                    "fingertip_pixel": [center[0], center[1]] if center is not None else None,
                    "fingertip_confidence": fingertip.confidence if fingertip is not None else None,
                    "pointing_finger_present": pointing_present,
                    "objects_considered": len(objects),
                    "latency_ms": round((time.perf_counter() - frame_started) * 1000.0, 3),
                }

            if display:
                key = _render(
                    frame,
                    gesture_frame,
                    objects,
                    center,
                    selected.object_id if selected is not None else None,
                    str(result["reason"]),
                )
                if key == ord("q"):
                    result = _base_result(session_id, "rejected", "operator_cancelled")
                    break

            frame_index += 1

        if result["status"] != "selected" and time.monotonic() - started >= timeout_seconds:
            result["reason"] = "selection_timed_out"
        _write_json(result_file, result)
        return 0 if result["status"] in {"selected", "rejected"} else 2
    except Exception as error:
        LOGGER.exception("speech_selection_service_failed")
        reason = (
            "camera_unavailable"
            if "could not open camera" in str(error).lower()
            else "service_failed"
        )
        result = _base_result(session_id, "error", reason)
        result["error"] = str(error)
        _write_json(result_file, result)
        return 2
    finally:
        camera.close()
        gesture.close()
        objects_source.close()
        if display:
            import cv2

            cv2.destroyAllWindows()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select one object during a speech session")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--result-file", required=True, type=Path)
    parser.add_argument("--request-file", required=True, type=Path)
    parser.add_argument("--ready-file", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=12.0)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--display", dest="display", action="store_true", default=True)
    parser.add_argument("--no-display", dest="display", action="store_false")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return run_session(
        config=load_config(arguments.config),
        session_id=arguments.session_id,
        result_file=arguments.result_file,
        request_file=arguments.request_file,
        ready_file=arguments.ready_file,
        timeout_seconds=arguments.timeout_seconds,
        hold_seconds=arguments.hold_seconds,
        display=arguments.display,
    )


if __name__ == "__main__":
    raise SystemExit(main())
