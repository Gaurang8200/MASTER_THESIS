# MO_Changes
"""One speech session to one safe fingertip selection result."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Sequence

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
for _folder in ("support", "detection", "logic"):
    _path = str(PIPELINE_ROOT / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)

from camera import CameraStream
from config import GestureConfig, load_config, resolve_device
from fingertip_selection import (
    HoldTimer,
    bbox_center,
    find_touched_object,
    place_grid_key,
    point_in_box,
)
from gesture_classes import GestureName
from gesture_detector import GestureDetector
from object_detector import Yolov5ObjectDetector, box_iou
from schemas import DetectedObject, GestureFrame, SelectionMode

LOGGER = logging.getLogger(__name__)
CONTRACT_VERSION = "1.0"
WINDOW_NAME = "multimodal fingertip selection"


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


def _object_payload(
    item: DetectedObject,
    camera: CameraStream,
    frame_shape: tuple[int, ...],
) -> dict[str, object]:
    sensor_box = camera.to_sensor_box(
        (item.box.x1, item.box.y1, item.box.x2, item.box.y2),
        frame_shape,
    )
    center_x = (sensor_box[0] + sensor_box[2]) / 2.0
    center_y = (sensor_box[1] + sensor_box[3]) / 2.0
    return {
        "live_object_id": item.object_id,
        "class_name": item.class_name,
        "confidence": item.confidence,
        "bbox": list(sensor_box),
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
        "left_mono_pixel": None,
        "left_mono_frame_width": None,
        "left_mono_frame_height": None,
        "fingertip_depth_mm": None,
        "object_depth_mm": None,
        "object_depth_sample_count": None,
        "depth_path": None,
        "left_mono_mapping_error": None,
        "fingertip_confidence": None,
        "pointing_finger_present": False,
        "objects_considered": 0,
        "selected_object": None,
        "latency_ms": None,
    }


def _left_mono_payload(
    camera: CameraStream,
    sensor_point: tuple[float, float] | None,
    depth_field: str = "fingertip_depth_mm",
    depth_mm: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "left_mono_pixel": None,
        "left_mono_frame_width": None,
        "left_mono_frame_height": None,
        "fingertip_depth_mm": None,
        "object_depth_mm": None,
        "object_depth_sample_count": None,
        "left_mono_mapping_error": None,
    }
    if sensor_point is None:
        return payload
    try:
        left_point, measured_depth_mm, left_size = camera.project_sensor_point_to_left(
            sensor_point,
            depth_mm,
        )
    except (RuntimeError, ValueError) as error:
        payload["left_mono_mapping_error"] = str(error)
        return payload
    payload.update(
        {
            "left_mono_pixel": [left_point[0], left_point[1]],
            "left_mono_frame_width": left_size[0],
            "left_mono_frame_height": left_size[1],
            depth_field: measured_depth_mm,
        }
    )
    return payload


def _tracked_object(
    selected: DetectedObject,
    objects: Sequence[DetectedObject],
) -> DetectedObject | None:
    for item in objects:
        if item.object_id == selected.object_id:
            return item
    same_class = [item for item in objects if item.class_name == selected.class_name]
    if not same_class:
        return None
    closest = max(same_class, key=lambda item: box_iou(selected.box, item.box))
    return closest if box_iou(selected.box, closest.box) >= 0.25 else None


REJECTION_INFORMATION_RANK = {
    "no_usable_frame": 0,
    "gesture_inference_failed": 1,
    "fingertip_not_detected": 2,
    "pointing_finger_not_detected": 3,
    "pointed_object_not_in_detection_list": 4,
    "pointing_is_ambiguous": 5,
    "pointing_not_stable": 6,
}


def _prefer_informative_rejection(
    current: dict[str, object],
    candidate: dict[str, object],
) -> dict[str, object]:
    """Keep the result that proves the furthest successful perception stage."""
    current_rank = REJECTION_INFORMATION_RANK.get(str(current.get("reason")), 0)
    candidate_rank = REJECTION_INFORMATION_RANK.get(str(candidate.get("reason")), 0)
    return candidate if candidate_rank >= current_rank else current


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
    if object_count == 0 or inside_count == 0:
        return "pointed_object_not_in_detection_list"
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
    cv2.imshow(WINDOW_NAME, canvas)
    return cv2.waitKey(1) & 0xFF


def _window_is_visible() -> bool:
    import cv2

    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1.0
    except cv2.error:
        return False


def _parent_is_alive(parent_pid: int | None) -> bool:
    return parent_pid is None or os.getppid() == parent_pid


def _recent_pointing_state(
    detected: bool,
    observed_at: float,
    last_detected_at: float | None,
    grace_seconds: float,
) -> tuple[bool, float | None]:
    if detected:
        return True, observed_at
    if last_detected_at is None:
        return False, None
    return observed_at - last_detected_at <= grace_seconds, last_detected_at


def run_session(
    config: GestureConfig,
    session_id: str,
    result_file: Path,
    request_file: Path,
    ready_file: Path,
    timeout_seconds: float,
    hold_seconds: float,
    selection_kind: str,
    display: bool,
    parent_pid: int | None,
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
    selection_complete = False
    selected_fingertip: tuple[float, float] | None = None
    selected_hold_seconds = 0.0
    depth_history: deque[float] = deque(maxlen=5)
    hand_clear_since: float | None = None
    display_reason = str(result["reason"])
    camera = CameraStream(config.camera)
    depth_file = result_file.with_name(f"{session_id}_depth.npy")
    frame_index = 0
    window_created = False
    last_pointing_at: float | None = None
    last_depth_error: str | None = None

    try:
        gesture.start()
        objects_source.start()
        camera.start()
        if display:
            import cv2

            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, config.camera.width, config.camera.height)
        _write_json(
            ready_file,
            {
                "schema_version": CONTRACT_VERSION,
                "session_id": session_id,
                "status": "ready",
            },
        )
        started = time.monotonic()

        while timeout_seconds <= 0.0 or time.monotonic() - started < timeout_seconds:
            if not _parent_is_alive(parent_pid):
                result = _base_result(session_id, "rejected", "parent_process_stopped")
                break
            if _stop_requested(request_file):
                break
            if display and window_created and not _window_is_visible():
                result = _base_result(session_id, "rejected", "operator_cancelled")
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
            observed_at = time.monotonic()
            pointing_detected = gesture_frame.has(GestureName.POINTING_FINGER)
            pointing_present, last_pointing_at = _recent_pointing_state(
                pointing_detected,
                observed_at,
                last_pointing_at,
                config.selection.pointing_grace_seconds,
            )
            center = bbox_center(fingertip.box) if fingertip is not None else None
            sensor_center = (
                camera.to_sensor_point(center, frame.shape)
                if center is not None
                else None
            )

            touch = None
            if selection_kind == "object" and center is not None and pointing_present:
                touch = find_touched_object(
                    objects,
                    center,
                    config.confidence.object,
                    config.selection.max_center_distance_ratio,
                )

            inside_count = touch.inside_count if touch is not None else 0
            candidate = (
                touch.touched
                if touch is not None and inside_count == 1
                else None
            )
            candidate_key = candidate.object_id if candidate is not None else None
            if selection_kind == "location" and center is not None and pointing_present:
                candidate_key = place_grid_key(center)
            hold = timer.update(candidate_key, observed_at)
            hold_confirmed = (
                candidate_key is not None and hold.confirmed_key == candidate_key
            )
            if (
                selection_kind == "object"
                and hold_confirmed
                and not selection_complete
            ):
                selected = candidate
                selection_complete = True
                selected_fingertip = sensor_center
                selected_hold_seconds = hold.held_s
                depth_history.clear()
                hand_clear_since = None
                display_reason = "selected, remove hand for depth"
            elif selection_kind == "object" and selection_complete and selected:
                tracked = _tracked_object(selected, objects)
                if tracked is None:
                    hand_clear_since = None
                    display_reason = "selected object not visible"
                elif (
                    pointing_detected
                    and center is not None
                    and point_in_box(center, tracked.box)
                ):
                    selected = tracked
                    hand_clear_since = None
                    display_reason = "selected, remove hand for depth"
                elif hand_clear_since is None:
                    selected = tracked
                    hand_clear_since = observed_at
                    display_reason = "selected, keep hand clear"
                elif observed_at - hand_clear_since < 2.0:
                    selected = tracked
                    display_reason = "selected, keep hand clear"
                else:
                    selected = tracked
                    sensor_box = camera.to_sensor_box(
                        (
                            tracked.box.x1,
                            tracked.box.y1,
                            tracked.box.x2,
                            tracked.box.y2,
                        ),
                        frame.shape,
                    )
                    try:
                        depth_mm, sample_count = camera.measure_object_depth(sensor_box)
                    except ValueError as error:
                        display_reason = "measuring object depth"
                        if str(error) != last_depth_error:
                            print(f"OAK D DEPTH REJECTED: {error}", flush=True)
                            last_depth_error = str(error)
                    else:
                        last_depth_error = None
                        depth_history.append(depth_mm)
                        stable_depth_mm = camera.stable_depth(list(depth_history))
                        print(
                            "OAK D DEPTH FRAME: "
                            f"depth={depth_mm:.1f} mm, samples={sample_count}, "
                            f"stable_frames={len(depth_history)}/3",
                            flush=True,
                        )
                        display_reason = (
                            f"measuring object depth {len(depth_history)}/3"
                            if stable_depth_mm is None
                            else "object depth ready"
                        )
                        if stable_depth_mm is not None:
                            object_center = (
                                (sensor_box[0] + sensor_box[2]) / 2.0,
                                (sensor_box[1] + sensor_box[3]) / 2.0,
                            )
                            mapping = _left_mono_payload(
                                camera,
                                object_center,
                                "object_depth_mm",
                                stable_depth_mm,
                            )
                            if mapping["left_mono_mapping_error"] is None:
                                mapping["object_depth_sample_count"] = sample_count
                                camera.save_latest_depth(depth_file)
                                result = {
                                    **_base_result(session_id, "selected", "selected"),
                                    "safe_to_use": True,
                                    "selection_kind": selection_kind,
                                    "selected_at_unix_s": time.time(),
                                    "last_seen_at_unix_s": time.time(),
                                    "frame_index": frame_index,
                                    "frame_width": int(frame.shape[1]),
                                    "frame_height": int(frame.shape[0]),
                                    "fingertip_pixel": (
                                        list(selected_fingertip)
                                        if selected_fingertip is not None
                                        else None
                                    ),
                                    **mapping,
                                    "depth_path": str(depth_file),
                                    "fingertip_confidence": None,
                                    "pointing_finger_present": False,
                                    "objects_considered": len(objects),
                                    "selected_object": _object_payload(
                                        tracked,
                                        camera,
                                        frame.shape,
                                    ),
                                    "hold_seconds": selected_hold_seconds,
                                    "latency_ms": round(
                                        (time.perf_counter() - frame_started) * 1000.0,
                                        3,
                                    ),
                                }
                                display_reason = "selected"
                                _write_json(result_file, result)
            elif (
                selection_kind == "location"
                and hold_confirmed
                and not selection_complete
            ):
                mapping = _left_mono_payload(camera, sensor_center)
                if mapping["left_mono_mapping_error"] is None:
                    selected = candidate
                    selection_complete = True
                    result = {
                        **_base_result(session_id, "selected", "selected"),
                        "safe_to_use": True,
                        "selection_kind": selection_kind,
                        "selected_at_unix_s": time.time(),
                        "last_seen_at_unix_s": time.time(),
                        "frame_index": frame_index,
                        "frame_width": int(frame.shape[1]),
                        "frame_height": int(frame.shape[0]),
                        "fingertip_pixel": [sensor_center[0], sensor_center[1]],
                        **mapping,
                        "depth_path": None,
                        "fingertip_confidence": fingertip.confidence,
                        "pointing_finger_present": True,
                        "objects_considered": 0,
                        "selected_object": None,
                        "hold_seconds": hold.held_s,
                        "latency_ms": round(
                            (time.perf_counter() - frame_started) * 1000.0,
                            3,
                        ),
                    }
                    display_reason = "selected"
                    _write_json(result_file, result)
            elif not selection_complete:
                if selection_kind == "location":
                    reason = (
                        "pointing_not_stable"
                        if center is not None and pointing_present
                        else _reason_for_frame(
                            gesture_frame,
                            pointing_present,
                            fingertip is not None,
                            1,
                            1,
                        )
                    )
                else:
                    reason = _reason_for_frame(
                        gesture_frame,
                        pointing_present,
                        fingertip is not None,
                        len(objects),
                        inside_count,
                    )
                frame_rejection = {
                    **_base_result(session_id, "rejected", reason),
                    "frame_index": frame_index,
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                    "fingertip_pixel": (
                        [sensor_center[0], sensor_center[1]]
                        if sensor_center is not None
                        else None
                    ),
                    **_left_mono_payload(camera, sensor_center),
                    "fingertip_confidence": fingertip.confidence if fingertip is not None else None,
                    "pointing_finger_present": pointing_present,
                    "objects_considered": len(objects),
                    "latency_ms": round((time.perf_counter() - frame_started) * 1000.0, 3),
                }
                result = _prefer_informative_rejection(result, frame_rejection)
                display_reason = reason

            if display:
                key = _render(
                    frame,
                    gesture_frame,
                    objects,
                    center,
                    selected.object_id if selected is not None else None,
                    display_reason,
                )
                window_created = True
                if key == ord("q"):
                    result = _base_result(session_id, "rejected", "operator_cancelled")
                    break

            frame_index += 1

        if (
            result["status"] != "selected"
            and timeout_seconds > 0.0
            and time.monotonic() - started >= timeout_seconds
        ):
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
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--hold-seconds", type=float, default=3.0)
    parser.add_argument("--parent-pid", type=int, default=None)
    parser.add_argument(
        "--selection-kind",
        choices=("object", "location"),
        default="object",
    )
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
        selection_kind=arguments.selection_kind,
        display=arguments.display,
        parent_pid=arguments.parent_pid,
    )


if __name__ == "__main__":
    raise SystemExit(main())
