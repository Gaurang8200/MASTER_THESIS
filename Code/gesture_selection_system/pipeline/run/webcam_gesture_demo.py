"""Webcam gesture demo.

Shows only the gesture layer, which is the fastest way to check that the model
sees the four classes on a bare hand and on a gloved hand and that the five
frame activation and deactivation rules behave. Object selection is not part of
this demo, use main_pipeline.py for that.

    python run/webcam_gesture_demo.py
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Sequence

import cv2

import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
for _folder in ("support", "detection", "logic", "integration"):
    _path = str(PIPELINE_ROOT / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)
_COMMON = str(PIPELINE_ROOT.parent / "common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)


from camera import CameraStream
from config import load_config
from gesture_state_machine import GestureStateMachine
from schemas import GestureName, SelectionMode
from visualization import draw_gestures, draw_fingertip, draw_hud

LOGGER = logging.getLogger(__name__)


def _build_detector(config):
    from gesture_detector import GestureDetector

    return GestureDetector(
        model_config=config.model,
        confidence=config.confidence,
        class_ids_by_index=config.gesture_by_class_id(),
    )


def run(config, max_frames: int) -> int:
    detector = _build_detector(config)
    state_machine = GestureStateMachine(config.stability)

    try:
        detector.start()
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"gesture detector could not start: {exc}", file=sys.stderr)
        return 2

    frame_index = 0
    fps = 0.0
    last_stamp = time.perf_counter()

    try:
        with CameraStream(config.camera) as camera:
            while max_frames <= 0 or frame_index < max_frames:
                frame = camera.read()
                if frame is None:
                    if camera.consecutive_failures >= 10:
                        print("camera stopped delivering frames", file=sys.stderr)
                        return 3
                    continue

                gesture_frame = detector.detect(frame, frame_index)
                state = state_machine.update(gesture_frame, time.monotonic())

                now = time.perf_counter()
                interval = now - last_stamp
                last_stamp = now
                if interval > 0:
                    fps = 0.9 * fps + 0.1 * (1.0 / interval) if fps else 1.0 / interval

                canvas = frame.copy()
                draw_gestures(canvas, gesture_frame)
                fingertip = gesture_frame.best(GestureName.INDEX_FINGERTIP)
                if fingertip is not None:
                    draw_fingertip(
                        canvas,
                        fingertip.box.center,
                        config.selection.fingertip_radius_px,
                        config.visualization.fingertip_marker_radius_px,
                        state.mode is SelectionMode.ON,
                    )

                if config.visualization.show_hud:
                    draw_hud(canvas, _hud_lines(state, gesture_frame, fps), state.mode)

                cv2.imshow("gesture demo", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    state_machine.reset()

                frame_index += 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
    except RuntimeError as exc:
        print(f"demo failed: {exc}", file=sys.stderr)
        return 3
    finally:
        cv2.destroyAllWindows()
        detector.close()

    return 0


def _hud_lines(state, gesture_frame, fps: float) -> list[str]:
    detected = ", ".join(sorted({d.gesture.value for d in gesture_frame.detections})) or "none"
    lines = [
        f"mode: {state.mode.value.upper()}  reason: {state.reason}",
        f"open held: {state.open_palm_held_s:.1f}s   stop held: {state.closed_palm_held_s:.1f}s",
        f"detected: {detected}",
        f"inference: {gesture_frame.latency_ms:.0f} ms   fps: {fps:.1f}",
        "q quit   r reset",
    ]
    if not gesture_frame.ok:
        lines.append("detector error, results are not usable")
    return lines


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam gesture demo")
    parser.add_argument("--config", type=Path, default=None, help="path to gesture_config.yaml")
    parser.add_argument("--max-frames", type=int, default=0, help="0 runs until q or interrupt")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return run(config=config, max_frames=args.max_frames)


if __name__ == "__main__":
    raise SystemExit(main())
