"""Gesture selection pipeline.

Composes the gesture detector, the object source, the mode state machine, the
fingertip object selection and the place point selection into one frame step
that returns structured JSON. Nothing in this module commands the robot. The
result is the contract the existing pick and drop pipeline will consume later.

Coordinate rule. Gesture classification and object selection stay in image
coordinates. The place pose is the only field that carries robot coordinates,
and it comes from the calibration of the existing repository.

Run it directly:

    python run/main_pipeline.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Sequence

import numpy as np

import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
for _folder in ("support", "detection", "logic", "integration"):
    _path = str(PIPELINE_ROOT / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)


from calibration import ExistingRepoCalibration, PlaceCalibrationError, calibration_available
from config import GestureConfig, load_config, resolve_device
from fingertip_selection import ObjectSelector, bbox_center
from gesture_state_machine import GestureStateMachine
from object_detector import Yolov5ObjectDetector
from place_point_selector import PlacePointSelector
from robot_handoff import RobotHandoff
from schemas import (
    BoxModel,
    DetectedObject,
    FingertipModel,
    GestureFrame,
    GestureName,
    GestureSource,
    InteractionMode,
    LatencyModel,
    ModeTransition,
    ObjectSource,
    PipelineOutput,
    PlaceCalibration,
    PlacePointModel,
    PointModel,
    PoseModel,
    SelectedObjectModel,
    SelectionMode,
)

LOGGER = logging.getLogger(__name__)


class GestureSelectionPipeline:
    """One frame in, one structured selection result out."""

    def __init__(
        self,
        config: GestureConfig,
        gesture_source: GestureSource,
        object_source: ObjectSource,
        place_calibration: PlaceCalibration,
    ) -> None:
        self._config = config
        self._gesture_source = gesture_source
        self._object_source = object_source
        self._state_machine = GestureStateMachine(config.stability)
        self._object_selector = ObjectSelector(
            selection=config.selection,
            object_confidence=config.confidence.object,
            hold_seconds=config.stability.select_seconds,
        )
        self._place_selector = PlacePointSelector(
            workspace=config.workspace,
            stability=config.stability,
            selection=config.selection,
            calibration=place_calibration,
        )
        self._last_objects: list[DetectedObject] = []
        self._last_gesture_frame = GestureFrame(frame_index=-1)
        self._started = False

    @property
    def started(self) -> bool:
        return self._started

    @property
    def last_objects(self) -> list[DetectedObject]:
        """Objects of the last processed frame, exposed for the overlays."""
        return self._last_objects

    @property
    def last_gesture_frame(self) -> GestureFrame:
        return self._last_gesture_frame

    def start(self) -> None:
        self._gesture_source.start()
        self._object_source.start()
        self._place_selector.start()
        self._started = True
        LOGGER.info("pipeline_started")

    def close(self) -> None:
        self._started = False
        for component in (self._gesture_source, self._object_source):
            try:
                component.close()
            except Exception:
                LOGGER.exception("component_close_failed component=%s", type(component).__name__)
        LOGGER.info("pipeline_closed")

    def reset(self) -> None:
        """Drop all interaction state without reloading the models."""
        self._state_machine.reset()
        self._object_selector.reset()
        self._place_selector.reset()
        LOGGER.info("pipeline_reset")

    def process_frame(self, frame: np.ndarray, frame_index: int) -> PipelineOutput:
        if not self._started:
            raise RuntimeError("GestureSelectionPipeline.start must be called before process_frame")
        if frame is None or frame.ndim != 3:
            raise ValueError("frame must be a three dimensional BGR image")

        started = time.perf_counter()
        timestamp = time.monotonic()
        notes: list[str] = []

        # No calibration is required for gesture classification.
        gesture_frame = self._gesture_source.detect(frame, frame_index)
        self._last_gesture_frame = gesture_frame
        if not gesture_frame.ok:
            notes.append("gesture_inference_failed")

        state = self._state_machine.update(gesture_frame, timestamp)
        if state.transition is ModeTransition.DEACTIVATED:
            self._object_selector.reset()
            self._place_selector.reset()
            notes.append(f"selection_cleared:{state.reason}")

        objects_ms = 0.0
        objects: Sequence[DetectedObject] = ()
        if state.mode is SelectionMode.ON:
            object_started = time.perf_counter()
            objects = self._object_source.get_objects(frame)
            objects_ms = (time.perf_counter() - object_started) * 1000.0
        self._last_objects = list(objects)

        frame_shape = (frame.shape[0], frame.shape[1])
        fingertip = gesture_frame.best(GestureName.INDEX_FINGERTIP)
        pointing_present = gesture_frame.has(GestureName.POINTING_FINGER)
        pointing_ok = pointing_present or not self._config.selection.require_pointing_finger
        center = bbox_center(fingertip.box) if fingertip is not None else None
        active = state.mode is SelectionMode.ON and center is not None and pointing_ok
        if state.mode is SelectionMode.ON and center is not None and not pointing_ok:
            notes.append("pointing_finger_missing")

        # No calibration is required for object selection because the fingertip
        # point and the object boxes are both in image coordinates.
        selection = self._object_selector.update(center, objects, active, timestamp)
        notes.extend(selection.notes)

        # Calibration is only used when converting a fingertip pixel on the
        # table into a robot place coordinate.
        place = self._place_selector.update(
            center=center,
            frame_shape=frame_shape,
            touching_object=selection.candidate_id is not None,
            has_selection=selection.selected is not None,
            active=active,
            now=timestamp,
        )
        notes.extend(place.notes)

        fingertip_model: FingertipModel | None = None
        if fingertip is not None and center is not None:
            fingertip_model = FingertipModel(
                center_px=PointModel(x=center[0], y=center[1]),
                confidence=fingertip.confidence,
                inside_workspace=place.inside_workspace,
                pointing_finger_present=pointing_present,
            )

        total_ms = (time.perf_counter() - started) * 1000.0
        budget_ms = self._config.model.latency_budget_ms
        within_budget = total_ms <= budget_ms
        if not within_budget:
            notes.append("latency_over_budget")

        degraded = (not gesture_frame.ok) or (not within_budget) or place.calibration_failed
        selected_model = self._selected_model(selection.selected, selection)
        place_model = self._place_model(place)
        interaction_mode = InteractionMode.IDLE
        if place_model is not None:
            interaction_mode = InteractionMode.PLACE_SELECTION
        elif selected_model is not None:
            interaction_mode = InteractionMode.OBJECT_SELECTION

        safe_to_execute = (
            not degraded
            and state.mode is SelectionMode.ON
            and interaction_mode is not InteractionMode.IDLE
        )

        return PipelineOutput(
            mode=interaction_mode,
            frame_index=frame_index,
            frame_width=frame.shape[1],
            frame_height=frame.shape[0],
            timestamp_monotonic_s=timestamp,
            selection_mode=state.mode,
            mode_transition=state.transition,
            calibration_used=place.calibration_used,
            fingertip_pixel=[center[0], center[1]] if center is not None else None,
            selected_object_id=selected_model.object_id if selected_model else None,
            place_pixel=(
                [place.place_pixel[0], place.place_pixel[1]] if place.place_pixel else None
            ),
            place_robot_pose=place.pose.as_list() if place.pose is not None else None,
            fingertip=fingertip_model,
            selected_object=selected_model,
            place_point=place_model,
            candidate_object_id=selection.candidate_id,
            latency=LatencyModel(
                gesture_ms=round(gesture_frame.latency_ms, 3),
                objects_ms=round(objects_ms, 3),
                total_ms=round(total_ms, 3),
                budget_ms=budget_ms,
                within_budget=within_budget,
            ),
            degraded=degraded,
            safe_to_execute=safe_to_execute,
            robot_command_dispatched=False,
            notes=notes,
        )

    def health(self) -> dict[str, object]:
        return {
            "component": "gesture_selection_pipeline",
            "started": self._started,
            "selection_mode": self._state_machine.mode.value,
            "selected_object_id": (
                self._object_selector.selected.object_id
                if self._object_selector.selected is not None
                else None
            ),
            "gesture_source": self._gesture_source.health(),
            "object_source": self._object_source.health(),
            "place_point_selector": self._place_selector.health(),
        }

    def _selected_model(self, selected, selection) -> SelectedObjectModel | None:
        if selected is None:
            return None
        center_x, center_y = selected.box.center
        return SelectedObjectModel(
            object_id=selected.object_id,
            class_name=selected.class_name,
            confidence=selected.confidence,
            bbox=BoxModel.from_box(selected.box),
            centroid_px=PointModel(x=center_x, y=center_y),
            held_s=self._object_selector.held_s,
        )

    def _place_model(self, place) -> PlacePointModel | None:
        if place.place_pixel is None or place.pose is None:
            return None
        return PlacePointModel(
            pixel=PointModel(x=place.place_pixel[0], y=place.place_pixel[1]),
            pose=PoseModel.from_pose(place.pose),
            held_s=place.held_s,
        )

    def __enter__(self) -> "GestureSelectionPipeline":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def build_pipeline(
    config: GestureConfig,
    gesture_source: GestureSource | None = None,
    object_source: ObjectSource | None = None,
    place_calibration: PlaceCalibration | None = None,
) -> GestureSelectionPipeline:
    """Wire the real components, allowing any of them to be replaced."""
    if gesture_source is None:
        from gesture_detector import GestureDetector

        gesture_source = GestureDetector(
            model_config=config.model,
            confidence=config.confidence,
            class_ids_by_index=config.gesture_by_class_id(),
        )
    if object_source is None:
        object_source = Yolov5ObjectDetector(
            config=config.object_model,
            device=resolve_device(config.model.device),
            confidence=config.confidence.object,
        )
    if place_calibration is None:
        place_calibration = ExistingRepoCalibration(config.place_calibration)
    return GestureSelectionPipeline(config, gesture_source, object_source, place_calibration)


def _result_signature(result: PipelineOutput) -> tuple:
    return (
        result.selection_mode,
        result.mode,
        result.selected_object_id,
        tuple(result.place_robot_pose) if result.place_robot_pose else None,
    )


def run(
    config: GestureConfig,
    display: bool,
    max_frames: int,
    json_out: Path | None,
    json_every_frame: bool,
    show_handoff: bool = False,
) -> int:
    """Run the pipeline over the webcam."""
    if display:
        import cv2

        from visualization import render_pipeline_frame

    pipeline = build_pipeline(config)
    handoff = RobotHandoff(config.robot, config.place_calibration)
    records: list[dict] = []
    previous_signature = None
    camera = None

    try:
        pipeline.start()
    except (FileNotFoundError, PlaceCalibrationError, RuntimeError) as exc:
        print(f"startup failed: {exc}", file=sys.stderr)
        return 2

    try:
        from camera import CameraStream

        camera = CameraStream(config.camera)
        camera.start()

        frame_index = 0
        while max_frames <= 0 or frame_index < max_frames:
            frame = camera.read()
            if frame is None:
                if camera.consecutive_failures >= 10:
                    print("camera stopped delivering frames", file=sys.stderr)
                    return 3
                continue

            result = pipeline.process_frame(frame, frame_index)
            records.append(json.loads(result.to_json(indent=None)))

            signature = _result_signature(result)
            if json_every_frame or signature != previous_signature:
                print(result.to_json())
                previous_signature = signature

            for request in handoff.handle(result):
                if show_handoff:
                    print(json.dumps(request.as_dict()))

            if display:
                canvas = render_pipeline_frame(
                    frame=frame,
                    result=result,
                    objects=pipeline.last_objects,
                    gesture_frame=pipeline.last_gesture_frame,
                    polygon=config.workspace.to_pixels(frame.shape[1], frame.shape[0]),
                    show_object_boxes=config.visualization.show_object_boxes,
                    show_workspace=config.visualization.show_workspace,
                    show_hud=config.visualization.show_hud,
                    marker_radius=config.visualization.fingertip_marker_radius_px,
                    extra_hud=("q quit   r reset",),
                )
                cv2.imshow("gesture selection pipeline", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    pipeline.reset()

            frame_index += 1
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
    finally:
        if camera is not None:
            camera.close()
        if display:
            cv2.destroyAllWindows()
        pipeline.close()

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"wrote {len(records)} frame results to {json_out}", file=sys.stderr)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the gesture selection pipeline")
    parser.add_argument("--config", type=Path, default=None, help="path to gesture_config.yaml")
    parser.add_argument("--display", dest="display", action="store_true", default=True)
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.add_argument("--max-frames", type=int, default=0, help="0 runs until q or interrupt")
    parser.add_argument("--json-out", type=Path, default=None, help="write every frame result")
    parser.add_argument("--json-every-frame", action="store_true")
    parser.add_argument(
        "--show-handoff",
        action="store_true",
        help="print the pick and place requests that would go to the robot",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config.logging.level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not calibration_available(config.place_calibration):
        print(
            "the calibration files of the existing repository are missing, "
            f"looked in {config.place_calibration.repo_dir}",
            file=sys.stderr,
        )
        return 2

    if config.robot.dispatch:
        print(
            f"robot dispatch is ON, motions will be sent to {config.robot.host}",
            file=sys.stderr,
        )

    return run(
        config=config,
        display=args.display,
        max_frames=args.max_frames,
        json_out=args.json_out,
        json_every_frame=args.json_every_frame,
        show_handoff=args.show_handoff,
    )


if __name__ == "__main__":
    raise SystemExit(main())
