"""Place point selection.

Decides whether the fingertip points at a spot on the calibrated table and turns
that pixel into a robot pose.

The rule is deliberately simple. Any point inside the table area counts. An
object already lying there does not block it, because the operator is naming a
destination and not picking something up.

This is the only step of the gesture pipeline that needs calibration. The
decision itself is made in image coordinates, the pixel to robot conversion is
delegated to the adapter around the existing pick and drop repository and never
reimplemented here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from config import SelectionConfig, StabilityConfig, WorkspaceConfig
from fingertip_selection import HoldTimer, place_grid_key, point_in_polygon
from schemas import PlaceCalibration, RobotPose

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaceDecision:
    """Outcome of one place point step."""

    inside_workspace: bool
    candidate_pixel: tuple[float, float] | None
    place_pixel: tuple[float, float] | None
    pose: RobotPose | None
    held_s: float
    calibration_used: bool
    calibration_failed: bool
    notes: tuple[str, ...] = ()


class PlacePointSelector:
    """Confirms a table spot and converts it once it has been held long enough.

    The conversion runs only when the hold completes, so the calibration is
    called once per place point and not on every frame.
    """

    def __init__(
        self,
        workspace: WorkspaceConfig,
        stability: StabilityConfig,
        selection: SelectionConfig,
        calibration: PlaceCalibration,
    ) -> None:
        self._workspace = workspace
        self._selection = selection
        self._calibration = calibration
        self._timer = HoldTimer(stability.place_seconds)
        self._place: tuple[tuple[float, float], RobotPose, float] | None = None

    @property
    def calibration_mode(self) -> str:
        return self._calibration.mode

    @property
    def place(self) -> tuple[tuple[float, float], RobotPose, float] | None:
        return self._place

    def start(self) -> None:
        self._calibration.start()

    def reset(self) -> None:
        self._timer.reset()
        self._place = None

    def workspace_polygon(self, frame_shape: tuple[int, int]) -> np.ndarray:
        height, width = frame_shape[0], frame_shape[1]
        return self._workspace.to_pixels(width, height)

    def update(
        self,
        center: tuple[float, float] | None,
        frame_shape: tuple[int, int],
        touching_object: bool,
        has_selection: bool,
        active: bool,
        now: float,
    ) -> PlaceDecision:
        notes: list[str] = []
        inside_workspace = False
        candidate_pixel: tuple[float, float] | None = None

        if active and center is not None:
            inside_workspace = point_in_polygon(center, self.workspace_polygon(frame_shape))
            blocked_by_object = touching_object and not self._selection.ignore_objects_for_place
            if not inside_workspace:
                notes.append("fingertip_outside_workspace")
            elif blocked_by_object:
                notes.append("fingertip_on_an_object")
            elif self._selection.require_selection_before_place and not has_selection:
                notes.append("place_requires_selection_first")
            else:
                candidate_pixel = center

        state = self._timer.update(
            place_grid_key(candidate_pixel) if candidate_pixel is not None else None, now
        )

        calibration_failed = False
        if state.just_confirmed and candidate_pixel is not None:
            try:
                pose = self._calibration.convert_place_pixel_to_robot_pose(
                    candidate_pixel, frame_shape
                )
            except Exception as exc:  # a bad transform must never reach the robot
                calibration_failed = True
                self._timer.reset()
                notes.append("place_calibration_failed")
                LOGGER.error("place_calibration_failed error=%s", exc)
            else:
                self._place = (candidate_pixel, pose, state.held_s)
                notes.append("place_point_confirmed")
                LOGGER.info(
                    "place_point_confirmed pixel=%.1f,%.1f pose=%.4f,%.4f,%.4f held_s=%.1f mode=%s",
                    candidate_pixel[0],
                    candidate_pixel[1],
                    pose.x_m,
                    pose.y_m,
                    pose.z_m,
                    state.held_s,
                    self._calibration.mode,
                )

        return self._decision(inside_workspace, candidate_pixel, calibration_failed, notes)

    def _decision(
        self,
        inside_workspace: bool,
        candidate_pixel: tuple[float, float] | None,
        calibration_failed: bool,
        notes: Sequence[str],
    ) -> PlaceDecision:
        pixel = self._place[0] if self._place is not None else None
        pose = self._place[1] if self._place is not None else None
        held = self._place[2] if self._place is not None else 0.0
        return PlaceDecision(
            inside_workspace=inside_workspace,
            candidate_pixel=candidate_pixel,
            place_pixel=pixel,
            pose=pose,
            held_s=round(held, 2),
            calibration_used=pose is not None,
            calibration_failed=calibration_failed,
            notes=tuple(notes),
        )

    def health(self) -> dict[str, object]:
        return {
            "component": "place_point_selector",
            "calibration": self._calibration.health(),
            "has_place_point": self._place is not None,
        }
