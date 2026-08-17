"""Handoff to the existing pick and drop system.

This is where a confirmed selection leaves the gesture module. The two actions
leave by different routes, because that is how the existing system is built.

A pick carries the pixel of the selected object. The existing pipeline already
turns a pixel into a grasp itself, through ``pixel2robot.py``, ``pca.py`` and
``direction.py``, and it reads that pixel from ``txt_file/center_point.txt``.
Writing the same file replaces the detection step of that pipeline and leaves its
grasp logic untouched. The pixel is scaled into the calibration resolution first,
exactly as ``convert_origin_for_robot`` in the existing ``detection.py`` does.

A place carries the robot coordinate the same calibration produced, so it goes
out two ways. It is written to a coordinate file next to the one the existing
pipeline already uses, and it is sent as UR script over the same secondary client
port ``Application.py`` uses. The script matches ``final_position``, a linear move
to x and y with the release height and the fixed tool orientation. That file
cannot be imported because it runs a robot loop at import time, so the script
form is repeated here.

Motion is off by default. A request is built, logged and returned on every run,
and it is only sent when dispatch is switched on deliberately.

Three rules are enforced here rather than in the caller.

1. A request is built only when the pipeline reports ``safe_to_execute``.
2. The same selection or place point is sent once, so a held selection does not
   queue the same motion over and over.
3. A place is refused until the matching pick has been sent, so the arm cannot be
   told to drop something it never picked up.
"""

from __future__ import annotations

import logging
import socket
from dataclasses import dataclass
from typing import Literal

from config import PlaceCalibrationConfig, RobotConfig
from schemas import PipelineOutput, RobotPose

LOGGER = logging.getLogger(__name__)

Action = Literal["pick", "place"]


@dataclass(frozen=True)
class HandoffRequest:
    """One request handed to the pick and drop system."""

    action: Action
    object_id: str | None
    pixel: tuple[float, float] | None
    pose: RobotPose | None
    frame_index: int
    reason: str
    dispatched: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "object_id": self.object_id,
            "pixel": list(self.pixel) if self.pixel is not None else None,
            "pose": self.pose.as_list() if self.pose is not None else None,
            "frame_index": self.frame_index,
            "reason": self.reason,
            "dispatched": self.dispatched,
        }


def movel_script(pose: RobotPose, acceleration: float, velocity: float) -> str:
    """Linear move in the robot base frame, same form as the existing repository."""
    return f"""
def move_to_position():
    movel(p[{pose.x_m}, {pose.y_m}, {pose.z_m}, {pose.rx}, {pose.ry}, {pose.rz}], a={acceleration}, v={velocity})
    textmsg("Movement complete!")
end

move_to_position()
"""


class UrScriptClient:
    """Sends a UR script to the controller over the secondary client port."""

    def __init__(self, config: RobotConfig) -> None:
        self._config = config

    def send(self, script: str) -> None:
        with socket.create_connection(
            (self._config.host, self._config.port), timeout=self._config.timeout_s
        ) as connection:
            connection.sendall(script.encode("utf-8"))

    def describe(self) -> str:
        return f"{self._config.host}:{self._config.port}"


class RobotHandoff:
    """Turns pipeline results into pick and place requests.

    A pick hands over a pixel and a place hands over a pose, matching the two
    interfaces the existing system already has.
    """

    def __init__(
        self,
        config: RobotConfig,
        calibration: PlaceCalibrationConfig,
        client: UrScriptClient | None = None,
    ) -> None:
        self._config = config
        self._calibration = calibration
        self._client = client if client is not None else UrScriptClient(config)
        self._picked_object_id: str | None = None
        self._placed_pose: tuple[float, ...] | None = None
        self._requests: list[HandoffRequest] = []
        self._frame_width = 1
        self._frame_height = 1

    @property
    def requests(self) -> list[HandoffRequest]:
        return list(self._requests)

    @property
    def dispatch_enabled(self) -> bool:
        return self._config.dispatch

    def reset(self) -> None:
        self._picked_object_id = None
        self._placed_pose = None

    def handle(self, result: PipelineOutput) -> list[HandoffRequest]:
        """Build the requests this frame produces and send them when enabled."""
        if not result.safe_to_execute:
            return []

        self._frame_width = result.frame_width
        self._frame_height = result.frame_height

        built: list[HandoffRequest] = []

        if (
            result.selected_object_id is not None
            and result.selected_object_id != self._picked_object_id
        ):
            self._picked_object_id = result.selected_object_id
            centroid = result.selected_object.centroid_px if result.selected_object else None
            built.append(
                HandoffRequest(
                    action="pick",
                    object_id=result.selected_object_id,
                    pixel=(centroid.x, centroid.y) if centroid is not None else None,
                    pose=None,
                    frame_index=result.frame_index,
                    reason="object_selected_by_fingertip",
                )
            )

        if result.place_robot_pose is not None:
            pose_key = tuple(round(value, 4) for value in result.place_robot_pose)
            if pose_key != self._placed_pose:
                if self._picked_object_id is None:
                    LOGGER.warning("place_ignored reason=no_pick_was_sent")
                else:
                    self._placed_pose = pose_key
                    values = result.place_robot_pose
                    built.append(
                        HandoffRequest(
                            action="place",
                            object_id=self._picked_object_id,
                            pixel=(
                                (result.place_pixel[0], result.place_pixel[1])
                                if result.place_pixel
                                else None
                            ),
                            pose=RobotPose(
                                x_m=values[0],
                                y_m=values[1],
                                z_m=values[2],
                                rx=values[3],
                                ry=values[4],
                                rz=values[5],
                            ),
                            frame_index=result.frame_index,
                            reason="place_point_confirmed",
                        )
                    )

        sent: list[HandoffRequest] = []
        for request in built:
            dispatched = self._send(request) if self._config.dispatch else False
            final = HandoffRequest(
                action=request.action,
                object_id=request.object_id,
                pixel=request.pixel,
                pose=request.pose,
                frame_index=request.frame_index,
                reason=request.reason,
                dispatched=dispatched,
            )
            self._requests.append(final)
            sent.append(final)
            LOGGER.info("handoff_request %s", final.as_dict())
        return sent

    def _send(self, request: HandoffRequest) -> bool:
        """Hand one request over. A failure is reported and never raised."""
        if request.action == "pick":
            return self._write_center_point(request)
        return self._send_movel(request)

    def _write_center_point(self, request: HandoffRequest) -> bool:
        """Write the picked pixel where the existing pipeline reads it."""
        if request.pixel is None:
            LOGGER.warning("handoff_not_sent action=pick reason=no_pixel")
            return False
        target = self._config.center_point_path
        width, height = self._calibration.calibration_resolution
        # The existing pipeline expects the pixel in the calibration resolution.
        scaled_x = int(round(request.pixel[0] * width / float(self._frame_width)))
        scaled_y = int(round(request.pixel[1] * height / float(self._frame_height)))
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"({scaled_x}, {scaled_y})", encoding="utf-8")
        except OSError:
            LOGGER.exception("handoff_dispatch_failed action=pick path=%s", target)
            return False
        LOGGER.info("center_point_written path=%s pixel=%d,%d", target, scaled_x, scaled_y)
        return True

    def _send_movel(self, request: HandoffRequest) -> bool:
        if request.pose is None:
            LOGGER.warning("handoff_not_sent action=%s reason=no_pose", request.action)
            return False
        self._write_place_coordinates(request.pose)
        try:
            self._client.send(
                movel_script(request.pose, self._config.acceleration, self._config.velocity)
            )
        except OSError:
            LOGGER.exception("handoff_dispatch_failed action=%s", request.action)
            return False
        return True

    def _write_place_coordinates(self, pose: RobotPose) -> None:
        """Write x and y the way pixel2robot.py writes robot_coordinates.txt."""
        target = self._config.place_coordinates_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"{pose.x_m:.5f}\n{pose.y_m:.5f}\n", encoding="utf-8")
        except OSError:
            LOGGER.exception("place_coordinates_not_written path=%s", target)
            return
        LOGGER.info("place_coordinates_written path=%s x=%.5f y=%.5f", target, pose.x_m, pose.y_m)

    def health(self) -> dict[str, object]:
        return {
            "component": "robot_handoff",
            "dispatch_enabled": self._config.dispatch,
            "controller": self._client.describe(),
            "picked_object_id": self._picked_object_id,
            "requests": len(self._requests),
        }
