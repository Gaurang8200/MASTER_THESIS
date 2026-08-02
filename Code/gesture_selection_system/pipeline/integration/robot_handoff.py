"""Handoff from the gesture pipeline to the existing pick and drop system.

This is the boundary where a confirmed selection leaves the gesture module. The
implementation here only records the request. It never moves the robot, and the
real connection to the existing repository is marked below.

Two rules are enforced here rather than in the caller.

1. A request is built only when the pipeline reports ``safe_to_execute``.
2. The same selection or place point is dispatched once, so a stable selection
   held for many frames does not queue the same motion over and over.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from schemas import PipelineOutput, RobotPose

LOGGER = logging.getLogger(__name__)

Action = Literal["pick", "place"]


@dataclass(frozen=True)
class HandoffRequest:
    """One request handed to the pick and drop system."""

    action: Action
    object_id: str | None
    pose: RobotPose | None
    frame_index: int
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "action": self.action,
            "object_id": self.object_id,
            "pose": self.pose.as_list() if self.pose is not None else None,
            "frame_index": self.frame_index,
            "reason": self.reason,
        }


@runtime_checkable
class RobotClient(Protocol):
    """The part of the existing pick and drop system this module needs."""

    def pick(self, object_id: str) -> None: ...

    def place(self, pose: RobotPose) -> None: ...


class MockRobotClient:
    """Records calls instead of moving the robot."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def pick(self, object_id: str) -> None:
        self.calls.append(("pick", object_id))
        LOGGER.info("mock_robot_pick object_id=%s", object_id)

    def place(self, pose: RobotPose) -> None:
        self.calls.append(("place", pose.as_list()))
        LOGGER.info("mock_robot_place pose=%s", pose.as_list())


class RobotHandoff:
    """Turns pipeline results into pick and place requests.

    Connect the existing system by passing its client as ``robot``. The client
    only has to offer ``pick(object_id)`` and ``place(pose)``, which keeps the
    gesture module independent from the transport the robot uses.
    """

    def __init__(self, robot: RobotClient | None = None, dispatch: bool = False) -> None:
        self._robot = robot if robot is not None else MockRobotClient()
        # Dispatch stays off by default so a demo run can never start a motion.
        self._dispatch = dispatch
        self._picked_object_id: str | None = None
        self._placed_pose: tuple[float, ...] | None = None
        self._requests: list[HandoffRequest] = []

    @property
    def requests(self) -> list[HandoffRequest]:
        return list(self._requests)

    def reset(self) -> None:
        self._picked_object_id = None
        self._placed_pose = None

    def handle(self, result: PipelineOutput) -> list[HandoffRequest]:
        """Build the requests this frame produces and optionally send them."""
        if not result.safe_to_execute:
            return []

        new_requests: list[HandoffRequest] = []

        if result.selected_object_id is not None and result.selected_object_id != self._picked_object_id:
            self._picked_object_id = result.selected_object_id
            pick_pose = (
                result.selected_object.pick_pose if result.selected_object is not None else None
            )
            new_requests.append(
                HandoffRequest(
                    action="pick",
                    object_id=result.selected_object_id,
                    pose=RobotPose(**pick_pose.model_dump()) if pick_pose is not None else None,
                    frame_index=result.frame_index,
                    reason="object_selected_by_fingertip",
                )
            )

        if result.place_robot_pose is not None:
            pose_key = tuple(round(value, 4) for value in result.place_robot_pose)
            if pose_key != self._placed_pose:
                self._placed_pose = pose_key
                values = result.place_robot_pose
                new_requests.append(
                    HandoffRequest(
                        action="place",
                        object_id=result.selected_object_id,
                        pose=RobotPose(
                            x_m=values[0],
                            y_m=values[1],
                            z_m=values[2],
                            rx_deg=values[3],
                            ry_deg=values[4],
                            rz_deg=values[5],
                        ),
                        frame_index=result.frame_index,
                        reason="place_point_confirmed",
                    )
                )

        for request in new_requests:
            self._requests.append(request)
            LOGGER.info("handoff_request %s", request.as_dict())
            if self._dispatch:
                self._send(request)

        return new_requests

    def _send(self, request: HandoffRequest) -> None:
        # Connect the existing pick and drop pipeline here. Nothing below this
        # line runs unless dispatch was switched on deliberately.
        try:
            if request.action == "pick" and request.object_id is not None:
                self._robot.pick(request.object_id)
            elif request.action == "place" and request.pose is not None:
                self._robot.place(request.pose)
        except Exception:
            LOGGER.exception("handoff_dispatch_failed action=%s", request.action)

    def health(self) -> dict[str, object]:
        return {
            "component": "robot_handoff",
            "dispatch_enabled": self._dispatch,
            "robot": type(self._robot).__name__,
            "requests": len(self._requests),
        }
