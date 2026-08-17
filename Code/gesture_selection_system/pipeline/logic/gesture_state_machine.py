"""Selection mode state machine.

Pure policy with no camera, model or robot dependency, which keeps the mode
rules fully testable. The stop gesture always wins over the start gesture in the
same frame because the safe state is selection mode off.

A gesture counts once it has been held for the configured number of seconds.
Time rather than frame count, so the operator gets the same feel on a fast
machine and on a slow one.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import StabilityConfig
from schemas import GestureFrame, GestureName, ModeTransition, SelectionMode

GESTURE_CLASSES = frozenset(GestureName)


@dataclass(frozen=True)
class StateUpdate:
    """Outcome of one state machine step."""

    mode: SelectionMode
    transition: ModeTransition | None
    reason: str
    open_palm_held_s: float
    closed_palm_held_s: float
    seconds_without_gesture: float


class GestureStateMachine:
    """Turns selection mode on and off from held gestures."""

    def __init__(self, stability: StabilityConfig) -> None:
        self._stability = stability
        self._mode = SelectionMode.OFF
        self._open_since: float | None = None
        self._closed_since: float | None = None
        self._last_gesture_at: float | None = None

    @property
    def mode(self) -> SelectionMode:
        return self._mode

    def reset(self) -> None:
        self._mode = SelectionMode.OFF
        self._open_since = None
        self._closed_since = None
        self._last_gesture_at = None

    def update(self, frame: GestureFrame, now: float) -> StateUpdate:
        """Advance the state machine with one detector result.

        ``now`` is a monotonic timestamp in seconds.
        """
        if not frame.ok:
            # A failed inference is treated as a frame with no evidence. It never
            # keeps a hold alive, so a broken detector drifts toward mode off.
            return self._step(False, False, False, now)

        return self._step(
            open_palm=frame.has(GestureName.OPEN_PALM_START),
            closed_palm=frame.has(GestureName.CLOSED_PALM_STOP),
            any_gesture=any(d.gesture in GESTURE_CLASSES for d in frame.detections),
            now=now,
        )

    def _step(
        self, open_palm: bool, closed_palm: bool, any_gesture: bool, now: float
    ) -> StateUpdate:
        if closed_palm:
            if self._closed_since is None:
                self._closed_since = now
            self._open_since = None
        elif open_palm:
            if self._open_since is None:
                self._open_since = now
            self._closed_since = None
        else:
            self._open_since = None
            self._closed_since = None

        if any_gesture or self._last_gesture_at is None:
            self._last_gesture_at = now

        open_held = now - self._open_since if self._open_since is not None else 0.0
        closed_held = now - self._closed_since if self._closed_since is not None else 0.0
        idle_for = now - self._last_gesture_at

        transition: ModeTransition | None = None
        reason = "no_change"

        if closed_held >= self._stability.deactivate_seconds:
            self._closed_since = None
            if self._mode is SelectionMode.ON:
                self._mode = SelectionMode.OFF
                transition = ModeTransition.DEACTIVATED
                reason = "closed_palm_stop"
        elif open_held >= self._stability.activate_seconds:
            self._open_since = None
            if self._mode is SelectionMode.OFF:
                self._mode = SelectionMode.ON
                transition = ModeTransition.ACTIVATED
                reason = "open_palm_start"

        timeout = self._stability.lost_gesture_timeout_seconds
        if (
            transition is None
            and timeout > 0.0
            and self._mode is SelectionMode.ON
            and idle_for >= timeout
        ):
            self._mode = SelectionMode.OFF
            transition = ModeTransition.DEACTIVATED
            reason = "gesture_lost_timeout"

        return StateUpdate(
            mode=self._mode,
            transition=transition,
            reason=reason,
            open_palm_held_s=round(open_held, 2),
            closed_palm_held_s=round(closed_held, 2),
            seconds_without_gesture=round(idle_for, 2),
        )
