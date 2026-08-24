# MO_Changes
from __future__ import annotations

import cv2
import numpy as np


WINDOW_NAME = "Fingertip and object box match"
OBJECT_BOX = (300, 150, 700, 500)
FINGERTIP_HALF_SIZE = 18


class BoxMatchDemo:
    def __init__(self) -> None:
        self._fingertip_center = (500, 320)

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self._handle_mouse)
        while True:
            cv2.imshow(WINDOW_NAME, self._render())
            if cv2.waitKey(20) & 0xFF in {27, ord("q")}:
                break
        cv2.destroyAllWindows()

    def _handle_mouse(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        parameter: object,
    ) -> None:
        del flags, parameter
        if event == cv2.EVENT_MOUSEMOVE:
            self._fingertip_center = (x, y)

    def _render(self) -> np.ndarray:
        canvas = np.full((650, 1000, 3), 245, dtype=np.uint8)
        object_left, object_top, object_right, object_bottom = OBJECT_BOX
        fingertip_x, fingertip_y = self._fingertip_center
        fingertip_box = (
            fingertip_x - FINGERTIP_HALF_SIZE,
            fingertip_y - FINGERTIP_HALF_SIZE,
            fingertip_x + FINGERTIP_HALF_SIZE,
            fingertip_y + FINGERTIP_HALF_SIZE,
        )
        matched = self._point_inside_object(fingertip_x, fingertip_y)

        cv2.rectangle(
            canvas,
            (object_left, object_top),
            (object_right, object_bottom),
            (40, 170, 40),
            4,
        )
        cv2.rectangle(
            canvas,
            (fingertip_box[0], fingertip_box[1]),
            (fingertip_box[2], fingertip_box[3]),
            (220, 120, 20),
            3,
        )
        cv2.circle(canvas, self._fingertip_center, 7, (30, 30, 220), -1)

        status = "MATCH" if matched else "NO MATCH"
        status_color = (30, 150, 30) if matched else (30, 30, 220)
        cv2.putText(canvas, "Green: object box", (30, 45), 0, 0.8, (40, 170, 40), 2)
        cv2.putText(canvas, "Blue: fingertip box", (30, 80), 0, 0.8, (220, 120, 20), 2)
        cv2.putText(canvas, "Red: fingertip centre", (30, 115), 0, 0.8, (30, 30, 220), 2)
        cv2.putText(
            canvas,
            f"Fingertip centre: ({fingertip_x}, {fingertip_y})",
            (30, 575),
            0,
            0.8,
            (40, 40, 40),
            2,
        )
        cv2.putText(canvas, status, (760, 320), 0, 1.1, status_color, 3)
        cv2.putText(canvas, "Move the mouse", (760, 370), 0, 0.65, (40, 40, 40), 2)
        cv2.putText(canvas, "Press q to close", (760, 405), 0, 0.65, (40, 40, 40), 2)
        return canvas

    @staticmethod
    def _point_inside_object(x: int, y: int) -> bool:
        left, top, right, bottom = OBJECT_BOX
        return left <= x <= right and top <= y <= bottom


def main() -> None:
    BoxMatchDemo().run()


if __name__ == "__main__":
    main()
