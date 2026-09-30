from __future__ import annotations

import cv2
import depthai as dai


SENSOR_SIZE = (1920, 1080)
FRAME_SIZE = (1280, 720)


def main() -> None:
    device = dai.Device()
    pipeline = dai.Pipeline(device)
    started = False
    try:
        camera = pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A,
            sensorResolution=SENSOR_SIZE,
            sensorFps=30,
        )
        output = camera.requestOutput(
            size=FRAME_SIZE,
            type=dai.ImgFrame.Type.BGR888p,
            fps=30,
            enableUndistortion=False,
        )
        queue = output.createOutputQueue(maxSize=1, blocking=False)
        pipeline.start()
        started = True
        size_printed = False

        while True:
            frame = queue.get().getCvFrame()
            height, width = frame.shape[:2]
            if (width, height) != FRAME_SIZE:
                raise RuntimeError(
                    f"OAK D returned {width} x {height}, expected 1280 x 720"
                )
            if not size_printed:
                print(f"OAK D RGB frame: {width} x {height}")
                size_printed = True
            cv2.imshow("OAK D RGB 1280 x 720", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        cv2.destroyAllWindows()
        if started:
            pipeline.stop()
            pipeline.wait()
        device.close()


if __name__ == "__main__":
    main()
