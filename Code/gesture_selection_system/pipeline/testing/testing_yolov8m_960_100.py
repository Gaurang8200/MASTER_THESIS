from ultralytics import YOLO
import cv2


MODEL_PATH = "/Users/gourangkumar/Desktop/Master_Thesis_Code/Code/gesture_selection_system/models/handgestureyolov8m960100.pt"

CAMERA_INDEX = 0
IMG_SIZE = 960

# General prediction threshold
CONFIDENCE = 0.55

# Extra class-wise thresholds
CLASS_THRESHOLDS = {
    "open_palm_start": 0.70,
    "pointing_finger": 0.70,
    "index_fingertip": 0.60,
}


def main():
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAMERA_INDEX}. "
            "Try CAMERA_INDEX = 0, 1, or 2."
        )

    print("Live detection started.")
    print("Press q to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame from camera.")
            break

        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONFIDENCE,
            verbose=False
        )

        result = results[0]
        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names[cls_id]

            if conf < CLASS_THRESHOLDS.get(label, CONFIDENCE):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            text = f"{label} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                frame,
                text,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

        cv2.imshow("Hand Gesture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()