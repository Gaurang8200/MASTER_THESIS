import cv2

def find_camera():
    """
    Try camera indices from 0 to 5 to find an active camera.
    """
    for index in range(10):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            print(f"Camera found at index {index}")
            return cap, index
        cap.release()
    print("No camera found on indices 0-5.")
    return None, None

def capture_image():
    """
    Open camera, show live feed, and save an image on spacebar press.
    """
    cap, index = find_camera()
    if cap is None:
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        cv2.imshow('Camera Feed - Press Space to Capture, ESC to Exit', frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC key to exit
            break
        if key == 32:  # Spacebar to capture image
            image_path = f"captured_image_index_{index}.png"
            cv2.imwrite(image_path, frame)
            print(f"Image saved as {image_path}")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    capture_image()
