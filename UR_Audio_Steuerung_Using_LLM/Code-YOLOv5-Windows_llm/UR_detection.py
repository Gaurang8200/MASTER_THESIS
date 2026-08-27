import cv2
import os
import subprocess
import time
import sys

# === Configuration Constants ===
SAVE_DIRECTORY = 'photos'
DETECTION_SCRIPT = 'yolov5/detect.py'
WEIGHTS = 'yolov5/my_model.pt'
LABEL_FILE_PATH = 'txt_file/label_path.txt'
CENTER_POINT_SAVE_PATH = 'txt_file/center_point.txt'
CAMERA_INDEX = 0
#IMAGE_RESOLUTION = (640, 480)
ROBOT_RESOLUTION = (2560, 1472)
IMAGE_RESOLUTION = (2560, 1472)


# === 1. Take Photo and Run Detection ===
def take_photo_and_run_detection():
    """Capture a photo from the camera and run object detection."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1472)

    os.makedirs(SAVE_DIRECTORY, exist_ok=True)

    photo_count = 0

    try:
        while True:
            time.sleep(5)  # Wait before taking a photo

            ret, frame = cap.read()
            if not ret:
                print("Error: Unable to capture frame")
                break

            photo_count += 1
            photo_path = os.path.join(SAVE_DIRECTORY, f'photo_{photo_count}.jpg')
            cv2.imwrite(photo_path, frame)
            print(f"Photo {photo_count} saved at {photo_path}")

            # Display the captured frame
            cv2.imshow('Frame', frame)

            run_detection(photo_path)
            break  # Capture and detect once, then exit loop

    finally:
        cap.release()
        cv2.destroyAllWindows()


# === 2. Run YOLOv5 Detection ===
def run_detection(photo_path):
    """Run the YOLOv5 detection script on the captured photo."""
    detection_command = [
        sys.executable,  # Ensure the current Python environment is used
        DETECTION_SCRIPT,
        '--weights', WEIGHTS,
        '--img', '416',
        '--save-txt', '--save-crop',
        '--conf', '0.25',
        '--source', photo_path
    ]
    try:
        subprocess.run(detection_command, check=True)
        print(f"Detection completed for {photo_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running detection: {e}")


# === 3. Get Label File Path ===
def get_label_path():
    """Retrieve the label file path from a text file."""
    try:
        with open(LABEL_FILE_PATH, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: Label file not found at {LABEL_FILE_PATH}")
        return None
    except Exception as e:
        print(f"Error reading label path: {e}")
        return None


# === 4. Calculate Object Origin ===
def calculate_origin(file_path):
    """Calculate the center point of the detected object."""
    try:
        with open(file_path, 'r') as file:
            content = file.read().strip()

        numbers = list(map(int, content.split()))
        if len(numbers) != 5:
            raise ValueError("File does not contain exactly 5 numbers")

        x = (numbers[1] + numbers[3]) / 2
        y = (numbers[2] + numbers[4]) / 2
        #print(x, y)
        return (x, y)
        

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except ValueError as e:
        print(f"Error processing label file: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# === 5. Convert Origin for Robot Coordinates ===
def convert_origin_for_robot(origin):
    """Convert the origin point to robot coordinate space."""
    try:
        x_robot = int((origin[0] * ROBOT_RESOLUTION[0]) / IMAGE_RESOLUTION[0])
        y_robot = int((origin[1] * ROBOT_RESOLUTION[1]) / IMAGE_RESOLUTION[1])
        #print(x_robot, y_robot)
        return (x_robot, y_robot)
    except Exception as e:
        print(f"Error converting origin for robot: {e}")
        return None


# === 6. Save Center Point ===
def save_center_point(center_point):
    """Save the calculated robot center point to a text file."""
    try:
        with open(CENTER_POINT_SAVE_PATH, 'w') as file:
            file.write(f"{center_point[0]} {center_point[1]}")
        print(f"Center point saved at {CENTER_POINT_SAVE_PATH}")
    except Exception as e:
        print(f"Error saving center point: {e}")


# === 7. Main Workflow ===
def main():
    """Main workflow: capture photo, run detection, process results."""
    take_photo_and_run_detection()

    label_path = get_label_path()
    if label_path:
        origin = calculate_origin(label_path)
        if origin:
            origin_for_robot = convert_origin_for_robot(origin)
            if origin_for_robot:
                save_center_point(origin_for_robot)


# === Entry Point ===
if __name__ == "__main__":
    main()
