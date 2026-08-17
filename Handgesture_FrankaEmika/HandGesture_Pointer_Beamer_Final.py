# Standard Bibliotheken
import os
import platform
import time
from datetime import datetime  # Für Zeitstempel in Dateinamen

# Externe Bibliotheken
import cv2
from franky import Affine, CartesianMotion, JointMotion, Robot, ReferenceType
import numpy as np
import pyttsx3
import subprocess
from scipy.spatial.transform import Rotation

# Eigene Module
import HandTrackingModule as htm
import function_pool as fp
import pixel2robot as p2r

# ****************************************************************************** 
# Settings
# ****************************************************************************** 
robot = Robot("172.16.0.2")

# Fixed Z-height for all robot movements (in m)
FIXED_Z_HEIGHT = 0.04

# ****************************************************************************** 
# Tool Configuration (TCP - Tool Center Point)
# ****************************************************************************** 
# If you have a tool/gripper mounted, define the tool offset here
# Format: Affine([x, y, z, roll, pitch, yaw]) or Affine([x, y, z])

# TOOL_OFFSET = None  # Set to None to use flange as TCP (default)
# TOOL_OFFSET = Affine([0.0, 0.0, 0.15])  # 15cm in Z-Richtung

# Apply tool offset if defined
# if TOOL_OFFSET is not None:
  #  robot.set_ee(TOOL_OFFSET) # Falscher Befehl um Tool festzulegen -> Prüfen!
  #  print(f"Tool offset applied: {TOOL_OFFSET}")
# else:
  #  print("Using robot flange as TCP (no tool offset)")

# Movement dynamics
robot.relative_dynamics_factor = 0.1

wCam, hCam = 1280, 720

os_name = platform.system()
print(f"Running on: {os_name}")

camera = 0 # 0 = Extern, 1 = Intern

if os_name == "Windows":
    cap = cv2.VideoCapture(camera, cv2.CAP_DSHOW)
else:
    cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Cannot open camera {camera}")
    exit()
    
cap.set(3, wCam)
cap.set(4, hCam)

detector = htm.handDetector(maxHands=1)

pTime = 0
markedPoints = []
coordinate_pairs = []
holdTimeStart = 3
holdTimeStop = 3
startGestureDetectedTime = None
stopGestureDetectedTime = None
lastMarkedTime = time.time() - 10
isStarted = False
startDetectionDelay = 10
modelStartTime = None

# ****************************************************************************** 
# Sprach-Ausgabe
# ****************************************************************************** 
def speak(text):
    """Speak text using the appropriate method for the OS."""
    os_name = os.uname().sysname  # Bestimme das Betriebssystem
    
    if os_name == "Linux":
        # Verwende espeak auf Linux, falls verfügbar
        subprocess.call(['espeak', text])
    elif os_name == "Darwin":  # macOS
        subprocess.call(['say', text])
    elif os_name == "Windows":
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    else:
        print(f"Unsupported OS: {os_name}")


# ****************************************************************************** 
# Hand Gesture Detection
# ****************************************************************************** 
class Handgesture_detection():

    def __init__(self, robot):
        self.robot = robot


        # Erstelle Snapshot-Ordner wenn nicht vorhanden
        self.snapshot_dir = "snapshots"
        if not os.path.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)
            print(f"Created snapshot directory: {self.snapshot_dir}/")

    def save_snapshot(self, img, marked_point, robot_coords, point_number):
        """
        Speichert ein Snapshot-Bild mit Markierungen.
        
        Args:
            img: Das Kamerabild
            marked_point: Tuple (x, y) der markierten Pixelkoordinaten
            robot_coords: Tuple (x, y, z) der Roboterkoordinaten
            point_number: Nummer des markierten Punkts
        """
        # Erstelle Kopie des Bildes für Annotation
        snapshot = img.copy()
        
        # Zeichne markierten Punkt (großer roter Kreis)
        cv2.circle(snapshot, marked_point, 15, (0, 0, 255), -1)
        cv2.circle(snapshot, marked_point, 20, (0, 0, 255), 3)
        
        # Zeichne Fadenkreuz
        cv2.line(snapshot, (marked_point[0] - 30, marked_point[1]), 
                 (marked_point[0] + 30, marked_point[1]), (0, 0, 255), 2)
        cv2.line(snapshot, (marked_point[0], marked_point[1] - 30), 
                 (marked_point[0], marked_point[1] + 30), (0, 0, 255), 2)
        
        # Text mit Koordinaten
        pixel_text = f"Pixel: ({marked_point[0]}, {marked_point[1]})"
        robot_text = f"Robot: ({robot_coords[0]:.3f}, {robot_coords[1]:.3f}, {robot_coords[2]:.3f})"
        point_text = f"Point #{point_number}"
        
        # Hintergrund für Text (bessere Lesbarkeit)
        cv2.rectangle(snapshot, (10, 10), (550, 100), (0, 0, 0), -1)
        
        # Text schreiben
        cv2.putText(snapshot, point_text, (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(snapshot, pixel_text, (20, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(snapshot, robot_text, (20, 85), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Dateiname mit Zeitstempel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_point{point_number:02d}_{timestamp}.png"
        filepath = os.path.join(self.snapshot_dir, filename)
        
        # Bild speichern
        cv2.imwrite(filepath, snapshot)
        print(f"Snapshot saved: {filepath}")
        
        return filepath

# Positionierung des Roboters 

    def move_to_mainPosition(self):

        print("Moving to main position")

        # Roboter Geschwindigkeit
        self.robot.relative_dynamics_factor = 0.1

        # Alte Position ueber Schachbrett
        # m_jp1 = JointMotion([-0.00539203, -0.63654644, -0.02320701, -2.29697155, -0.04513998,  1.61391624,
        #                     0.78993209])

        m_jp1 = JointMotion([ 0.13702847, -0.2575123,  0.5149722, -1.97199728,  0.14447674,  1.71555068,
        1.40325516])
        
        # Execute movement
        self.robot.move(m_jp1)
        print(f"Successfully reached main position")


    def run(self):
        global isStarted, startGestureDetectedTime, modelStartTime, stopGestureDetectedTime
        global lastMarkedTime, markedPoints, pTime, coordinate_pairs

        cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Image", wCam, hCam)


        prev_fingers = [0,0,0,0,0]
        mark_cooldown = 1.0
        while True:
            success, img = cap.read()
            if not success:
                print("Failed to grab frame")
                continue

            img = cv2.flip(img, 1)  # Bild spiegeln
            # is_flipped = True  # Setze auf False wenn Flip deaktiviert wird

            img = detector.findHands(img, draw=True)
            lmList, bbox = detector.findPosition(img, draw=False)
            # First compute raw finger detection to populate history,
            # then use temporally smoothed finger states.
            if len(lmList) != 0:
                raw_fingers = detector.fingersUp()
                fingers = detector.fingersUpStable()
                print("Fingers:", fingers, "Raw:", raw_fingers)
            else:
                fingers = []

            # --- Start-Geste: Zeige + Mittelfinger ausstrecken ---
            if not isStarted:
                # Start when index + middle finger are stably extended
                if lmList and len(fingers) >= 3 and fingers[1] == 1 and fingers[2] == 1:
                    if startGestureDetectedTime is None:
                        startGestureDetectedTime = time.time()
                        print("Start gesture timer started...")
                    elif time.time() - startGestureDetectedTime > holdTimeStart:
                        isStarted = True
                        modelStartTime = time.time()
                        print("Start gesture detected. Point detection activated.")
                        speak("Starting the point detecting Model.")
                        startGestureDetectedTime = None
                else:
                    startGestureDetectedTime = None
            
            # --- Markierungs-Logik (nur wenn gestartet) ---
            else:  # isStarted == True
                if len(lmList) != 0 and len(fingers) == 5:
                    x1, y1 = lmList[8][1:]
                    # Visual feedback: draw index fingertip
                    cv2.circle(img, (x1, y1), 8, (0, 255, 0), cv2.FILLED)
                    cv2.circle(img, (x1, y1), 12, (0, 255, 0), 2)

                    # Mark point when middle finger becomes extended (rising edge)
                    # while index finger remains extended
                    if len(prev_fingers) == 5:
                        # Rising edge: middle finger goes from 0 to 1
                        if prev_fingers[2] == 0 and fingers[2] == 1 and fingers[1] == 1:
                            if (time.time() - lastMarkedTime) >= mark_cooldown:
                                
                                # Bild: Y nach unten (0=oben, 720=unten)
                                # Roboter: Y nach oben (muss invertiert werden)

                                # y1_transformed = hCam - y1  # Y-Achse invertieren: 720 - y
                                x1_transformed = wCam - x1  # X-Achse invertieren: 1280 - x

                                markedPoint = (x1, y1)
                                markedPoints.append(markedPoint)
                                print(f"Marked Point: {markedPoint}")
                                speak("Point detected")

                                # Hole aktuelle Roboterpose (nur für Snapshot)
                                cartesian_state = self.robot.current_cartesian_state
                                ee_pose = cartesian_state.pose.end_effector_pose
                                translation = ee_pose.translation

                                print(f"Current EE Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}] m")

                                # Pixel → Roboter-Koordinaten speichern
                                robot_x, robot_y, _ = p2r.pixel2robot(x1_transformed, y1, 15)

                                coordinate_pairs.append((robot_x, robot_y, FIXED_Z_HEIGHT))
                                print(f"Saved Robot Coordinates: X={robot_x:.3f}, Y={robot_y:.3f}, Z={FIXED_Z_HEIGHT:.3f} (fixed)")

                                # Snapshot speichern# **NEU: Speichere Snapshot**
                                point_number = len(coordinate_pairs)
                                self.save_snapshot(img, markedPoint, 
                                                 (robot_x, robot_y, FIXED_Z_HEIGHT), 
                                                 point_number)

                                lastMarkedTime = time.time()
                
                # --- Stop-Geste (Faust) ---
                if len(fingers) == 5 and all(finger == 0 for finger in fingers):
                    if stopGestureDetectedTime is None:
                        stopGestureDetectedTime = time.time()
                        print("Stop gesture timer started...")
                    elif time.time() - stopGestureDetectedTime > holdTimeStop:
                        print("Stop gesture detected. Executing robot movements.")
                        speak("Stopping detection. Now moving to all points.")
                        self.execute_robot_movements()
                        break
                else:
                    stopGestureDetectedTime = None

            # update previous fingers for edge detection
            if len(fingers) == 5:
                prev_fingers = fingers.copy()

            # --- Visualisierung ---
            for point in markedPoints:
                cv2.circle(img, point, 5, (0, 0, 0), cv2.FILLED)
            cv2.putText(img, f'Points: {len(markedPoints)}', (20, 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (20, 100),
                        cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # --- Nach Stop-Geste: Punkte abfahren --

    def execute_robot_movements(self):
        global coordinate_pairs
        if not coordinate_pairs:
            print("No points to move to.")
            speak("No points to move to.")
            return

        # Debug-Ausgabe der Punkte
        print(f"\n{'='*50}")
        print(f"Starting robot movement sequence...")
        print(f"Total points to visit: {len(coordinate_pairs)}")
        print(f"Movement mode: Absolute")
        print(f"Coordinate system: Robot Base Frame")
        print(f"{'='*50}\n")

        # Set dynamics factor for safe movements
        self.robot.relative_dynamics_factor = 0.05

        successful_moves = 0
        failed_moves = 0

        # TCP-Orientierung: Konvertiere Euler-Winkel zu Quaternion
        # Senkrecht nach unten: Roll=180°, Pitch=0°, Yaw=0°
        
        euler_angles = [3.14, 0.0, 0.0]  # [roll, pitch, yaw] in radians
        rotation = Rotation.from_euler('xyz', euler_angles)
        quaternion = rotation.as_quat()  # returns [x, y, z, w]
        
        print(f"Using orientation: Roll={euler_angles[0]:.2f}, Pitch={euler_angles[1]:.2f}, Yaw={euler_angles[2]:.2f}")
        print(f"Quaternion: [{quaternion[0]:.3f}, {quaternion[1]:.3f}, {quaternion[2]:.3f}, {quaternion[3]:.3f}]\n")

        for i, (x, y, z) in enumerate(coordinate_pairs):
            try:
                print(f"\n[{i+1}/{len(coordinate_pairs)}] Moving to point: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
                speak(f"Moving to point {i+1}")

                # Create Affine with translation and quaternion
                translation = np.array([x, y, z])
                motion = CartesianMotion(Affine(translation, quaternion), ReferenceType.Absolute)

                # Execute movement
                self.robot.move(motion)
                print(f"Successfully reached point {i+1}")
                successful_moves += 1

            except Exception as e:
                failed_moves += 1
                print(f"Failed to move to point {i+1}: {e}")
                speak(f"Error at point {i+1}")
                
                # Stop sequence after error for safety
                print("Stopping movement sequence for safety.")
                break

        print(f"\n{'='*50}")
        print(f"Movement sequence completed.")
        print(f"Successful: {successful_moves}/{len(coordinate_pairs)}")
        print(f"Failed: {failed_moves}/{len(coordinate_pairs)}")
        print(f"{'='*50}\n")
        
        if failed_moves == 0:
            speak("All points reached successfully!")
        else:
            speak(f"Completed with {failed_moves} errors.")


# ****************************************************************************** 
# Main
# ****************************************************************************** 
if __name__ == '__main__':

    # Franka Emika verbinden
    try:
        robot = Robot("172.16.0.2")
        print(f"Connected to Franka Emika at 172.16.0.2")
        
        # Robot is ready after connection
        print("Robot initialized and ready")
        
        # move_to_mainPosition(robot)

        move_to_main = input("Move to main position? (y/n): ")
        if move_to_main.lower() == 'y':
            hand_detection = Handgesture_detection(robot)
            hand_detection.move_to_mainPosition()   

        # Gestenerkennung starten
        hand_detection = Handgesture_detection(robot)
        hand_detection.run()

        move_to_main = input("Move back to main position? (y/n): ")
        if move_to_main.lower() == 'y':
            hand_detection = Handgesture_detection(robot)
            hand_detection.move_to_mainPosition()   
        
    except Exception as e:
        print(f"Failed to connect to robot: {e}")
        print("Please check:")
        print("  - Robot is powered on")
        print("  - Network connection to 172.16.0.2")
        print("  - Robot is unlocked and in User mode")
 