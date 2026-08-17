# Standard Bibliotheken
from collections import deque
import math
import time


# Externe Bibliotheken
import cv2
import mediapipe as mp
import numpy as np


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=float(self.detectionCon),
            min_tracking_confidence=float(self.trackCon)
        )
        self.mpDraw = mp.solutions.drawing_utils

        self.results = None
        self.lmList = []
        self.handLabel = "Right"

        # history buffer for temporal smoothing of finger states
        self.finger_history = deque(maxlen=7)

        # Landmark IDs
        self.tipIds = [4, 8, 12, 16, 20]


        """     def detectHandSide(self):
        if len(self.lmList) < 6:
            return "Right"

        thumb_mcp_x = self.lmList[2][1]    # Thumb MCP
        index_mcp_x = self.lmList[5][1]    # Index MCP

        # Wenn der Daumen weiter rechts im Bild ist → linke Hand
        if thumb_mcp_x > index_mcp_x:
            return "Left"
        else:
            return "Right" """
        
    def detectHandSideMP(self, handNo):
        if self.results and self.results.multi_handedness:
            try:
                return self.results.multi_handedness[handNo].classification[0].label
            except:
                pass
        return "Right"

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results and self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        self.lmList = []
        bbox = []
        self.bbox = None

        if self.results and self.results.multi_hand_landmarks:

            myHand = self.results.multi_hand_landmarks[handNo]

            xList, yList = [], []
            h, w, c = img.shape

            for id, lm in enumerate(myHand.landmark):
                px, py = int(lm.x * w), int(lm.y * h)
                xList.append(px)
                yList.append(py)
                self.lmList.append([id, px, py])

                if draw:
                    cv2.circle(img, (px, py), 5, (255, 0, 255), cv2.FILLED)

            self.handLabel = self.detectHandSideMP(handNo)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = (xmin, ymin, xmax, ymax)

            if draw:
                cv2.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20), (0,255,0), 2)
            # store bbox for downstream usage (e.g. scale-dependent thresholds)
            self.bbox = (xmin, ymin, xmax, ymax)

        return self.lmList, bbox
    # --------------------------------------------------------------
    #                   Vektorbasierte Fingererkennung
    # --------------------------------------------------------------

    def angle(self, a, b, c):
        """Berechnet Winkel ABC in Grad."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ab = a - b
        cb = c - b

        dot = ab.dot(cb)
        norm = np.linalg.norm(ab) * np.linalg.norm(cb)
        if norm == 0:
            return 0

        return np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0)))
    

    def fingersUp(self):
        if len(self.lmList) == 0:
            return [0,0,0,0,0]

        fingers = [0,0,0,0,0]
        
        # Get wrist position as reference point
        try:
            wrist = self.lmList[0][1:3]  # [x, y]
        except Exception:
            return [0,0,0,0,0]
        
        # ===== OTHER FINGERS (Index 1-4) =====
        # Use distance-based comparison: TIP should be farther from wrist than PIP
        # This works regardless of hand orientation
        tip_ids = [8, 12, 16, 20]   # Index, Middle, Ring, Pinky tips
        pip_ids = [6, 10, 14, 18]   # PIP joints
        
        for i in range(4):
            try:
                tip = self.lmList[tip_ids[i]][1:3]
                pip = self.lmList[pip_ids[i]][1:3]
                
                # Calculate distances from wrist
                tip_dist = math.hypot(tip[0] - wrist[0], tip[1] - wrist[1])
                pip_dist = math.hypot(pip[0] - wrist[0], pip[1] - wrist[1])
                
                # Extended finger: tip is farther from wrist than PIP
                if tip_dist > pip_dist:
                    fingers[i+1] = 1
            except Exception:
                continue
        
        # ===== THUMB (Index 0) =====
        # Use distance comparison: thumb tip should be far from index finger MCP
        try:
            thumb_tip = self.lmList[4][1:3]
            thumb_ip = self.lmList[3][1:3]
            index_mcp = self.lmList[5][1:3]
            
            # Distance from thumb tip to index MCP
            tip_to_index = math.hypot(thumb_tip[0] - index_mcp[0], thumb_tip[1] - index_mcp[1])
            # Distance from thumb IP to index MCP
            ip_to_index = math.hypot(thumb_ip[0] - index_mcp[0], thumb_ip[1] - index_mcp[1])
            
            # Extended thumb: tip is farther from index MCP than IP joint
            if tip_to_index > ip_to_index:
                fingers[0] = 1
        except Exception:
            pass
        
        # append to temporal history for smoothing
        try:
            self.finger_history.append(fingers.copy())
        except Exception:
            pass

        print("Fingers:", fingers, "Hand:", self.handLabel)
        return fingers

    def fingersUpStable(self, min_votes=4):
        """Return a temporally smoothed finger state.

        Each finger is 1 if it appears as 1 in at least `min_votes`
        entries of the history buffer. If history is small, return the
        latest detected state.
        """
        if len(self.finger_history) == 0:
            return [0, 0, 0, 0, 0]

        # if not enough history, just return the most recent
        if len(self.finger_history) < min_votes:
            return list(self.finger_history[-1])

        sums = [0, 0, 0, 0, 0]
        for entry in self.finger_history:
            for i, v in enumerate(entry):
                sums[i] += int(bool(v))

        stable = [1 if s >= min_votes else 0 for s in sums]
        return stable



    # --------------------------------------------------------------
    #                   Abstand zwischen 2 Landmarken
    # --------------------------------------------------------------

    def findDistance(self, p1, p2, img, draw=True, r=10, t=3):
        if len(self.lmList) <= max(p1, p2):
            return 0, img, []

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        if not success:
            print("Kamerafehler")
            break

        img = cv2.flip(img, 1)  # Bild spiegeln


        img = detector.findHands(img, draw=True)
        lmList, bbox = detector.findPosition(img)

        if len(lmList) != 0:
            print("Daumenspitze:", lmList[4])

        cTime = time.perf_counter()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
