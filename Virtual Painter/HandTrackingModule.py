"""
Hand Tracking Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone
"""

import cv2
import mediapipe as mp
import time
from typing import List, Tuple
class HandDetector:
    def __init__(
        self,
        mode: bool = False,
        maxHands: int = 2,
        detectionCon: float = 0.5,
        trackCon: float = 0.5,
    ):
        """
        Initializes the hand detector.
        """
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.results = None

    def findHands(self, img, draw: bool = True):
        """
        Processes an image and draws hand landmarks if found.
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        img, handLms, self.mpHands.HAND_CONNECTIONS
                    )
        return img
         
    def fingersUp(self) -> List[int]:
        """
        Returns a list indicating which fingers are up.
        1 means finger is up, 0 means down.
        Order: [Thumb, Index, Middle, Ring, Pinky]
        """
        fingers = []
        if not self.results or not self.results.multi_hand_landmarks:
            return [0, 0, 0, 0, 0]
        myHand = self.results.multi_hand_landmarks[0]
        lmList = []
        for id, lm in enumerate(myHand.landmark):
            lmList.append((id, lm.x, lm.y))
        if not lmList:
            return [0, 0, 0, 0, 0]
        # Thumb
        if lmList[4][1] > lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Fingers
        for tipId in [8, 12, 16, 20]:
            if lmList[tipId][2] < lmList[tipId - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findPosition(
        self, img, handNo: int = 0, draw: bool = True
    ) -> List[Tuple[int, int, int]]:
        """
        Returns a list of landmark positions for the specified hand.
        """
        lmList = []
        if self.results and self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                h, w, _ = img.shape
                for id, lm in enumerate(myHand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((id, cx, cy))
                    if draw:
                        cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
        return lmList


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    detector = HandDetector()
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame.")
            break

        # if we provide the `draw` parameter, it will draw the landmarks on the image otherwise it will just return the image without drawing
        img = detector.findHands(img)

        lmList = detector.findPosition(img, draw=False)
        if lmList:
            print(lmList[4])  # Tip of the thumb

        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime

        cv2.putText(
            img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3
        )

        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
