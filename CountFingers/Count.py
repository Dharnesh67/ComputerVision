import cv2 
import time 
import numpy as np
import HandTrackingModule as htm
import math
import os
cap= cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 1280)   # Set height
path = os.path.join(os.path.dirname(__file__), "FingerImages")
mylist = os.listdir(path)
overlaylist = []
for imgPath in mylist:
    image = cv2.imread(os.path.join(path, imgPath))
    overlaylist.append(image)  # Append each image to the overlay list

# print(mylist)  # List all images in the FingerImages directory
Detector = htm.HandDetector(detectionCon=0.7
                            , maxHands=2)

prevtime=0
# Flip image for mirror effect
while  True:
    success, img =  cap.read()
    if not success:
        continue
    img=cv2.flip(img, 1)  # Flip the image horizontally
    img=Detector.findHands(img)  # Detect hands in the image
    lms=Detector.findPosition(img, draw=False)  # Get the landmarks of the detected hands
    # Resize the overlay image to a smaller size (e.g., 150x150)
    fingers = []
    if len(lms) != 0:
        if lms[8][2] < lms[6][2]:
            fingers.append(1)
        if lms[12][2] < lms[10][2]:
            fingers.append(1)
        if lms[16][2] < lms[14][2]:
            fingers.append(1)
        if lms[20][2] < lms[18][2]:
            fingers.append(1)
        if lms[4][1] < lms[2][1]:
            fingers.append(1)
    lenfinger=len(fingers)  # Count the number of fingers detected
    if lenfinger == 0:
        lenfinger = 1
    overlay_resized = cv2.resize(overlaylist[lenfinger-1], (200, 300))
    img[0:overlay_resized.shape[0], 0:overlay_resized.shape[1]] = overlay_resized  # Place the resized overlay on the image
    ctime = time.time()
    fps=1/(ctime-prevtime)
    prevtime=ctime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)  # Display FPS on the image
    cv2.imshow("Image",  img)  # Display the image with detected hands
    cv2.waitKey(1)