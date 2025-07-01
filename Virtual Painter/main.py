import cv2
import numpy as np
import time
import HandTrackingModule as htm

# Setup
brushThickness = 7
eraserThickness = 50

xp, yp = 0, 0  # Previous coordinates

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.HandDetector(detectionCon=0.75, maxHands=1)

# Create a canvas for drawing
canvas = np.zeros((720, 1280, 3), np.uint8)

prevTime = 0

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if lmList and len(lmList) >= 21:
        # Get tip positions
        x1, y1 = lmList[8][1:]  # Index finger tip
        x2, y2 = lmList[12][1:]  # Middle finger tip

        # Check which fingers are up
        fingers = detector.fingersUp()
        
        if fingers[1] and not any(fingers[2:]):
            # Draw Mode: Only index finger up
            cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            cv2.putText(img, "Draw Mode", (10, 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
            cv2.line(canvas, (xp, yp), (x1, y1), (0, 0, 255), brushThickness)
            xp, yp = x1, y1

        elif fingers[1] and fingers[2]:
            # Eraser Mode: Index and middle finger up
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, "Eraser Mode", (10, 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
            cv2.line(canvas, (xp, yp), (x1, y1), (0, 0, 0), eraserThickness)
            xp, yp = x1, y1

        else:
            xp, yp = 0, 0  # Reset if no valid drawing finger pattern

    # Merge canvas and live frame
    grayCanvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(grayCanvas, 50, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, mask)
    img = cv2.bitwise_or(img, canvas)

    # FPS display
    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime
    cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Show result
    cv2.imshow("Virtual Painter", img)
    # cv2.imshow("Canvas", canvas)  # Uncomment to see canvas separately

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
