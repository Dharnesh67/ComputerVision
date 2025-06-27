import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)


mpFaceMesh = mp.solutions.face_mesh
FaceMesh = mpFaceMesh.FaceMesh(
    # static_image_mode=False,
    # max_num_faces=1,
    # refine_landmarks=True,
    # min_detection_confidence=0.5,
    # min_tracking_confidence=0.5,
)
mpDraw= mp.solutions.drawing_utils
prevtime = 0
currenttime = 0
while True:
    success, img = cap.read()
    currenttime = time.time()
    fps = 1 / (currenttime - prevtime)
    prevtime = currenttime
    cv.putText(
        img, f"FPS: {int(fps)}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = FaceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            # print(faceLms.landmark)
            mpDraw.draw_landmarks(
                img, faceLms, mpFaceMesh.FACEMESH_TESSELATION,
                mpDraw.DrawingSpec(color=(255,0, 0), thickness=1, circle_radius=1),
                mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
            )
    else:
        cv.putText(img, "No Face Detected", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
