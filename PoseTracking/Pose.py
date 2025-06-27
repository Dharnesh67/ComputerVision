import cv2
import mediapipe as mp
from ultralytics import YOLO

# Load YOLOv8 model (use 'yolov8n.pt' or 'yolov8s.pt' for better speed)
yolo_model = YOLO("yolov8n.pt")  # Make sure this file is in your directory or provide full path

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# Open video
cap = cv2.VideoCapture("PoseTracking/Video/Video2.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect people using YOLOv8
    results = yolo_model(frame)[0]
    for det in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        if int(class_id) == 0 and score > 0.5:  # Class 0 is 'person'
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Crop person region
            person_img = frame[y1:y2, x1:x2]
            if person_img.size == 0:
                continue

            # Pose estimation
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            results_pose = pose.process(person_rgb)

            if results_pose.pose_landmarks:
                mp_draw.draw_landmarks(person_img, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Replace modified region back to original frame
                frame[y1:y2, x1:x2] = person_img

    frame = cv2.resize(frame, (1000, 600))
    cv2.imshow("Multi-Person Pose Tracking", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
