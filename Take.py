import cv2
import mediapipe as mp
import os

# Create dataset directory
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

person_name = input("Enter person's name: ")
person_path = os.path.join(DATASET_DIR, person_name)
os.makedirs(person_path, exist_ok=True)

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)
count = 0

print("Capturing images. Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x1 = int(bboxC.xmin * w)
            y1 = int(bboxC.ymin * h)
            x2 = x1 + int(bboxC.width * w)
            y2 = y1 + int(bboxC.height * h)

            face_crop = frame[y1:y2, x1:x2]
            face_crop_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

            file_path = os.path.join(person_path, f"{count}.jpg")
            cv2.imwrite(file_path, face_crop_gray)
            count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Capture Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
