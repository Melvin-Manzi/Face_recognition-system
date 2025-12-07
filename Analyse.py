import cv2
import mediapipe as mp
import json

# Load model + label map
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("models/lbph_face_model.xml")

with open("models/label_map.json") as f:
    label_map = json.load(f)

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

cap = cv2.VideoCapture(0)

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
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

            # Predict person
            label, confidence = recognizer.predict(gray)
            name = label_map[str(label)]

            cv2.putText(frame, f"{name} ({int(confidence)})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
