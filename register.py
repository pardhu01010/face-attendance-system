# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "deepface>=0.0.98",
#     "mediapipe==0.10.9",
#     "tf-keras>=2.19.0",
# ]
# ///

import cv2
import mediapipe as mp
import numpy as np
import os
from deepface import DeepFace

# ================= CONFIG =================
SAMPLES_REQUIRED = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX
# =========================================

name = input("Enter user name: ").strip()
os.makedirs("embeddings", exist_ok=True)

# MediaPipe face detector
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

cap = cv2.VideoCapture(0)
embeddings = []

print("Registering face... Look at the camera")

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_face.process(rgb)

    if result.detections:
        for det in result.detections:
            box = det.location_data.relative_bounding_box
            h, w, _ = frame.shape

            x = int(box.xmin * w)
            y = int(box.ymin * h)
            bw = int(box.width * w)
            bh = int(box.height * h)

            face = frame[y:y + bh, x:x + bw]

            try:
                emb = DeepFace.represent(
                    face,
                    model_name="Facenet",
                    enforce_detection=False
                )[0]["embedding"]

                embeddings.append(emb)
                print(f"Captured {len(embeddings)}")
            except:
                continue

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + bw, y + bh),
                (0, 255, 0),
                2
            )

            # Status text
            cv2.putText(
                frame,
                f"Capturing {len(embeddings)}/{SAMPLES_REQUIRED}",
                (x, y - 10),
                FONT,
                0.7,
                (0, 255, 0),
                2
            )

            break  # only one face

    else:
        cv2.putText(
            frame,
            "No face detected",
            (20, 40),
            FONT,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("Face Registration", frame)

    if len(embeddings) >= SAMPLES_REQUIRED:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================= SAVE EMBEDDING =================
cap.release()
cv2.destroyAllWindows()

mean_embedding = np.mean(embeddings, axis=0)
np.save(f"embeddings/{name}.npy", mean_embedding)

print(f"Registration complete for {name}")
