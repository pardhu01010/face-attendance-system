# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "deepface>=0.0.98",
#     "mediapipe==0.10.9",
#     "numpy>=2.1.3",
#     "pandas>=2.3.3",
#     "scipy>=1.15.3",
#     "tf-keras>=2.19.0",
# ]
# ///

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from deepface import DeepFace
from similarity import cosine_sim
from datetime import datetime

# ================= CONFIG =================
THRESHOLD = 0.7
# FRAME_SKIP = 5
FRAME_SKIP = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX

# -------- UI / STATE --------
last_name = None
last_color = (0, 255, 0)
status_text = ""
status_color = (255, 255, 255)
status_timer = 0


mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

db = {
    f.replace(".npy", ""): np.load(os.path.join("embeddings", f))
    for f in os.listdir("embeddings")
}

if not os.path.exists("attendance.csv"):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(
        "attendance.csv", index=False
    )

cap = cv2.VideoCapture(0)
frame_count = 0

print("Authenticating... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
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

            # -------- DEFAULT LABEL FROM STATE --------
            if last_name:
                label = last_name
                color = last_color
            else:
                label = "Unknown"
                color = (0, 0, 255)

            # -------- RECOGNITION (NOT EVERY FRAME) --------
            if frame_count % FRAME_SKIP == 0:
                try:
                    emb = DeepFace.represent(
                        face,
                        model_name="Facenet",
                        enforce_detection=False
                    )[0]["embedding"]
                except:
                    break

                matched = False
                for name, ref in db.items():
                    sim = cosine_sim(emb, ref)

                    if sim > THRESHOLD:
                        matched = True
                        last_name = f"{name} ({sim:.2f})"
                        last_color = (0, 255, 0)

                        now = datetime.now()
                        date = now.strftime("%Y-%m-%d")
                        time = now.strftime("%H:%M:%S")

                        df = pd.read_csv("attendance.csv")

                        if not ((df.Name == name) & (df.Date == date)).any():
                            df.loc[len(df)] = [name, date, time]
                            df.to_csv("attendance.csv", index=False)
                            status_text = "Attendance Marked"
                            status_color = (0, 255, 0)
                        else:
                            status_text = "Already Marked"
                            status_color = (0, 255, 255)

                        status_timer = 30
                        break

                if not matched:
                    last_name = None

            # -------- DRAW BOX & LABEL --------
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
            cv2.putText(frame, label, (x, y - 10), FONT, 0.7, color, 2)
            break  # one face only

    # -------- STATUS MESSAGE (GLOBAL) --------
    if status_timer > 0:
        cv2.putText(
            frame,
            status_text,
            (20, 40),
            FONT,
            1.0,
            status_color,
            3
        )
        status_timer -= 1

    cv2.imshow("Face Authentication", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
