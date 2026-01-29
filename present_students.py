# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "deepface>=0.0.98",
#     "scipy>=1.15.3",
#     "ultralytics>=8.4.8",
#     "tf-keras>=2.19.0",
# ]
# ///
import cv2
import numpy as np
import pandas as pd
import os
from deepface import DeepFace
from similarity import cosine_sim
from ultralytics import YOLO
from datetime import datetime

# ================= CONFIG =================
THRESHOLD = 0.7
FRAME_SKIP = 10               
FONT = cv2.FONT_HERSHEY_SIMPLEX


# Load YOLO (person detector)
yolo = YOLO("yolov8n.pt")

# Load attendance (only today)
today = datetime.now().strftime("%Y-%m-%d")
df = pd.read_csv("attendance.csv")
present_students = df[df["Date"] == today]["Name"].unique().tolist()

print("Present students:", present_students)

# Load embeddings of present students only
db = {
    name: np.load(os.path.join("embeddings", f"{name}.npy"))
    for name in present_students
    if os.path.exists(os.path.join("embeddings", f"{name}.npy"))
}

cap = cv2.VideoCapture(0)
frame_count = 0

# -------- STATE CACHE --------
last_results = {}  # key: box index → (label, color)


# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    results = yolo(frame, verbose=False)

    for idx, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        face = frame[y1:y2, x1:x2]

        # -------- DEFAULT FROM CACHE --------
        label, color = last_results.get(
            idx, ("Unknown", (0, 0, 255))
        )

        # -------- RUN EMBEDDING EVERY N FRAMES --------
        if frame_count % FRAME_SKIP == 0:
            try:
                emb = DeepFace.represent(
                    face,
                    model_name="Facenet",
                    enforce_detection=False
                )[0]["embedding"]
            except:
                continue

            matched = False
            for name, ref in db.items():
                sim = cosine_sim(emb, ref)
                if sim > THRESHOLD:
                    label = name
                    color = (0, 255, 0)
                    matched = True
                    break

            if not matched:
                label = "Unknown"
                color = (0, 0, 255)

            # update cache
            last_results[idx] = (label, color)

        # -------- DRAW --------
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            FONT,
            0.7,
            color,
            2
        )

    cv2.imshow("Present Students Viewer", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
