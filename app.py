import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from typing import List, Annotated
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from deepface import DeepFace
from datetime import datetime
from threading import Lock
from similarity import cosine_sim

# ================= CONFIG =================
THRESHOLD = 0.7
os.makedirs("embeddings", exist_ok=True)

if not os.path.exists("attendance.csv"):
    pd.DataFrame(columns=["Name", "Date", "Time"]).to_csv(
        "attendance.csv", index=False
    )

# Disable GPU/X11 requirements for OpenCV on server
os.environ["OPENCV_VIDEOIO_PRIORITY_BACKEND"] = "0"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

app = FastAPI(title="Face Attendance API")

# Initialize global models
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

# Global DB and lock for concurrency
db_lock = Lock()
db = {
    f.replace(".npy", ""): np.load(os.path.join("embeddings", f))
    for f in os.listdir("embeddings") if f.endswith(".npy")
}

def get_face_roi(image_np):
    """Detect face using MediaPipe and return the cropped face image."""
    rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    result = mp_face.process(rgb)
    if not result.detections:
        return None
    
    # Take the first detection
    det = result.detections[0]
    box = det.location_data.relative_bounding_box
    h, w, _ = image_np.shape
    x = int(box.xmin * w)
    y = int(box.ymin * h)
    bw = int(box.width * w)
    bh = int(box.height * h)
    
    # Ensure bounds
    x = max(0, x)
    y = max(0, y)
    bw = min(w - x, bw)
    bh = min(h - y, bh)
    
    face = image_np[y:y + bh, x:x + bw]
    if face.size == 0:
        return None
    return face

@app.get("/")
def health_check():
    """Health check for Render deployment"""
    return {"status": "ok", "message": "Face Attendance API is running"}

@app.post("/register/")
def register_user(
    name: Annotated[str, Form(...)], 
    files: Annotated[List[UploadFile], File(...)]
):
    """Register a new user by analyzing a list of images."""
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    if not name:
        raise HTTPException(status_code=400, detail="Name is required.")

    embeddings = []
    
    for file in files:
        # Read image to numpy backend
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue
            
        face = get_face_roi(img)
        if face is None:
            continue
            
        try:
            emb = DeepFace.represent(
                face,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]
            embeddings.append(emb)
        except Exception as e:
            continue
            
    if not embeddings:
        raise HTTPException(status_code=400, detail="Could not extract face or generate embeddings from the provided images.")
        
    mean_embedding = np.mean(embeddings, axis=0)
    
    # Save to disk
    np.save(os.path.join("embeddings", f"{name}.npy"), mean_embedding)
    
    # Update global cache thread-safely
    with db_lock:
        db[name] = mean_embedding
        
    return {"status": "success", "message": f"Successfully registered user: {name}", "samples_used": len(embeddings)}

@app.post("/authenticate/")
def authenticate_user(file: UploadFile = File(...)):
    """Authenticate a face and mark attendance."""
    contents = file.file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
        
    face = get_face_roi(frame)
    if face is None:
        return JSONResponse(status_code=400, content={"status": "failure", "message": "No face detected in the image."})
        
    try:
        emb = DeepFace.represent(
            face,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": "Error extracting embedding from face."})
        
    matched = False
    best_match_name = None
    best_sim = 0.0
    
    with db_lock:
        local_db = db.copy()
        
    for db_name, ref in local_db.items():
        sim = cosine_sim(emb, ref)
        if sim > THRESHOLD and sim > best_sim:
            matched = True
            best_match_name = db_name
            best_sim = sim
            
    if not matched:
        return JSONResponse(status_code=401, content={"status": "failure", "message": "User not recognized."})
        
    # User is matched, log attendance
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    with db_lock:
        df = pd.read_csv("attendance.csv")
        # Check if already marked today
        if not ((df.Name == best_match_name) & (df.Date == date_str)).any():
            new_row = pd.DataFrame([{"Name": best_match_name, "Date": date_str, "Time": time_str}])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv("attendance.csv", index=False)
            status_text = "Attendance Marked"
        else:
            status_text = "Already Marked"
            
    return {"status": "success", "name": best_match_name, "similarity": float(best_sim), "message": status_text}

@app.get("/attendance/")
def get_attendance(date: str = None):
    """Get list of users marked present for a specific date (defaults to today)."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
        
    with db_lock:
        df = pd.read_csv("attendance.csv")
        
    present = df[df["Date"] == date].to_dict(orient="records")
    return {"date": date, "attendance": present}
