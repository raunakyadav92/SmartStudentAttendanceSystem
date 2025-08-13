# recognize_and_mark.py
import cv2
import os
import time
import csv
import sqlite3
import json
from datetime import datetime
import pandas as pd

TRAINER_PATH = "trainer"
TRAINER_FILE = os.path.join(TRAINER_PATH, "trainer.yml")
LABELS_FILE = os.path.join(TRAINER_PATH, "labels.json")
CSV_DIR = "attendance_csv"
DB_FILE = os.path.join("db", "attendance.db")

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs("db", exist_ok=True)

# load model and labels
if not os.path.exists(TRAINER_FILE) or not os.path.exists(LABELS_FILE):
    print("Train the model first (trainer/trainer.yml missing).")
    exit(1)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(TRAINER_FILE)
with open(LABELS_FILE, "r") as f:
    label_map = json.load(f)   # numeric id (str) -> name

# invert label_map keys to int
label_map = {int(k): v for k, v in label_map.items()}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# initialize DB
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    date TEXT,
    time TEXT,
    timestamp INTEGER
)
""")
conn.commit()

def mark_attendance(name):
    """Mark attendance in CSV (one per date) and SQLite (no duplicate same day/time)."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    ts = int(time.time())

    # CSV
    csv_path = os.path.join(CSV_DIR, f"attendance_{date_str}.csv")
    # Check for existing entry today for same name (to avoid duplicates)
    exists = False
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if ((df['name'] == name) & (df['date'] == date_str)).any():
            exists = True

    if not exists:
        # append to CSV
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['name', 'date', 'time', 'timestamp'])
            writer.writerow([name, date_str, time_str, ts])

        # insert into sqlite
        c.execute("INSERT INTO attendance (name, date, time, timestamp) VALUES (?, ?, ?, ?)",
                  (name, date_str, time_str, ts))
        conn.commit()
        print(f"[{date_str} {time_str}] Attendance marked for {name}")
    else:
        # Already marked today
        pass

# recognition loop
cap = cv2.VideoCapture(0)
print("Starting webcam. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200,200))
        label_id, confidence = recognizer.predict(face_img)
        # lower confidence = better match for LBPH; threshold depends on your data
        if confidence < 70:
            name = label_map.get(label_id, "Unknown")
            cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            mark_attendance(name)
        else:
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
