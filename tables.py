import cv2
import os
import numpy as np
from datetime import datetime
import mysql.connector

# -------- SETTINGS --------
DATASET_DIR = "faces"
DB_HOST = "localhost"
DB_USER = "root"       # Change if your MySQL user is different
DB_PASSWORD = "Pooja@91"
DB_NAME = "face_attendance"

face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------- DATABASE SETUP --------
def init_db():
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD
    )
    cursor = conn.cursor()
    # Create database if it doesn't exist
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
    conn.database = DB_NAME
    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INT AUTO_INCREMENT PRIMARY KEY,
            name VARCHAR(50),
            date DATE,
            time TIME,
            status VARCHAR(20)
        )
    """)
    conn.commit()
    return conn, cursor

# -------- MARK ATTENDANCE --------
def mark_attendance(cursor, conn, name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    # Check if already marked today
    cursor.execute("SELECT * FROM attendance WHERE name=%s AND date=%s", (name, date_str))
    row = cursor.fetchone()
    if row:
        print(f"ℹ️ {name} already marked today")
    else:
        cursor.execute(
            "INSERT INTO attendance (name, date, time, status) VALUES (%s,%s,%s,%s)",
            (name, date_str, time_str, "Present")
        )
        conn.commit()
        print(f"✅ Marked {name} as Present")

# -------- FACE REGISTRATION --------
def register_face(person_name):
    person_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0
    print("📸 Collecting 50 images. Press 'q' to quit early.")

    while count < 50:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            cv2.imwrite(os.path.join(person_dir, f"{count}.jpg"), face_img)
            count += 1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Collected {count} images for {person_name}")

# -------- TRAIN RECOGNIZER --------
def train_recognizer():
    # Check if opencv-contrib-python is installed for face module
    try:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
    except AttributeError:
        print("❌ Please install opencv-contrib-python: pip install opencv-contrib-python")
        return None, None

    faces = []
    labels = []
    label_map = {}
    current_label = 0

    for person_name in os.listdir(DATASET_DIR):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if os.path.isdir(person_dir):
            label_map[current_label] = person_name
            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = cv2.resize(img, (200, 200))
                faces.append(img)
                labels.append(current_label)
            current_label += 1

    if len(faces) == 0:
        print("❌ No faces found to train recognizer")
        return None, None

    recognizer.train(faces, np.array(labels))
    print("✅ Recognizer trained successfully")
    return recognizer, label_map

# -------- REAL-TIME ATTENDANCE --------
def recognize_faces():
    conn, cursor = init_db()
    recognizer, label_map = train_recognizer()
    if recognizer is None:
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not found")
        return

    print("📸 Starting attendance. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            label, confidence = recognizer.predict(face_img)
            name = label_map[label]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            if confidence < 85:  # Adjust threshold if needed
                mark_attendance(cursor, conn, name)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cursor.close()
    conn.close()

# -------- MAIN --------
if __name__ == "__main__":
    os.makedirs(DATASET_DIR, exist_ok=True)
    print("1. Register new face")
    print("2. Start attendance")
    choice = input("Enter choice: ")
    if choice == "1":
        person_name = input("Enter name: ")
        register_face(person_name)
    elif choice == "2":
        recognize_faces()
    else:
        print("❌ Invalid choice")
