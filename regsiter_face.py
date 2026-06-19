import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

# -------- SETTINGS --------
DATASET_DIR = "faces"
ATTENDANCE_FILE = "attendance.csv"
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------- FUNCTION TO REGISTER FACE --------
def register_face(person_name, num_images=20):
    person_dir = os.path.join(DATASET_DIR, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not found")
        return
    
    count = 0
    print(f"📸 Capturing images for {person_name}. Press 'q' to quit early.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            face_img = gray[y:y+h, x:x+w]
            img_path = os.path.join(person_dir, f"{count}.jpg")
            cv2.imwrite(img_path, face_img)
            count += 1

        cv2.imshow("Register Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= num_images:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Saved {count} images for {person_name}")

# -------- TRAIN LBPH FACE RECOGNIZER --------
def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
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
                faces.append(img)
                labels.append(current_label)
            current_label += 1

    if len(faces) == 0:
        print("❌ No faces found to train recognizer")
        return None, None

    recognizer.train(faces, np.array(labels))
    print("✅ Recognizer trained successfully")
    return recognizer, label_map

# -------- MARK ATTENDANCE --------
def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE)
    else:
        df = pd.DataFrame(columns=["Name", "Date", "Time", "Status"])

    # Avoid duplicates
    if not ((df["Name"] == name) & (df["Date"] == date_str)).any():
        df = pd.concat([df, pd.DataFrame([[name, date_str, time_str, "Present"]], columns=df.columns)], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        print(f"✅ Marked {name} as Present")
    else:
        print(f"ℹ️ {name} already marked today")

# -------- REAL-TIME ATTENDANCE --------
def recognize_faces():
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
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_img)
            name = label_map[label]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f"{name} ({int(confidence)})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            if confidence < 70:  # confidence threshold for recognition
                mark_attendance(name)

        cv2.imshow("Attendance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------- MAIN --------
if __name__ == "__main__":
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
