import cv2
import os

# -------- SETTINGS --------
PERSON_NAME = "pooja"
SAVE_DIR = f"faces/{PERSON_NAME}"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("❌ Camera not opening")
    exit()

print("📸 Camera started. Press 's' to save face, 'q' to quit.")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face_img = frame[y:y+h, x:x+w]

    cv2.imshow("Capture Face", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s') and len(faces) > 0:
        img_path = f"{SAVE_DIR}/{count}.jpg"
        cv2.imwrite(img_path, face_img)
        print(f"✅ Saved {img_path}")
        count += 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("📁 Face capture completed")
