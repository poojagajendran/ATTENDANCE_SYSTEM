import cv2
import datetime
from db_config import get_db_connection

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.2, 5)

    for (x,y,w,h) in faces:
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 60:
            now = datetime.datetime.now()
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO attendance (user_id, date, time) VALUES (%s,%s,%s)",
                (id_, now.date(), now.time())
            )
            conn.commit()
            conn.close()
            print("Attendanc e marked")
            cam.release()
            cv2.destroyAllWindows()
            exit()

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
