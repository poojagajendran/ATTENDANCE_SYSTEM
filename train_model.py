import cv2
import os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = []
ids = []

for user_id in os.listdir("dataset"):
    for img in os.listdir(f"dataset/{user_id}"):
        path = f"dataset/{user_id}/{img}"
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        faces.append(image)
        ids.append(int(user_id))

recognizer.train(faces, np.array(ids))
recognizer.save("trainer.yml")

print("Model trained successfully")
