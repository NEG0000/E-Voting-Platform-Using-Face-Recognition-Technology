import cv2
import numpy as np
import os
import pickle
from sklearn import preprocessing

# Path to dataset
dataset_path = "dataset"

# Initialize face detector and recognizer
cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Prepare data and labels
faces = []
labels = []
label_names = []

for aadhar_id in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, aadhar_id)
    if not os.path.isdir(person_dir):
        continue

    for filename in os.listdir(person_dir):
        img_path = os.path.join(person_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in detected_faces:
            faces.append(gray[y:y+h, x:x+w])
            labels.append(aadhar_id)

# Encode labels
if not faces or not labels:
    raise ValueError("No face data found. Please check the dataset.")

le = preprocessing.LabelEncoder()
encoded_labels = le.fit_transform(labels)

# Train recognizer
recognizer.train(faces, np.array(encoded_labels))
recognizer.save("Trained.yml")

# Save the label encoder
with open("encoder.pkl", "wb") as f:
    pickle.dump(le, f)

"Training complete. Files created: Trained.yml and encoder.pkl"
