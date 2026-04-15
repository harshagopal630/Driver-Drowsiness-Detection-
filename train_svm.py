import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# Image size
img_size = 64

data = []
labels = []

# Dataset path
dataset_path = "dataset/train"

categories = ["alert", "drowsy"]

for category in categories:

    path = os.path.join(dataset_path, category)
    label = categories.index(category)

    for img in os.listdir(path):

        img_path = os.path.join(path, img)

        try:
            image = cv2.imread(img_path)
            image = cv2.resize(image, (img_size, img_size))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            data.append(image.flatten())
            labels.append(label)

        except:
            pass

data = np.array(data)
labels = np.array(labels)

print("Training SVM Model...")

# Train SVM
model = svm.SVC(kernel="linear")

model.fit(data, labels)

# Save model
with open("svm_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("SVM Model saved as svm_model.pkl")
