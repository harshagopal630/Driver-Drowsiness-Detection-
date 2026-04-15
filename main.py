import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained CNN model
model = load_model("model.h5")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Resize image for CNN input
        face = cv2.resize(face, (64, 64))
        face = face / 255.0
        face = np.reshape(face, (1, 64, 64, 3))

        prediction = model.predict(face)

        if prediction[0][0] > 0.5:
            label = "Drowsy"
            color = (0, 0, 255)
        else:
            label = "Alert"
            color = (0, 255, 0)

        cv2.rectangle(frame,
                      (x, y),
                      (x+w, y+h),
                      color,
                      2)

        cv2.putText(frame,
                    label,
                    (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
