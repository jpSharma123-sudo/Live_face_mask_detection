import cv2
import cv2.data
from keras.models import load_model
import numpy as np

# Load trained model
model = load_model('mask_detector_model.h5')
if model is None:
    raise RuntimeError("❌ Failed to load the mask detector model. Please check the model file path.")

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)  # Webcam

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠ Failed to grab frame from webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        face = np.expand_dims(face / 255.0, axis=0)

        prediction = model.predict(face)
        label = 'Mask' if np.argmax(prediction) == 0 else 'No Mask'
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)

        if label == 'No Mask':
            print("⚠ No Mask Detected!")

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Mask Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
