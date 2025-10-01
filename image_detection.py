import cv2
import cv2.data
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt

# Correct path (inside /content)
img_path = "/content/Live_face_mask_detection/masked.jpg"

# Load trained model
model = load_model('/content/Live_face_mask_detection/mask_detector_model.h5')

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read the image
img = cv2.imread(img_path)

if img is None:
    raise FileNotFoundError(f"⚠ Could not load image at {img_path}. Check the filename/path.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (100, 100))
    face = np.expand_dims(face / 255.0, axis=0)

    # Predict with model
    prediction = model.predict(face)
    label = 'Mask' if np.argmax(prediction) == 0 else 'No Mask'
    color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)

    cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# Show the result
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
