import cv2
import os
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load dataset
data = []
labels = []
categories = ['with_mask', 'without_mask']

for category in categories:
    path = os.path.join('dataset', category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img))
            resized_img = cv2.resize(img_array, (100, 100))
            data.append(resized_img)
            labels.append(categories.index(category))
        except:
            print(f"⚠ Skipping corrupted image: {img}")
            pass

X = np.array(data) / 255.0
y = to_categorical(labels, num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # 2 classes
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)
model.save('mask_detector_model.h5')
print("✅ Model trained and saved as mask_detector_model.h5")
