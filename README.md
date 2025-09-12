# Face Mask Detection (CNN + OpenCV)

This project detects whether a person is wearing a mask in real-time using a Convolutional Neural Network (CNN) and OpenCV.

## 📂 Project Structure
- `dataset/with_mask/` → Images of people with masks
- `dataset/without_mask/` → Images of people without masks
- `cnn_model.py` → Trains CNN model
- `train_model.py` → Alternate training script
- `real_time_detection.py` → Runs live mask detection with webcam
- `requirements.txt` → Required dependencies

## 🚀 How to Run

### 1. Clone Repository
```bash
git clone https://github.com/your-username/face-mask-detection.git
cd face-mask-detection
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python train_model.py
```

This will save the trained model as `mask_detector_model.h5`.

### 4. Run Real-Time Detection
```bash
python real_time_detection.py
```

Press **Q** to quit the webcam window.

---

## 📊 Model
- CNN architecture with Conv2D, MaxPooling, Dense layers
- Trained on dataset of masked & unmasked faces
- Uses Haar Cascade for face detection in real-time

---

## ✅ Output Example
- Shows **"Mask"** (green box) if mask is detected
- Shows **"No Mask"** (red box) if no mask is detected
