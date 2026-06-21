# 😷 Face Mask Detection using CNN & OpenCV

## 📖 Project Overview

This project is a real-time Face Mask Detection system developed using Convolutional Neural Networks (CNN) and OpenCV. The model detects whether a person is wearing a face mask and provides instant visual feedback through webcam-based monitoring.

The system was designed to demonstrate the application of Deep Learning and Computer Vision in public health and safety monitoring.

---

## 🎯 Project Objectives

- Detect faces in real time using a webcam.
- Classify faces as **Mask** or **No Mask**.
- Apply Deep Learning techniques for image classification.
- Demonstrate practical use of Computer Vision in safety compliance monitoring.

---

## 🛠️ Technologies Used

- Python
- OpenCV
- TensorFlow
- Keras
- NumPy
- CNN (Convolutional Neural Network)
- Haar Cascade Classifier

---

## 📂 Dataset

The dataset contains two categories:

```text
dataset/
├── with_mask/
└── without_mask/
```

### Classes

- With Mask
- Without Mask

---

## 🏗️ Project Architecture

### Training Phase

1. Image Collection
2. Data Preprocessing
3. CNN Model Training
4. Model Evaluation
5. Model Saving

### Detection Phase

1. Webcam Input
2. Face Detection using Haar Cascade
3. Mask Classification using CNN
4. Real-Time Prediction Display

---

## 📁 Repository Structure

```text
Live_face_mask_detection/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── cnn_model.py
├── train_model.py
├── real_time_detection.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/jpSharma123-sudo/Live_face_mask_detection.git
cd Live_face_mask_detection
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train the Model

```bash
python train_model.py
```

### Run Real-Time Detection

```bash
python real_time_detection.py
```

Press **Q** to close the webcam window.

---

## 🧠 Model Details

- CNN-based binary image classification
- Trained on masked and unmasked face images
- Real-time prediction using webcam feed
- Face detection using OpenCV Haar Cascades

---

## 📊 Output

The system displays:

✅ **Mask Detected**

❌ **No Mask Detected**

in real time with bounding boxes around detected faces.

---

## 💡 Applications

- Public Health Monitoring
- Smart Surveillance Systems
- Workplace Safety Compliance
- Airport & Railway Screening
- Educational Deep Learning Projects

---

## 🔮 Future Improvements

- Mobile Application Integration
- YOLO-based Face Detection
- Multi-Face Detection Optimization
- Cloud Deployment
- Higher Accuracy Deep Learning Models

---

## 👨‍💻 Author

**Jaiprakash Sharma**

Aspiring Data Scientist | Machine Learning & Computer Vision Enthusiast

🔗 GitHub: https://github.com/jpSharma123-sudo

---

## 📄 License

This project is licensed under the MIT License.
