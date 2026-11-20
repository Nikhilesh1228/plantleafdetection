# 🌿 PlantDoc: AI-Powered Plant Leaf Disease Detection & Diagnosis

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An intelligent agricultural computer vision system designed to detect, segment, and classify plant leaf diseases from digital photographs, providing automated diagnostic feedback and targeted treatment recommendations.

---

## 📌 Project Abstract
Crop loss due to leaf diseases is a major threat to agricultural productivity and food security. Early, accurate diagnosis is essential for minimizing pesticide misuse and preventing crop failure. 

**PlantDoc** integrates classical **Digital Image Processing (DIP)** with **Deep Convolutional Neural Networks (CNNs)** to:
1. Isolate diseased foliage against complex agricultural backgrounds.
2. Segment necrotic and chlorotic lesion regions using adaptive thresholding and color-space analysis.
3. Classify pathogen classes (bacterial, fungal, viral, and healthy leaves) with high classification accuracy.
4. Provide immediate actionable organic and chemical remediation guidelines via an interactive user dashboard.

---

## 🔬 Computer Vision & Image Processing Pipeline

```
  [Input Leaf Image]
          │
          ▼
┌───────────────────────────────────────────┐
│ 1. Digital Image Preprocessing (OpenCV)   │
│    • Resizing & Aspect Ratio Correction   │
│    • Gaussian Denoising & Bilateral Blur  │
│    • Color Transformation (RGB ➔ HSV/Lab) │
└─────────────────────┬─────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────┐
│ 2. Feature Extraction & Segmentation      │
│    • Otsu's Adaptive Thresholding         │
│    • Morphological Closing & Opening      │
│    • Lesion Contour Area Extraction       │
└─────────────────────┬─────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────┐
│ 3. Deep Learning Classifier (CNN)         │
│    • Convolution & Pooling Layers         │
│    • Dropout Regularization & Batch Norm  │
│    • Softmax Multi-Class Probability      │
└─────────────────────┬─────────────────────┘
                      │
                      ▼
┌───────────────────────────────────────────┐
│ 4. Diagnosis & Remediation Output         │
│    • Disease Identification & Severity    │
│    • Chemical / Organic Treatment Plan    │
│    • Interactive Streamlit Visualization  │
└───────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Domain | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Primary Language** | Python 3.9+ | Core application logic and model development |
| **Image Processing** | OpenCV (`cv2`), PIL | Denoising, color space conversion, edge detection, morphological filtering |
| **Deep Learning** | TensorFlow / Keras, PyTorch | CNN design, feature mapping, transfer learning, and inference |
| **Scientific Computing** | NumPy, SciPy, Pandas | Array transformations, matrix math, data structuring |
| **Evaluation & Metrics** | Scikit-Learn, Matplotlib, Seaborn | Confusion matrices, loss curves, precision, recall, F1-score |
| **Deployment / UI** | Streamlit | Interactive web application for real-time leaf image diagnosis |

---

## ✨ Key Technical Highlights

### 1. Classical Image Processing (OpenCV)
* **Color Space Analysis:** Converts images from standard `RGB` to `HSV` and `CIE Lab` to isolate chlorophyll breakdown (yellowing/browning) from green healthy tissue.
* **Noise Suppression:** Employs **Gaussian & Bilateral Filtering** to smooth texture noise while preserving sharp lesion edges.
* **Morphological Operations:** Uses dilation and erosion kernel filters to remove background artifacts and close segmentation holes.

### 2. Deep Convolutional Architecture
* **Convolutional Layers:** Extracts hierarchical spatial features from micro-lesions to full-leaf patterns.
* **Data Augmentation:** Prevents overfitting using real-time spatial transformations (random rotations, affine transformations, horizontal/vertical flips, and brightness jittering).
* **Evaluation Metrics:** Evaluated using categorical cross-entropy loss, accuracy, precision, recall, and multi-class confusion matrices.

### 3. Interactive Web Dashboard
* Allows users to drag-and-drop leaf photos from mobile cameras or field datasets.
* Real-time visual pipeline: displays the original image, segmented lesion mask, predicted diagnosis, confidence score, and remediation steps.

---

## 📂 Project Structure

```text
plantleafdetection/
├── app.py                      # Streamlit interactive application
├── requirements.txt            # Project dependencies
├── models/
│   └── plant_disease_model.h5  # Trained model weights (or download script)
├── notebooks/
│   └── model_training.ipynb    # Model architecture, training, and metrics
├── src/
│   ├── preprocessing.py        # OpenCV image preprocessing & segmentation
│   └── inference.py            # Model loading and prediction pipeline
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Nikhilesh1228/plantleafdetection.git
cd plantleafdetection
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser to test leaf images.

---

## 📊 Sample Dependencies (`requirements.txt`)
```text
streamlit>=1.30.0
tensorflow>=2.12.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
pillow>=10.0.0
matplotlib>=3.7.0
scikit-learn>=1.2.0
```

---

## 🔮 Future Scope
* **Grad-CAM Saliency Maps:** Visualizing activation heatmaps directly on leaf images to verify model explainability.
* **Edge Deployment:** Quantizing weights into **TensorFlow Lite (`.tflite`)** for offline execution on embedded devices (Raspberry Pi, mobile devices).
* **Multi-Crop Expansion:** Expanding training datasets across additional regional crops and soil-borne deficiencies.

---

## 👤 Author
**Nikhilesh Sriram Ruthala**  
* GitHub: [@Nikhilesh1228](https://github.com/Nikhilesh1228)
