# BISINDO Classifier - Enhanced Computer Vision Project
**Indonesian Sign Language Recognition System**

Created by: **Krisna Santosa**  
Course: **Computer Vision - Final Project**  
Date: December 31, 2025

---

## 📋 Project Overview

This project implements a comprehensive **BISINDO (Bahasa Isyarat Indonesia)** classifier that demonstrates various Computer Vision techniques learned throughout the course (Pertemuan 1-16). The system recognizes Indonesian sign language gestures in real-time using both traditional machine learning and deep learning approaches.

---

## 🎯 Computer Vision Techniques Implemented

### 1. **Image Formation & Camera Model** (Pertemuan 1-2)
- Camera intrinsic and extrinsic parameters
- Pinhole camera model implementation
- Image coordinate systems
- Lens distortion handling

### 2. **Image Preprocessing** (Pertemuan 3-4)
- RGB/BGR color space conversion
- Brightness and contrast adjustment
- Saturation control
- Grayscale conversion

### 3. **Histogram Processing** (Pertemuan 5)
- Histogram equalization for contrast enhancement
- YUV color space histogram manipulation
- Histogram analysis and visualization

### 4. **Thresholding Techniques** (Pertemuan 6)
- Binary thresholding
- Adaptive thresholding (Gaussian)
- Otsu's method for automatic threshold selection

### 5. **Image Segmentation** (Pertemuan 7)
- Hand region segmentation using convex hull
- Background subtraction
- Mask-based segmentation
- ROI (Region of Interest) extraction

### 6. **Feature Extraction** (Pertemuan 8-9)
- MediaPipe hand landmarks detection (21 keypoints)
- 3D coordinate extraction (x, y, z)
- Feature normalization
- 126-dimensional feature vector (21 points × 3 coords × 2 hands)

### 7. **Object Detection & Tracking** (Pertemuan 10-11)
- Real-time hand detection using MediaPipe
- Multi-hand tracking (up to 2 hands)
- Confidence-based detection thresholds
- Frame-by-frame tracking with motion prediction

### 8. **3D Reconstruction & Visualization** (Pertemuan 12)
- 3D landmark projection
- Depth-based color coding
- Z-coordinate visualization
- Spatial relationship representation

### 9. **Augmented Reality** (Pertemuan 13)
- AR overlay with bounding boxes
- Real-time label rendering
- Confidence score display
- Interactive UI elements

### 10. **Object Recognition - Machine Learning** (Pertemuan 14)
- Random Forest classifier
- Feature importance analysis
- Ensemble learning approach
- Cross-validation

### 11. **CNN (Convolutional Neural Network)** (Pertemuan 15-16)
- 1D Convolutional layers for sequence processing
- Batch Normalization
- Dropout for regularization
- Multi-layer architecture
- Adam optimizer
- Categorical cross-entropy loss

---



## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
pip
webcam (for real-time detection)
```

### Install Dependencies
```bash
pip install streamlit opencv-python mediapipe numpy pandas scikit-learn joblib matplotlib seaborn tensorflow
```

---

## 💻 Usage

### 1. Run Enhanced Dashboard (Real-time Detection)
```bash
streamlit run dashboard_enhanced.py
```

**Features:**
- Real-time hand gesture recognition
- Multiple visualization modes:
  - ✅ Histogram Equalization
  - ✅ Adaptive Thresholding
  - ✅ Hand Segmentation
  - ✅ 3D Projection
  - ✅ AR Overlay
- Adjustable camera settings (brightness, contrast, saturation)
- Confidence threshold tuning
- FPS counter
- Live prediction with confidence scores

### 2. Run Jupyter Notebook (Training & Analysis)
```bash
jupyter notebook BISINDO-CLASSIFIER-ENHANCED.ipynb
```

**Notebook Includes:**
- Data exploration and visualization
- Preprocessing demonstrations
- Feature correlation analysis
- Random Forest training
- CNN model architecture and training
- Model comparison and evaluation
- Confusion matrices
- Classification reports

---

## 📊 Dataset

- **Total Samples**: 10,145 hand gesture samples
- **Classes**: 26 (A-Z letters in BISINDO)
- **Features**: 126 dimensions (21 landmarks × 3 coordinates × 2 hands)
- **Data Split**:
  - Training: 70%
  - Validation: 15%
  - Test: 15%

---

## 🎯 Model Performance

### Random Forest Classifier
- **Algorithm**: Ensemble of 100 decision trees
- **Test Accuracy**: ~99% (depends on data quality)
- **Pros**: Fast inference, interpretable, robust
- **Cons**: Limited feature learning

### CNN Model
- **Architecture**: 
  - Conv1D layers (64, 128, 256 filters)
  - Batch Normalization
  - Dropout (0.3-0.5)
  - Dense layers (512, 256)
  - Softmax output
- **Test Accuracy**: Competitive with RF
- **Pros**: Automatic feature learning, better generalization
- **Cons**: Requires more training time

---

## 🔧 Configuration

### Dashboard Settings
- **Min Detection Confidence**: 0.0 - 1.0 (default: 0.5)
- **Min Tracking Confidence**: 0.0 - 1.0 (default: 0.5)
- **Brightness**: -100 to +100 (default: 0)
- **Contrast**: -100 to +100 (default: 0)
- **Saturation**: -100 to +100 (default: 0)

### CNN Training Parameters
- **Batch Size**: 32
- **Epochs**: 50
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Early Stopping**: Patience 10

---

## 📸 Screenshots & Demos

### Visualization Modes

1. **Normal Mode**: Standard hand landmark detection
2. **Histogram Equalization**: Enhanced contrast for better feature visibility
3. **Thresholding**: Binary representation for edge detection
4. **Segmentation**: Isolated hand region
5. **3D Projection**: Depth-aware landmark visualization
6. **AR Overlay**: Professional bounding boxes with labels

---

## 🧪 Technical Details

### MediaPipe Hand Landmarks
21 keypoints per hand:
- Wrist (0)
- Thumb: 1-4
- Index: 5-8
- Middle: 9-12
- Ring: 13-16
- Pinky: 17-20

Each landmark has:
- x: normalized [0-1] horizontal position
- y: normalized [0-1] vertical position
- z: depth relative to wrist

### Feature Engineering
- **Input**: Raw image from webcam
- **Processing**: MediaPipe detection → 3D landmarks
- **Output**: 126D feature vector
- **Normalization**: Coordinates normalized to [0, 1]

---

## 📈 Evaluation Metrics

- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of classification performance

---

## 🎓 Course Topics Coverage

✅ **Pertemuan 1-2**: Image formation, camera model  
✅ **Pertemuan 3-4**: Image preprocessing, color spaces  
✅ **Pertemuan 5**: Histogram processing  
✅ **Pertemuan 6**: Thresholding techniques  
✅ **Pertemuan 7**: Segmentation  
✅ **Pertemuan 8-9**: Feature extraction  
✅ **Pertemuan 10-11**: Object detection & tracking  
✅ **Pertemuan 12**: 3D reconstruction  
✅ **Pertemuan 13**: Augmented reality  
✅ **Pertemuan 14**: Traditional machine learning  
✅ **Pertemuan 15-16**: Deep learning (CNN)  

---

## 🚧 Future Improvements

1. **Temporal Models**: LSTM/GRU for gesture sequences
2. **Transfer Learning**: Pre-trained models (ResNet, EfficientNet)
3. **Data Augmentation**: Rotation, scaling, noise injection
4. **Multi-modal**: Combine RGB + depth data
5. **Real-time Optimization**: Model quantization, pruning
6. **Mobile Deployment**: TensorFlow Lite conversion
7. **Sentence Recognition**: Word and sentence level detection

---

## 📝 Citation

If you use this project, please cite:

```
@project{bisindo-classifier-2025,
  author = {Krisna Santosa},
  title = {BISINDO Classifier: Indonesian Sign Language Recognition},
  year = {2025},
  institution = {Computer Vision Course - Final Project}
}
```

---

## 📧 Contact

**Krisna Santosa**  
LinkedIn: [krisna-santosa](https://www.linkedin.com/in/krisna-santosa/)  
GitHub: [Your GitHub Profile]

---

## 📄 License

This project is created for educational purposes as part of a Computer Vision course final project.

---

## 🙏 Acknowledgments

- **MediaPipe** by Google for hand landmark detection
- **OpenCV** for image processing
- **Streamlit** for interactive dashboard
- **TensorFlow/Keras** for deep learning
- **Scikit-learn** for machine learning utilities
- **Computer Vision Course** instructors and teaching assistants

---

## 🎉 Happy Coding!

Feel free to reach out for collaborations, questions, or suggestions!

**Last Updated**: December 31, 2025
