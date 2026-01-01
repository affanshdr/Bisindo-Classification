import os
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import platform
import warnings
from collections import deque
import time
import tempfile
from PIL import Image

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def is_cloud_environment():
    """Check if running in Streamlit Cloud"""
    return os.getenv('STREAMLIT_RUNTIME_ENVIRONMENT') == 'cloud'

def apply_histogram_equalization(image):
    """Apply histogram equalization for better contrast"""
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

def enhance_low_light(image):
    """Enhance image in low light conditions"""
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    # Slight brightness boost
    enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=20)
    
    return enhanced

def apply_adaptive_threshold(image):
    """Apply adaptive thresholding"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

def segment_hand(image, hand_landmarks):
    """Segment hand region using convex hull"""
    if hand_landmarks is None:
        return image

    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for hand_landmark in hand_landmarks:
        points = []
        for lm in hand_landmark.landmark:
            h, w, _ = image.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            points.append([cx, cy])

        points = np.array(points, dtype=np.int32)
        hull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, hull, 255)

    segmented = cv2.bitwise_and(image, image, mask=mask)
    return segmented

def draw_3d_projection(image, hand_landmarks):
    """Draw 3D projection overlay"""
    if hand_landmarks is None:
        return image

    for hand_landmark in hand_landmarks:
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            start = hand_landmark.landmark[start_idx]
            end = hand_landmark.landmark[end_idx]

            h, w, _ = image.shape
            start_point = (int(start.x * w), int(start.y * h))
            end_point = (int(end.x * w), int(end.y * h))

            depth = abs(start.z + end.z) / 2
            color_intensity = int(255 * (1 - min(depth * 2, 1)))
            color = (color_intensity, 255 - color_intensity, 128)

            cv2.line(image, start_point, end_point, color, 2)

    return image

def draw_ar_overlay(image, prediction, confidence, hand_landmarks):
    """Draw AR overlay with enhanced visualization"""
    h, w, _ = image.shape
    overlay = image.copy()

    if hand_landmarks:
        for hand_landmark in hand_landmarks:
            points = []
            for lm in hand_landmark.landmark:
                points.append([int(lm.x * w), int(lm.y * h)])
            points = np.array(points)

            x, y, box_w, box_h = cv2.boundingRect(points)

            thickness = 3
            length = 30
            cv2.line(overlay, (x, y), (x + length, y), (0, 255, 0), thickness)
            cv2.line(overlay, (x, y), (x, y + length), (0, 255, 0), thickness)
            cv2.line(overlay, (x + box_w, y), (x + box_w - length, y), (0, 255, 0), thickness)
            cv2.line(overlay, (x + box_w, y), (x + box_w, y + length), (0, 255, 0), thickness)
            cv2.line(overlay, (x, y + box_h), (x + length, y + box_h), (0, 255, 0), thickness)
            cv2.line(overlay, (x, y + box_h), (x, y + box_h - length), (0, 255, 0), thickness)
            cv2.line(overlay, (x + box_w, y + box_h), (x + box_w - length, y + box_h), (0, 255, 0), thickness)
            cv2.line(overlay, (x + box_w, y + box_h), (x + box_w, y + box_h - length), (0, 255, 0), thickness)

            panel_height = 80
            cv2.rectangle(overlay, (x, y - panel_height), (x + box_w, y), (0, 255, 0), -1)

            cv2.putText(overlay, f'Sign: {prediction}', (x + 10, y - 45),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(overlay, f'Confidence: {confidence:.2f}%', (x + 10, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image

def extract_landmarks(image, min_detection_confidence=0.5, min_tracking_confidence=0.5):
    """Extract hand landmarks with static mode for images"""
    # Use static_image_mode=True for better detection on static images
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2,
                       min_detection_confidence=min_detection_confidence,
                       min_tracking_confidence=min_tracking_confidence) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])

            if len(results.multi_hand_landmarks) == 1:
                landmarks.extend([0] * 63)

            return landmarks, results.multi_hand_landmarks

        return None, None

def process_frame(frame, clf, settings):
    """Process a single frame with all settings applied"""
    original_frame = frame.copy()
    
    # Apply low light enhancement first if enabled
    if settings['show_low_light_enhance']:
        frame = enhance_low_light(frame)
    
    # Apply basic adjustments
    frame = cv2.convertScaleAbs(frame, alpha=1 + settings['contrast']/100, 
                                beta=settings['brightness'])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = cv2.add(hsv[..., 1], settings['saturation'])
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Apply histogram equalization if enabled
    if settings['show_histogram_eq']:
        frame = apply_histogram_equalization(frame)

    # Extract landmarks and make prediction
    landmarks, hand_landmarks = extract_landmarks(
        frame,
        min_detection_confidence=settings['min_detection_confidence'],
        min_tracking_confidence=settings['min_tracking_confidence']
    )

    applied_techniques = []
    predicted_label = "No hand detected"
    confidence = 0.0

    if landmarks:
        landmarks_np = np.array(landmarks).reshape(1, -1)
        prediction = clf.predict(landmarks_np)
        predicted_label = prediction[0]

        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(landmarks_np)
            confidence = np.max(proba) * 100
        else:
            confidence = 100.0

        if settings['show_segmentation']:
            frame = segment_hand(frame, hand_landmarks)
            applied_techniques.append(" Hand Segmentation, ")

        if settings['show_3d_projection']:
            frame = draw_3d_projection(frame, hand_landmarks)
            applied_techniques.append(" 3D Projection, ")
        else:
            for hand_landmark in hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmark, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

        if settings['show_ar_overlay']:
            frame = draw_ar_overlay(frame, predicted_label, confidence, hand_landmarks)
            applied_techniques.append(" AR Overlay, ")
        else:
            cv2.putText(frame, f'Sign: {predicted_label}', (10, 40),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        applied_techniques.append(" Feature Extraction, ")
        applied_techniques.append(" Object Tracking, ")

    if settings['show_threshold']:
        frame = apply_adaptive_threshold(frame)
        applied_techniques.append(" Adaptive Thresholding, ")

    if settings['show_low_light_enhance']:
        applied_techniques.append(" Low Light Enhancement, ")

    if settings['show_histogram_eq']:
        applied_techniques.append(" Histogram Equalization, ")

    return frame, predicted_label, confidence, applied_techniques

def initialize_camera():
    """Initialize camera with error handling"""
    try:
        if is_cloud_environment():
            st.warning(" Running in cloud environment. Camera access might be limited.")

        for index in [0, 1]:
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                return cap
            cap.release()

        st.error("🎥 Camera not available.")
        return None

    except Exception as e:
        st.error(f" Error initializing camera: {str(e)}")
        return None

def main():
    st.set_page_config(page_title="BISINDO Classifier", layout="wide")

    st.title('Advanced BISINDO Classification System')
    st.markdown('*Real-time Indonesian Sign Language Recognition with Computer Vision Techniques*')

    # Initialize session state
    if 'camera' not in st.session_state:
        st.session_state.camera = None
        st.session_state.camera_initialized = False
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = deque(maxlen=30)

    # Sidebar configuration
    st.sidebar.header(' Input Mode')
    input_mode = st.sidebar.radio(
        'Select Input Source',
        ['Webcam', 'Upload Image']
    )

    st.sidebar.header('Model Selection')
    model = st.sidebar.selectbox('Select Model', ['RF_BISINDO_99'], disabled=True)

    st.sidebar.header('🎥 Image Settings')
    brightness = st.sidebar.slider('Brightness', -100, 100, 0)
    contrast = st.sidebar.slider('Contrast', -100, 100, 0)
    saturation = st.sidebar.slider('Saturation', -100, 100, 0)

    st.sidebar.header('Model Settings')
    min_detection_confidence = st.sidebar.slider('Min Detection Confidence', 0.0, 1.0, 0.5)
    min_tracking_confidence = st.sidebar.slider('Min Tracking Confidence', 0.0, 1.0, 0.5)

    st.sidebar.header('Visualization Modes')
    show_low_light_enhance = st.sidebar.checkbox('Low Light Enhancement', value=False)
    show_histogram_eq = st.sidebar.checkbox('Histogram Equalization', value=False)
    show_threshold = st.sidebar.checkbox('Adaptive Thresholding', value=False)
    show_segmentation = st.sidebar.checkbox('Hand Segmentation', value=False)
    show_3d_projection = st.sidebar.checkbox('3D Projection', value=True)
    show_ar_overlay = st.sidebar.checkbox('AR Overlay', value=True)

    # Create settings dictionary
    settings = {
        'brightness': brightness,
        'contrast': contrast,
        'saturation': saturation,
        'min_detection_confidence': min_detection_confidence,
        'min_tracking_confidence': min_tracking_confidence,
        'show_low_light_enhance': show_low_light_enhance,
        'show_histogram_eq': show_histogram_eq,
        'show_threshold': show_threshold,
        'show_segmentation': show_segmentation,
        'show_3d_projection': show_3d_projection,
        'show_ar_overlay': show_ar_overlay
    }

    # Load model
    try:
        model_path = f'model/{model.lower()}.pkl'
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return
        clf = joblib.load(model_path)
        st.sidebar.success(f" Model loaded: {model}")
    except Exception as e:
        st.error(f" Error loading model: {str(e)}")
        return

    # Main interface based on input mode
    if input_mode == 'Webcam':
        handle_webcam_mode(clf, settings)
    elif input_mode == 'Upload Image':
        handle_image_upload_mode(clf, settings)

    # Footer
    with st.container():
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 4, 1])

        with col2:
            st.markdown("### 👨‍💻 Project Information")
            st.info("""
            **Created by: Affan Suhendar**

            **Computer Vision Techniques Implemented:**
            - Camera Model & Calibration
            - Image Formation & Preprocessing
            - Histogram Equalization
            - Adaptive Thresholding
            - Image Segmentation (Hand Segmentation)
            - Feature Extraction (MediaPipe Landmarks)
            - Object Detection & Tracking
            - Object Recognition (Random Forest Classifier)
            - 3D Projection & Visualization
            - Augmented Reality Overlay

            **Input Modes:**
            - Real-time Webcam
            - Image Upload (JPG, PNG, JPEG)

            This is an enhanced BISINDO (Indonesian Sign Language) Classification system
            for Computer Vision Final Project.
            """)

        st.markdown("---")

def handle_webcam_mode(clf, settings):
    """Handle webcam input mode"""
    col1, col2 = st.columns([2, 1])

    with col1:
        run = st.checkbox('▶️ Start Webcam', value=False)
        FRAME_WINDOW = st.empty()

    with col2:
        st.subheader(' Prediction Info')
        prediction_text = st.empty()
        confidence_text = st.empty()
        fps_text = st.empty()

        st.subheader(' Applied Techniques')
        techniques_display = st.empty()

    if run:
        if not st.session_state.camera_initialized:
            st.session_state.camera = initialize_camera()
            st.session_state.camera_initialized = True

        if st.session_state.camera is not None:
            prev_time = time.time()

            while run:
                try:
                    ret, frame = st.session_state.camera.read()
                    if not ret:
                        st.error("Failed to get frame from camera")
                        break

                    frame = cv2.flip(frame, 1)
                    
                    processed_frame, predicted_label, confidence, applied_techniques = process_frame(
                        frame, clf, settings
                    )

                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time)
                    prev_time = curr_time

                    FRAME_WINDOW.image(processed_frame, channels='BGR', use_container_width=True)

                    prediction_text.markdown(f"###  **{predicted_label}**")
                    confidence_text.markdown(f"**Confidence:** {confidence:.2f}%")
                    fps_text.markdown(f"**FPS:** {fps:.2f}")

                    techniques_display.markdown("\n".join(applied_techniques) if applied_techniques else "No techniques applied")

                except Exception as e:
                    st.error(f"Error processing frame: {str(e)}")
                    break
    else:
        if st.session_state.camera is not None:
            st.session_state.camera.release()
            st.session_state.camera = None
            st.session_state.camera_initialized = False

def handle_image_upload_mode(clf, settings):
    """Handle image upload mode"""
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
        FRAME_WINDOW = st.empty()

    with col2:
        st.subheader('Prediction Info')
        prediction_text = st.empty()
        confidence_text = st.empty()

        st.subheader('Applied Techniques')
        techniques_display = st.empty()

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        processed_frame, predicted_label, confidence, applied_techniques = process_frame(
            frame, clf, settings
        )

        FRAME_WINDOW.image(processed_frame, channels='BGR', use_container_width=True)

        prediction_text.markdown(f"###  **{predicted_label}**")
        confidence_text.markdown(f"**Confidence:** {confidence:.2f}%")

        techniques_display.markdown("\n".join(applied_techniques) if applied_techniques else "No techniques applied")

if __name__ == "__main__":
    main()