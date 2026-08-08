import gc
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

# DeepFace logs emoji while downloading its models. Windows consoles commonly
# default to cp1252, which cannot encode them and aborts the download.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# DeepFace accesses ``tensorflow.keras`` during import. Import it first so
# TensorFlow initializes its lazy Keras module correctly on Windows.
try:
    import tensorflow.keras  # noqa: F401
except Exception as exc:  # pragma: no cover - import-time fallback
    tensorflow = None
    TENSORFLOW_IMPORT_ERROR = exc
else:
    TENSORFLOW_IMPORT_ERROR = None

# Keep DeepFace model downloads in the project, where this app can write them.
os.environ.setdefault("DEEPFACE_HOME", str(Path(__file__).resolve().parent))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import streamlit as st

try:
    from deepface import DeepFace
except Exception as exc:  # pragma: no cover - import-time fallback
    DeepFace = None
    DEEPFACE_IMPORT_ERROR = exc
else:
    DEEPFACE_IMPORT_ERROR = None

# Use the project's OpenCV age model for more reliable age estimates on
# smaller and younger faces than the generic DeepFace age path.
FACE_PROTO = str(Path(__file__).resolve().parent / "opencv_face_detector.pbtxt")
FACE_MODEL = str(Path(__file__).resolve().parent / "opencv_face_detector_uint8.pb")
AGE_PROTO = str(Path(__file__).resolve().parent / "age_deploy.prototxt")
AGE_MODEL = str(Path(__file__).resolve().parent / "age_net.caffemodel")
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
AGE_LABELS = ['0–2', '4–6', '8–12', '15–20', '25–32', '38–43', '48–53', '60+']

face_net = cv2.dnn.readNetFromTensorflow(FACE_MODEL, FACE_PROTO)
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)

st.set_page_config(page_title="Face Insight AI", layout="wide", page_icon="🧠")

# ---------- Custom styling ----------
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #9ca3af;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #1f2937;
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #374151;
    }
    .metric-label {
        color: #9ca3af;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #f9fafb;
        font-size: 1.8rem;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🧠 Face Insight AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a photo to analyze age and gender with a lighter model setup.</p>', unsafe_allow_html=True)

if DeepFace is None:
    st.error("The analysis engine could not be loaded. Please try again shortly.")
    with st.expander("Technical details"):
        st.code(str(DEEPFACE_IMPORT_ERROR or TENSORFLOW_IMPORT_ERROR))
    st.stop()

EMOTION_EMOJIS = {
    "happy": "😄", "sad": "😢", "angry": "😠", "surprise": "😲",
    "fear": "😨", "disgust": "🤢", "neutral": "😐"
}


def age_group(age: int) -> str:
    """Map a model age bucket to a readable label."""
    return AGE_LABELS[min(max(age, 0), len(AGE_LABELS) - 1)]


def detect_faces(frame: np.ndarray, conf_threshold: float = 0.7):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    face_boxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            face_boxes.append([x1, y1, x2, y2])
    return face_boxes


def predict_age(face: np.ndarray) -> str:
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_idx = int(age_preds[0].argmax())
    return AGE_LABELS[age_idx]


def predict_gender(face: np.ndarray) -> tuple[str, str]:
    """Best-effort gender prediction from a cropped face image."""
    try:
        gender_result = DeepFace.analyze(
            img_path=face,
            actions=['gender'],
            enforce_detection=False,
            detector_backend='opencv',
            align=False,
            silent=True,
        )
        if isinstance(gender_result, list):
            gender_result = gender_result[0]
        gender = gender_result.get('dominant_gender', 'unknown')
        gender_conf = gender_result.get('gender', {}).get(gender, 0)
        label = gender.capitalize() if gender != 'unknown' else 'Unknown'
        return label, f"{float(gender_conf):.0f}%"
    except Exception:
        return 'Unknown', 'N/A'


def prepare_image(uploaded_file) -> np.ndarray:
    """Load and downscale an uploaded image to reduce memory pressure."""
    image = Image.open(uploaded_file)
    image = ImageOps.exif_transpose(image).convert("RGB")

    max_dim = 720
    if max(image.size) > max_dim:
        scale = max_dim / max(image.size)
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    return np.array(image)


uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.fromarray(prepare_image(uploaded_file))
    img_array = np.array(image)

    col1, col2 = st.columns([1, 1.3])

    with st.spinner("Analyzing face..."):
        try:
            frame_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            face_boxes = detect_faces(frame_bgr)
            if not face_boxes:
                st.warning("No face was detected. Use a clear, front-facing photo with good lighting.")
                st.stop()

            for x1, y1, x2, y2 in face_boxes:
                face_crop = frame_bgr[max(0, y1 - 20):min(y2 + 20, frame_bgr.shape[0] - 1),
                                      max(0, x1 - 20):min(x2 + 20, frame_bgr.shape[1] - 1)]
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (99, 102, 241), 3)

            result_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            with col1:
                st.image(result_img, use_container_width=True, caption="Detected face(s)")

            with col2:
                for i, (x1, y1, x2, y2) in enumerate(face_boxes):
                    if len(face_boxes) > 1:
                        st.markdown(f"#### Face {i + 1}")

                    face_crop = frame_bgr[max(0, y1 - 20):min(y2 + 20, frame_bgr.shape[0] - 1),
                                          max(0, x1 - 20):min(x2 + 20, frame_bgr.shape[1] - 1)]
                    estimated_age_group = predict_age(face_crop)
                    gender_label, gender_conf = predict_gender(face_crop)
                    m1, m2 = st.columns(2)
                    with m1:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Age group</div>
                                <div class="metric-value">{estimated_age_group}</div>
                            </div>
                        """, unsafe_allow_html=True)
                    with m2:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">Gender</div>
                                <div class="metric-value">{gender_label}</div>
                                <div class="metric-label">{gender_conf} confidence</div>
                            </div>
                        """, unsafe_allow_html=True)

                    st.divider()
        except Exception as e:
            message = str(e)
            st.error("Analysis could not complete. Open Technical details below for the exact reason.")
            with st.expander("Technical details"):
                st.code(message)

    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()
