import os
import sys
from pathlib import Path

# DeepFace logs emoji while downloading its models. Windows consoles commonly
# default to cp1252, which cannot encode them and aborts the download.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# DeepFace accesses ``tensorflow.keras`` during import. Import it first so
# TensorFlow initializes its lazy Keras module correctly on Windows.
import tensorflow.keras  # noqa: F401

# Keep DeepFace model downloads in the project, where this app can write them.
os.environ.setdefault("DEEPFACE_HOME", str(Path(__file__).resolve().parent))

import streamlit as st
from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image

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
st.markdown('<p class="subtitle">Upload a photo to analyze age, gender, and emotion using deep learning.</p>', unsafe_allow_html=True)

EMOTION_EMOJIS = {
    "happy": "😄", "sad": "😢", "angry": "😠", "surprise": "😲",
    "fear": "😨", "disgust": "🤢", "neutral": "😐"
}


def age_group(age: int) -> str:
    """Convert DeepFace's estimated age into a more reliable age range."""
    if age <= 12:
        return "0–12"
    if age <= 19:
        return "13–19"
    if age <= 29:
        return "20–29"
    if age <= 44:
        return "30–44"
    if age <= 59:
        return "45–59"
    return "60+"

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    col1, col2 = st.columns([1, 1.3])

    with st.spinner("Analyzing face..."):
        try:
            analysis = DeepFace.analyze(
                img_path=img_array,
                actions=['age', 'gender', 'emotion'],
                enforce_detection=True,
                detector_backend="retinaface",
            )
            # DeepFace returns one dictionary for a single face and a list for
            # multiple faces. Keep the rendering code consistent in both cases.
            results = analysis if isinstance(analysis, list) else [analysis]
        except Exception as e:
            results = None
            message = str(e)
            if "face could not be detected" in message.lower():
                st.warning("No face was detected. Use a clear, front-facing photo with good lighting.")
            else:
                st.error("Analysis could not complete. Open Technical details below for the exact reason.")
            with st.expander("Technical details"):
                st.code(message)

    if results:
        # DeepFace returns a list, one entry per detected face
        frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        for face in results:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            cv2.rectangle(frame, (x, y), (x + w, y + h), (99, 102, 241), 3)

        result_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        with col1:
            st.image(result_img, use_container_width=True, caption="Detected face(s)")

        with col2:
            for i, face in enumerate(results):
                if len(results) > 1:
                    st.markdown(f"#### Face {i + 1}")

                estimated_age_group = age_group(face['age'])
                gender = face['dominant_gender']
                gender_conf = face['gender'][gender]
                emotion = face['dominant_emotion']
                emotion_conf = face['emotion'][emotion]
                emoji = EMOTION_EMOJIS.get(emotion, "🙂")

                m1, m2, m3 = st.columns(3)
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
                            <div class="metric-value">{gender}</div>
                            <div class="metric-label">{gender_conf:.0f}% confidence</div>
                        </div>
                    """, unsafe_allow_html=True)
                with m3:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Emotion</div>
                            <div class="metric-value">{emoji} {emotion.capitalize()}</div>
                        </div>
                    """, unsafe_allow_html=True)

                st.write("")
                st.markdown("**Emotion breakdown**")
                emotion_scores = dict(sorted(face['emotion'].items(), key=lambda x: x[1], reverse=True))
                for emo, score in emotion_scores.items():
                    st.write(f"{EMOTION_EMOJIS.get(emo, '')} {emo.capitalize()}")
                    st.progress(min(int(score), 100))

                st.divider()
