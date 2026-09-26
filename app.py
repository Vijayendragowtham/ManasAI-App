import os
import tempfile

import librosa
import numpy as np
import streamlit as st
import tensorflow as tf

CLASSES = [
    "angry",
    "calm",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

MODEL_PATH = os.path.join("model", "emotion_model.keras")


@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    max_len = 174
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, max_len - mfcc.shape[1])),
            mode="constant",
        )
    else:
        mfcc = mfcc[:, :max_len]

    return np.expand_dims(np.expand_dims(mfcc, axis=-1), axis=0)


def emotion_analysis(emotion):
    responses = {
        "sad": (
            "Possible emotional distress",
            "Medium",
            "Try slow breathing for a few minutes and consider talking to someone you trust.",
        ),
        "angry": (
            "High emotional arousal",
            "Medium",
            "Pause, step away briefly, and take several slow breaths.",
        ),
        "fear": (
            "Possible anxiety or stress response",
            "High",
            "Focus on slow breathing and consider talking to someone you trust.",
        ),
        "happy": (
            "Positive emotional state",
            "Low",
            "Maintain activities and habits that support your well-being.",
        ),
        "neutral": (
            "Neutral emotional state",
            "Low",
            "No strong emotional signal was detected from the recording.",
        ),
        "calm": (
            "Relaxed emotional state",
            "Low",
            "Maintain your calm mindset.",
        ),
        "disgust": (
            "Strong negative emotional response",
            "Medium",
            "Take a short break and give yourself time to process the situation.",
        ),
        "surprise": (
            "Unexpected emotional response",
            "Low",
            "Take a moment to process the situation.",
        ),
    }
    return responses.get(
        emotion,
        ("Unknown", "Unknown", "Unable to generate an interpretation."),
    )


st.set_page_config(
    page_title="ManasAI",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 ManasAI")
st.subheader("AI-Powered Emotional & Mental Health Detection System")
st.write(
    "Upload a voice recording to analyze speech emotion using MFCC "
    "features and a TensorFlow/Keras emotion-classification model."
)

st.warning(
    "ManasAI is an awareness and emotion-analysis prototype, not a medical "
    "diagnosis or a substitute for professional mental-health care."
)

uploaded_file = st.file_uploader(
    "Upload an audio recording",
    type=["wav", "mp3", "ogg", "m4a"],
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("Analyze Emotion", type="primary"):
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=os.path.splitext(uploaded_file.name)[1] or ".wav",
            ) as tmp:
                tmp.write(uploaded_file.getbuffer())
                temp_path = tmp.name

            with st.spinner("Loading model and analyzing speech..."):
                model = load_emotion_model()
                features = extract_features(temp_path)
                prediction = model.predict(features, verbose=0)
                predicted_index = int(np.argmax(prediction[0]))
                emotion = CLASSES[predicted_index]
                confidence = float(np.max(prediction[0]))

            mental_state, risk_level, suggestion = emotion_analysis(emotion)

            st.success(f"Detected Emotion: {emotion.upper()}")
            st.metric("Model Confidence", f"{confidence * 100:.2f}%")
            st.write(f"**Mental State:** {mental_state}")
            st.write(f"**Risk Level:** {risk_level}")
            st.write(f"**Suggestion:** {suggestion}")

        except Exception as exc:
            st.error("The audio could not be analyzed.")
            st.exception(exc)

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
