import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import tempfile

# -----------------------------
# Load Model
# -----------------------------
model = tf.keras.models.load_model("model/emotion_model.keras")

classes = [
    'angry',
    'calm',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise'
]

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(file_path):

    y, sr = librosa.load(file_path, sr=None)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=40
    )

    max_len = 174

    if mfcc.shape[1] < max_len:

        pad_width = max_len - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, pad_width)),
            mode='constant'
        )

    else:
        mfcc = mfcc[:, :max_len]

    mfcc = np.expand_dims(mfcc, axis=-1)
    mfcc = np.expand_dims(mfcc, axis=0)

    return mfcc


# -----------------------------
# Emotion Analysis
# -----------------------------
def emotion_analysis(emotion):

    responses = {

        "sad": {
            "mental_state": "Possible emotional distress",
            "risk_level": "Medium",
            "suggestion": "Try deep breathing for 2 minutes."
        },

        "angry": {
            "mental_state": "High emotional arousal",
            "risk_level": "Medium",
            "suggestion": "Pause and take 5 slow breaths."
        },

        "fear": {
            "mental_state": "Anxiety or stress detected",
            "risk_level": "High",
            "suggestion": "Focus on slow breathing and talk to someone you trust."
        },

        "happy": {
            "mental_state": "Positive emotional state",
            "risk_level": "Low",
            "suggestion": "Great! Keep doing what makes you happy."
        },

        "neutral": {
            "mental_state": "Emotionally stable",
            "risk_level": "Low",
            "suggestion": "Everything seems balanced."
        },

        "calm": {
            "mental_state": "Relaxed state",
            "risk_level": "Low",
            "suggestion": "Maintain your calm mindset."
        },

        "disgust": {
            "mental_state": "Strong negative reaction",
            "risk_level": "Medium",
            "suggestion": "Take a short break from the situation."
        },

        "surprise": {
            "mental_state": "Unexpected emotional shift",
            "risk_level": "Low",
            "suggestion": "Take a moment to process what happened."
        }
    }

    return responses.get(emotion)


# -----------------------------
# Streamlit Page Config
# -----------------------------
st.set_page_config(
    page_title="ManasAI",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# UI
# -----------------------------
st.title("🧠 ManasAI")

st.subheader(
    "AI-Powered Emotional & Mental Health Detection System"
)

st.write(
    "Upload an audio file to analyze emotions and mental state."
)

uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "mp3"]
)

# -----------------------------
# Prediction Section
# -----------------------------
if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:

        tmp_file.write(uploaded_file.read())

        temp_path = tmp_file.name

    st.info("Analyzing Emotion...")

    features = extract_features(temp_path)

    prediction = model.predict(features)

    predicted_index = np.argmax(prediction)

    emotion = classes[predicted_index]

    confidence = float(np.max(prediction))

    analysis = emotion_analysis(emotion)

    st.success(f"Detected Emotion: {emotion.upper()}")

    st.write(f"Confidence Score: {confidence:.2f}")

    st.write(f"Mental State: {analysis['mental_state']}")

    st.write(f"Risk Level: {analysis['risk_level']}")

    st.write(f"Suggestion: {analysis['suggestion']}")