import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import librosa
import tensorflow as tf
import joblib

# Load model and label encoder
model = tf.keras.models.load_model("emotion_model.keras")
label_encoder = joblib.load("label_encoder.pkl")

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, duration=3, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc = np.pad(mfcc, ((0,0),(0,174 - mfcc.shape[1])), mode='constant')
    return mfcc.reshape(1,40,174,1)

def predict_emotion(file_path):
    features = extract_features(file_path)
    prediction = model.predict(features)
    predicted_index = np.argmax(prediction)
    emotion = label_encoder.inverse_transform([predicted_index])
    return emotion[0]

# Test with sample file
file_path = r"C:\Users\vijay\Desktop\project\RAVDESS\Actor_01\03-01-01-01-01-01-01.wav"

emotion = predict_emotion(file_path)
print("Predicted Emotion:", emotion)