from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
import librosa
import soundfile as sf
import os
import subprocess

app = Flask(__name__)
app.secret_key = "supersecretkey"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained models
mcq_model = load_model("models/mcq_model.h5")
audio_model = load_model("models/audio_model.h5")
fusion_model = load_model("models/fusion_model.h5")  # Fusion model

# Load dataset for normalization
df = pd.read_csv("data/integrated_dataset.csv")

# Feature columns
mcq_features = [
    'Feeling sad or Tearful', 'Irritable towards baby & partner',
    'Trouble sleeping at night', 'Problems concentrating or making decision',
    'Overeating or loss of appetite', 'Feeling anxious', 'Feeling of guilt',
    'Problems of bonding with baby', 'Suicide attempt'
]

audio_features = ['MFCC_1', 'MFCC_2', 'MFCC_3', 'Chroma_1', 'Spectral_Centroid', 'Zero_Crossing_Rate']

# Scalers
scaler_mcq = StandardScaler().fit(df[mcq_features])
scaler_audio = StandardScaler().fit(df[audio_features])

def extract_audio_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"🎧 Audio Duration: {duration:.2f} seconds")

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

        features = [
            np.mean(mfccs[0]), np.mean(mfccs[1]), np.mean(mfccs[2]),
            np.mean(chroma), np.mean(spectral_centroid), np.mean(zero_crossing_rate)
        ]
        print("🧪 Extracted Audio Features:", features)
        return features
    except Exception as e:
        print("❌ Audio Feature Extraction Error:", e)
        return None

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/questionnaire', methods=['GET', 'POST'])
def questionnaire():
    if request.method == 'POST':
        return process_questionnaire()
    return render_template('questionnaire.html')

@app.route('/process_questionnaire', methods=['POST'])
def process_questionnaire():
    print("📥 Received form data:", request.form)
    try:
        # Extract questionnaire responses
        data = {feature: float(request.form.get(feature, 0)) for feature in mcq_features}
        test_mcq_df = pd.DataFrame([data], columns=mcq_features)

        # ✅ Save raw MCQ values to session for fusion use
        session['mcq_raw'] = list(test_mcq_df.iloc[0].values)

        # Normalize the MCQ responses
        test_mcq_scaled = scaler_mcq.transform(test_mcq_df)

        # Scaled MCQ score (for display/debug)
        mcq_scaled_score = np.mean(test_mcq_scaled)

        # Predict using the MCQ model
        mcq_pred = mcq_model.predict(test_mcq_scaled)[0][0]

        # Store model prediction
        session['mcq_score'] = float(mcq_pred)
        session.pop('audio_score', None)

        print(f"✅ MCQ Prediction (Model-Based): {mcq_pred}")

        return jsonify({
            "mcq_scaled_score": float(mcq_scaled_score),
            "mcq_model_pred": float(mcq_pred),
            "message": "MCQ processed successfully."
        })

    except Exception as e:
        print("❌ Error processing questionnaire:", e)
        return jsonify({"error": "Failed to process questionnaire."}), 500


@app.route('/audio_analysis', methods=['GET', 'POST'])
def audio_analysis():
    if request.method == 'POST':
        if 'audio' not in request.files:
            return jsonify({"message": "No file uploaded."}), 400

        file = request.files['audio']
        webm_path = os.path.join(UPLOAD_FOLDER, "audio.webm")
        wav_path = os.path.join(UPLOAD_FOLDER, "audio.wav")
        file.save(webm_path)

        try:
            subprocess.run(
                ["ffmpeg", "-i", webm_path, "-ac", "1", "-ar", "22050", "-f", "wav", wav_path, "-y"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            print("❌ Audio Conversion Error:", e)
            return jsonify({"message": "Error converting audio format."}), 500

        # Extract features
        test_audio = extract_audio_features(wav_path)
        if test_audio is None:
            return jsonify({"message": "Audio feature extraction failed."}), 500

        # Create dataframe
        test_audio_df = pd.DataFrame([test_audio], columns=audio_features)

        # ✅ Save raw audio values to session for fusion use
        session['audio_raw'] = list(test_audio_df.iloc[0].values)

        # Normalize and predict
        test_audio_scaled = scaler_audio.transform(test_audio_df)
        audio_pred = audio_model.predict(test_audio_scaled)[0][0]

        session['audio_score'] = float(audio_pred)

        print("✅ Audio Prediction:", audio_pred)

        return jsonify({"audio_score": float(audio_pred), "message": "Audio analysis successful."})
    
    return render_template('audio_analysis.html')


# MODIFY /result ROUTE
@app.route('/result')
def result():
    mcq_pred = session.get('mcq_score')
    audio_pred = session.get('audio_score')

    if mcq_pred is None:
        return redirect(url_for('questionnaire'))

    if audio_pred is None:
        return redirect(url_for('audio_analysis'))

    if 'mcq_raw' not in session or 'audio_raw' not in session:
        return "⚠️ Raw features missing for fusion prediction.", 500

    mcq_raw = np.array(session['mcq_raw'])
    audio_raw = np.array(session['audio_raw'])

    # Normalize and reshape for fusion model
    mcq_scaled = scaler_mcq.transform([mcq_raw]).reshape(1, -1, 1)
    audio_scaled = scaler_audio.transform([audio_raw]).reshape(1, -1, 1)

    # Predict with fusion model
    prob = fusion_model.predict([mcq_scaled, audio_scaled])[0][0]

    # Final fusion diagnosis
    final_pred = "Depressed" if prob >= 0.5 else "Not Depressed"

    # Interpret individual predictions
    mcq_result = "Depressed" if mcq_pred >= 0.5 else "Not Depressed"
    audio_result = "Depressed" if audio_pred <=1.2 else "Not Depressed"

    return render_template(
        'result.html',
        final_pred=final_pred,
        mcq_score=f"{mcq_pred:.3f}",
        audio_score=f"{audio_pred:.3f}",
        fusion_score=f"{prob:.3f}",
        mcq_result=mcq_result,
        audio_result=audio_result
    )

if __name__ == '__main__':
    app.run(debug=True)
