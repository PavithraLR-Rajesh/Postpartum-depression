from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
import librosa
import os
import subprocess

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Ensure the uploads directory exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained models
mcq_model = load_model("models/mcq_model.h5")  # MCQ model
audio_model = load_model("models/audio_model.h5")  # Audio-based depression model

# Load dataset for normalization
DATA_CSV = "data/integrated_dataset.csv"
df = pd.read_csv(DATA_CSV)

# Define feature columns
mcq_features = [
    'Feeling sad or Tearful', 'Irritable towards baby & partner', 
    'Trouble sleeping at night', 'Problems concentrating or making decision',
    'Overeating or loss of appetite', 'Feeling anxious', 'Feeling of guilt',
    'Problems of bonding with baby', 'Suicide attempt'
]

audio_features = ['MFCC_1', 'MFCC_2', 'MFCC_3', 'Chroma_1', 'Spectral_Centroid', 'Zero_Crossing_Rate']

# Normalize MCQ & Audio data using dataset statistics
scaler_mcq = StandardScaler().fit(df[mcq_features])
scaler_audio = StandardScaler().fit(df[audio_features])

def extract_audio_features(file_path):
    """Extracts relevant audio features using Librosa"""
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)

        features = [
            np.mean(mfccs[0]), np.mean(mfccs[1]), np.mean(mfccs[2]),
            np.mean(chroma), np.mean(spectral_centroid), np.mean(zero_crossing_rate)
        ]
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
    print("📥 Received form data:", request.form)  # Debugging
    try:
        # Extract questionnaire responses
        data = {feature: float(request.form.get(feature, 0)) for feature in mcq_features}
        test_mcq_df = pd.DataFrame([data], columns=mcq_features)

        # Normalize the MCQ responses
        test_mcq_scaled = scaler_mcq.transform(test_mcq_df)

        # Calculate the scaled MCQ score (between 0 and 1)
        mcq_scaled_score = np.mean(test_mcq_scaled)  # Example of scaling into a single score

        # Print the scaled MCQ score (lies between 0 and 1)
        print(f"🔹 Scaled MCQ Score (0-1 Range): {mcq_scaled_score}")  # ✅ Added this line

        # Predict using the MCQ model
        mcq_pred = mcq_model.predict(test_mcq_scaled)[0][0]  # Use the trained model

        # Store the MCQ model's prediction in the session
        session['mcq_score'] = float(mcq_pred)  
        session.pop('audio_score', None)  # Clear previous audio score if exists

        print(f"✅ MCQ Prediction (Model-Based): {mcq_pred}")  # Debugging
        
        # Return JSON response with MCQ prediction
        return jsonify({
            "mcq_scaled_score": float(mcq_scaled_score),  # ✅ Return scaled score for debugging
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
            # Convert to WAV with explicit format settings
            subprocess.run(
                ["ffmpeg", "-i", webm_path, "-ac", "1", "-ar", "22050", "-f", "wav", wav_path, "-y"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            print("❌ Audio Conversion Error:", e)
            return jsonify({"message": "Error converting audio format."}), 500

        # Extract features from the audio file
        test_audio = extract_audio_features(wav_path)
        if test_audio is None:
            return jsonify({"message": "Audio feature extraction failed."}), 500

        # Normalize and predict using audio model
        test_audio_df = pd.DataFrame([test_audio], columns=audio_features)
        test_audio_scaled = scaler_audio.transform(test_audio_df)
        audio_pred = audio_model.predict(test_audio_scaled)[0][0]

        session['audio_score'] = float(audio_pred)  # Store in session

        print("✅ Audio Prediction:", audio_pred)  # Debugging

        # Return JSON response to the frontend
        return jsonify({"audio_score": float(audio_pred), "message": "Audio analysis successful."})  # ✅ Fixed float32 issue
    return render_template('audio_analysis.html')


@app.route('/result')
def result():
    mcq_pred = session.get('mcq_score')
    audio_pred = session.get('audio_score')

    if mcq_pred is None:
        return redirect(url_for('questionnaire'))  # Redirect if MCQ is missing

    if audio_pred is None:
        return redirect(url_for('audio_analysis'))  # Redirect if audio is missing

    mcq_pred = float(mcq_pred)
    audio_pred = float(audio_pred)

    # Convert MCQ score to categorical (Depressed or Not Depressed)
    mcq_result = "Depressed" if mcq_pred > 1 else "Not Depressed"
    
    # Convert audio score to categorical (Depressed or Not Depressed)
    audio_result = "Depressed" if audio_pred > 1 else "Not Depressed"

    # Final decision using AND logic
    final_pred = "Depressed" if (mcq_pred > 1 and audio_pred > 1 ) else "Not Depressed"

    return render_template('result.html', final_pred=final_pred, mcq_result=mcq_result, audio_result=audio_result)


if __name__ == '__main__':
    app.run(debug=True)
