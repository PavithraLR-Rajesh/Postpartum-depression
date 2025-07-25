import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
import librosa
import os

# Load trained models
mcq_model = load_model("D:/Postpartum_Depression_Detection/models/mcq_model.h5")
audio_model = load_model("D:/Postpartum_Depression_Detection/models/audio_model.h5")
fusion_model = load_model("D:/Postpartum_Depression_Detection/models/fusion_model.h5")

# Load dataset for feature normalization
DATA_CSV = "D:/Postpartum_Depression_Detection/data/integrated_dataset.csv"
df = pd.read_csv(DATA_CSV)

# Define Features
mcq_features = ['Feeling sad or Tearful', 'Irritable towards baby & partner', 
                'Trouble sleeping at night', 'Problems concentrating or making decision',
                'Overeating or loss of appetite', 'Feeling anxious', 'Feeling of guilt',
                'Problems of bonding with baby', 'Suicide attempt']

audio_features = ['MFCC_1', 'MFCC_2', 'MFCC_3', 'Chroma_1', 'Spectral_Centroid', 'Zero_Crossing_Rate']

# Normalize with training data
scaler_mcq = StandardScaler().fit(df[mcq_features])
scaler_audio = StandardScaler().fit(df[audio_features])

# Function to classify severity
def classify_severity(prob):
    if prob < 0.25:
        return "No Depression"
    elif prob < 0.5:
        return "Mild Depression"
    elif prob < 0.75:
        return "Moderate Depression"
    else:
        return "Severe Depression"

# Get user choice for testing
print("\nChoose testing mode:")
print("1️⃣ MCQ Only")
print("2️⃣ Audio Only")
print("3️⃣ Both MCQ & Audio")
mode = input("Enter choice (1/2/3): ")

# Get MCQ responses if required
test_mcq_scaled = None
if mode in ['1', '3']:
    print("\n📋 Enter MCQ Responses (0 for No, 1 for Yes)")
    test_mcq = []
    for feature in mcq_features:
        while True:
            try:
                value = int(input(f"{feature}: "))
                if value in [0, 1]:
                    test_mcq.append(value)
                    break
                else:
                    print("⚠️ Please enter only 0 or 1.")
            except ValueError:
                print("⚠️ Invalid input! Enter 0 or 1.")
    test_mcq_scaled = scaler_mcq.transform([test_mcq]).reshape(1, len(mcq_features), 1)

# Get Audio file if required
test_audio_scaled = None
if mode in ['2', '3']:
    wav_file = "D:/Postpartum_Depression_Detection/uploads/test_audio.wav"
    if not os.path.exists(wav_file):
        print(f"❌ Error: File not found at {wav_file}. Exiting.")
        exit()
    
    def extract_audio_features(file_path):
        try:
            y, sr = librosa.load(file_path, sr=22050)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            return [
                np.mean(mfccs[0]), np.mean(mfccs[1]), np.mean(mfccs[2]),
                np.mean(chroma), np.mean(spectral_centroid), np.mean(zero_crossing_rate)
            ]
        except Exception as e:
            print(f"❌ Error extracting features from {file_path}: {e}")
            return None
    
    test_audio = extract_audio_features(wav_file)
    if test_audio is None:
        print("❌ Failed to extract audio features. Exiting.")
        exit()
    test_audio_scaled = scaler_audio.transform([test_audio]).reshape(1, len(audio_features), 1)

# Get predictions based on selected mode
if mode == '1':  # MCQ Only
    mcq_pred = mcq_model.predict(test_mcq_scaled)[0][0]
    print(f"\n📝 MCQ Model Output: {mcq_pred:.4f} → {classify_severity(mcq_pred)}")

elif mode == '2':  # Audio Only
    audio_pred = audio_model.predict(test_audio_scaled)[0][0]
    print(f"\n🎵 Audio Model Output: {audio_pred:.4f} → {classify_severity(audio_pred)}")

elif mode == '3':  # Both MCQ and Audio
    fusion_pred = fusion_model.predict([test_mcq_scaled, test_audio_scaled])[0][0]
    print(f"\n🔄 Fusion Model (Final Prediction Score): {fusion_pred:.4f} → {classify_severity(fusion_pred)}")
else:
    print("⚠️ Invalid choice. Exiting.")
