import os
import librosa
import numpy as np
import pandas as pd

# Directory containing WAV files
AUDIO_DIR = r"D:\miniproject\archive\New folder"

# Function to extract features
def extract_features(file_path):
    try:
        # Validate if the file is a valid WAV file
        if not file_path.lower().endswith('.wav'):
            print(f"Skipping non-wav file: {file_path}")
            return None

        # Load audio file using librosa (ensures correct sample rate handling)
        audio_array, sample_rate = librosa.load(file_path, sr=None)

        # Check for empty audio
        if len(audio_array) == 0:
            print(f"Skipping empty file: {file_path}")
            return None

        # Normalize audio
        audio_array /= np.max(np.abs(audio_array))

        # Ensure audio length is sufficient for feature extraction
        if len(audio_array) < 2048:
            print(f"Skipping short file: {file_path}")
            return None

        # Compute features
        mfccs = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio_array, sr=sample_rate)
        spec_centroid = librosa.feature.spectral_centroid(y=audio_array, sr=sample_rate)
        zero_crossings = librosa.feature.zero_crossing_rate(audio_array)

        # Compute mean values
        mfccs_mean = np.mean(mfccs, axis=1).tolist() if mfccs is not None else [None] * 13
        chroma_mean = np.mean(chroma, axis=1).tolist() if chroma is not None else [None] * 12
        spec_centroid_mean = np.mean(spec_centroid).item() if spec_centroid is not None else None
        zero_crossings_mean = np.mean(zero_crossings).item() if zero_crossings is not None else None

        return [sample_rate] + mfccs_mean + chroma_mean + [spec_centroid_mean, zero_crossings_mean]

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Process all WAV files in the directory
data = []
for file in os.listdir(AUDIO_DIR):
    file_path = os.path.join(AUDIO_DIR, file)
    features = extract_features(file_path)
    if features:
        data.append([file] + features)

# Define column names
columns = ["Filename", "Sample Rate"] + [f"MFCC_{i}" for i in range(1, 14)] + \
          [f"Chroma_{i}" for i in range(1, 13)] + ["Spectral_Centroid", "Zero_Crossing_Rate"]

# Convert to DataFrame
df = pd.DataFrame(data, columns=columns)

# Save to CSV
output_csv = r"D:\miniproject\audio_features.csv"
df.to_csv(output_csv, index=False)

print(f"✅ Feature extraction complete! Data saved to: {output_csv}")

