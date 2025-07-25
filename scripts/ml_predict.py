import os
import subprocess
import numpy as np
import tensorflow as tf
import librosa

# Set paths
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_model.h5')
FFMPEG_PATH = r"C:\ffmpeg\ffmpeg.exe"  # Ensure this path is correct

def load_model():
    """Load trained model."""
    try:
        print(f"📢 Loading model from: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

# Load the model once during script initialization
model = load_model()

def convert_to_wav(input_path, output_path):
    """Convert any audio file to WAV format using FFmpeg."""
    try:
        command = [
            FFMPEG_PATH, "-y", "-i", input_path, "-acodec", "pcm_s16le", "-ar", "16000", output_path
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode == 0:
            print("✅ Audio converted to WAV successfully!")
            return True
        else:
            print(f"❌ FFmpeg Error: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return False

def extract_audio_features(file_path):
    """Extract audio features using librosa."""
    try:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return [0] * 17  # Default feature vector

        # Load the WAV file
        y, sr = librosa.load(file_path, sr=16000)
        if len(y) == 0:
            print("❌ Empty audio file. Extraction failed.")
            return [0] * 17

        # Compute audio features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(y)

        # Compute mean values for features
        extracted_features = (
            [round(np.mean(mfcc), 4) for mfcc in mfccs] +  # 13 MFCCs
            [round(np.mean(chroma), 4)] +  # 1 Chroma
            [round(np.mean(spec_centroid), 4)] +  # 1 Spectral Centroid
            [round(np.mean(zero_crossing), 4)]  # 1 Zero Crossing Rate
        )

        print("✅ Feature Extraction Successful! Extracted Features:", extracted_features)
        return extracted_features

    except Exception as e:
        print(f"❌ Error extracting features:\n{e}")
        return [0] * 17  # Default vector in case of an error

def predict_depression(features):
    """Predict depression using the extracted features."""
    try:
        print(f"📢 Received features for prediction: {features}")
        if model is None:
            print("❌ Model is not loaded")
            return "Error: Model not loaded"

        features_array = np.array(features).reshape(1, -1)
        print(f"📊 Reshaped features: {features_array.shape}")

        prediction = model.predict(features_array)
        print(f"🎯 Raw Prediction: {prediction}")

        result = "Depressed" if prediction[0][0] > 0.5 else "Not Depressed"
        print(f"✅ Final Prediction: {result}")
        return result
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return "Error: Prediction failed"


