import os
import subprocess
import librosa
import numpy as np
import soundfile as sf

# Define FFmpeg executable path (modify this if needed)
FFMPEG_PATH = r"C:\ffmpeg\ffmpeg.exe"  # Ensure this path is correct

# Define input and output file paths
INPUT_AUDIO = r"D:\Postpartum_Depression_Detection\uploads\recorded_audio.wav"
OUTPUT_WAV = r"D:\Postpartum_Depression_Detection\uploads\converted_audio.wav"

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
    """Extracts audio features from WAV and prints them"""
    try:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return None

        # Load the converted WAV file
        y, sr = librosa.load(file_path, sr=16000)

        if len(y) == 0:
            print("❌ Empty audio file. Extraction failed.")
            return None

        # Compute features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        zero_crossing = librosa.feature.zero_crossing_rate(y)

        # Compute mean values
        extracted_features = {
            "Sample Rate": sr,
            "MFCCs": [round(np.mean(mfcc), 4) for mfcc in mfccs],
            "Chroma": [round(np.mean(c), 4) for c in chroma],
            "Spectral Centroid": round(np.mean(spec_centroid), 4),
            "Zero Crossing Rate": round(np.mean(zero_crossing), 4),
        }

        print("\n✅ Feature Extraction Successful! Extracted Values:")
        for key, value in extracted_features.items():
            print(f"{key}: {value}")

        return extracted_features

    except Exception as e:
        print(f"❌ Error extracting features:\n{e}")
        return None


# Run conversion and extraction
if convert_to_wav(INPUT_AUDIO, OUTPUT_WAV):
    extracted_data = extract_audio_features(OUTPUT_WAV)
    if extracted_data:
        print("\n🎯 Final Extracted Features:", extracted_data)
else:
    print(f"❌ Conversion failed! File not created: {OUTPUT_WAV}")
