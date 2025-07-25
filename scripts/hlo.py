import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the audio classification model (sigmoid output)
model_path = "models/audio_classification_model.h5"
audio_model = load_model(model_path)
print("✅ Model loaded successfully!")

# Load dataset (same one used for training)
df = pd.read_csv("data/integrated_dataset.csv")

# Audio feature columns used during training
audio_features = ['MFCC_1', 'MFCC_2', 'MFCC_3', 'Chroma_1', 'Spectral_Centroid', 'Zero_Crossing_Rate']

# Check that required columns exist
if not all(col in df.columns for col in audio_features + ['label']):
    raise ValueError("❌ Dataset missing required columns!")

# Preprocess
X_audio = df[audio_features]
y_true = df['label']

# Normalize using StandardScaler
scaler = StandardScaler()
X_audio_scaled = scaler.fit_transform(X_audio)

# Predict using the model
y_pred_probs = audio_model.predict(X_audio_scaled)

# Convert sigmoid output to binary prediction
y_pred_classes = (y_pred_probs >= 0.5).astype(int).flatten()

# Print a few sample predictions
for i in range(5):
    print(f"Sample {i+1}: Prob={y_pred_probs[i][0]:.4f}, Pred={y_pred_classes[i]}, True={y_true.iloc[i]}")

# Optional: Compute basic metrics
from sklearn.metrics import accuracy_score, classification_report

print("\n🔍 Classification Report:")
print(classification_report(y_true, y_pred_classes, digits=4))

print(f"🎯 Accuracy: {accuracy_score(y_true, y_pred_classes):.4f}")
