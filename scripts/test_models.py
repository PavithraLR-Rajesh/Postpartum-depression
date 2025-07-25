import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MCQ_MODEL_PATH = os.path.join(BASE_DIR, '../models/mcq_model.h5')
AUDIO_MODEL_PATH = os.path.join(BASE_DIR, '../models/audio_model.h5')
DATA_PATH = os.path.join(BASE_DIR, '../data/integrated_dataset.csv')

# Load dataset
df = pd.read_csv(DATA_PATH)

# Identify feature columns
mcq_features = ['feature1', 'feature2', 'feature3']  # Replace with actual MCQ features
audio_features = ['audio_feature1', 'audio_feature2']  # Replace with actual audio features
label_column = 'depression_label'  # Replace with actual label column

# Handle missing values (if any)
df.dropna(inplace=True)

# Convert categorical features (if applicable)
if df[label_column].dtype == 'object':
    label_encoder = LabelEncoder()
    df[label_column] = label_encoder.fit_transform(df[label_column])

# Ensure numerical encoding for MCQ features
for col in mcq_features:
    if df[col].dtype == 'object':  # Convert categorical text to numerical
        df[col] = LabelEncoder().fit_transform(df[col])

# Scale features
scaler_mcq = StandardScaler()
scaler_audio = StandardScaler()

X_mcq = scaler_mcq.fit_transform(df[mcq_features])
X_audio = scaler_audio.fit_transform(df[audio_features])
y = df[label_column].values  # Labels

# Load trained models
try:
    mcq_model = tf.keras.models.load_model(MCQ_MODEL_PATH)
    audio_model = tf.keras.models.load_model(AUDIO_MODEL_PATH)
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# Predict using both models
mcq_predictions = mcq_model.predict(X_mcq)
audio_predictions = audio_model.predict(X_audio)

# Convert predictions to binary labels (assuming binary classification)
mcq_labels = (mcq_predictions > 0.5).astype(int)
audio_labels = (audio_predictions > 0.5).astype(int)

# Evaluate model performance
print("MCQ Model Evaluation:")
print(classification_report(y, mcq_labels))

print("Audio Model Evaluation:")
print(classification_report(y, audio_labels))

# Fusion model (simple averaging of both predictions)
fusion_predictions = (mcq_predictions + audio_predictions) / 2
fusion_labels = (fusion_predictions > 0.5).astype(int)

print("Fusion Model Evaluation:")
print(classification_report(y, fusion_labels))
