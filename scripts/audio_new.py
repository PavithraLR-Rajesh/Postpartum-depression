import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Dropout, Flatten, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight

# 🔹 Load dataset
DATA_CSV = "D:/Postpartum_Depression_Detection/data/integrated_dataset.csv"
df = pd.read_csv(DATA_CSV)

# 🔹 Drop unnecessary columns
df = df.drop(columns=["Timestamp", "Filename"], errors='ignore')

# 🔹 Convert 'Age' to numeric

def convert_age(age_str):
    if isinstance(age_str, str) and "-" in age_str:
        parts = age_str.split("-")
        return (int(parts[0]) + int(parts[1])) / 2
    return np.nan if not isinstance(age_str, (int, float)) else age_str

if "Age" in df.columns:
    df["Age"] = df["Age"].apply(convert_age)

# 🔹 Ensure the binary depression label exists
if "label" not in df.columns:
    raise KeyError("❌ No 'label' column found. Check dataset.")

# 🔹 Handle missing values
df = df.dropna()

# 🔹 Define Audio Features
audio_features = [col for col in [
    'MFCC_1', 'MFCC_2', 'MFCC_3', 'Chroma_1', 'Spectral_Centroid', 'Zero_Crossing_Rate'
] if col in df.columns]

# 🔹 Normalize audio features
scaler_audio = StandardScaler()
X_audio = scaler_audio.fit_transform(df[audio_features])

# 🔹 Labels
y = df["label"]

# 🔹 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_audio, y, test_size=0.2, random_state=42)

# 🔹 Reshape for CNN/LSTM
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# 🔹 Compute class weights
weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: weights[i] for i in range(len(weights))}
print("⚖️ Class Weights:", class_weights)

# 🔹 Audio Classification Model
audio_input = Input(shape=(X_train.shape[1], 1))
x = Conv1D(64, kernel_size=3, activation="relu")(audio_input)
x = LSTM(64, return_sequences=True)(x)
x = LSTM(32)(x)
x = Dense(32, activation="relu")(x)
x = Dropout(0.3)(x)
audio_output = Dense(1, activation="sigmoid")(x)

audio_model = Model(inputs=audio_input, outputs=audio_output, name="Audio_Classifier")

# 🔹 Compile
audio_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 🔹 Train
audio_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    class_weight=class_weights
)

# 🔹 Save Model
audio_model.save("D:/Postpartum_Depression_Detection/models/audio_classification_model.h5")
print("✅ Audio classification model trained and saved successfully!")
