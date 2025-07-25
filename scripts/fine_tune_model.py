import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Dropout, Flatten, Input, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# 🔹 Define Features
mcq_features = [col for col in [
    'Feeling sad or Tearful', 'Irritable towards baby & partner', 
    'Trouble sleeping at night', 'Problems concentrating or making decision',
    'Overeating or loss of appetite', 'Feeling anxious', 'Feeling of guilt',
    'Problems of bonding with baby', 'Suicide attempt'
] if col in df.columns]

audio_features = [col for col in [
    'MFCC_1', 'MFCC_2', 'MFCC_3', 'Chroma_1', 'Spectral_Centroid', 'Zero_Crossing_Rate'
] if col in df.columns]

# 🔹 Normalize features
scaler_mcq = StandardScaler()
scaler_audio = StandardScaler()

X_mcq = scaler_mcq.fit_transform(df[mcq_features])
X_audio = scaler_audio.fit_transform(df[audio_features])

# 🔹 Labels
y = df["label"]  # Binary: 1 = Depressed, 0 = Not Depressed

# 🔹 Train-test split
X_train_mcq, X_test_mcq, y_train, y_test = train_test_split(X_mcq, y, test_size=0.2, random_state=42)
X_train_audio, X_test_audio, _, _ = train_test_split(X_audio, y, test_size=0.2, random_state=42)

# 🔹 Reshape for CNN/LSTM
X_train_mcq = np.expand_dims(X_train_mcq, axis=2)
X_test_mcq = np.expand_dims(X_test_mcq, axis=2)
X_train_audio = np.expand_dims(X_train_audio, axis=2)
X_test_audio = np.expand_dims(X_test_audio, axis=2)

# 🔹 1️⃣ MCQ Model
mcq_input = Input(shape=(X_train_mcq.shape[1], 1))
mcq_layer = Conv1D(64, kernel_size=3, activation="relu")(mcq_input)
mcq_layer = LSTM(64, return_sequences=True)(mcq_layer)
mcq_layer = LSTM(32)(mcq_layer)
mcq_layer = Dense(32, activation="relu")(mcq_layer)
mcq_layer = Dropout(0.2)(mcq_layer)
mcq_output = Dense(16, activation="relu", name="mcq_output")(mcq_layer)  # Final MCQ output

mcq_model = Model(inputs=mcq_input, outputs=mcq_output, name="MCQ_Model")

# 🔹 2️⃣ Audio Model
audio_input = Input(shape=(X_train_audio.shape[1], 1))
audio_layer = Conv1D(64, kernel_size=3, activation="relu")(audio_input)
audio_layer = LSTM(64, return_sequences=True)(audio_layer)
audio_layer = LSTM(32)(audio_layer)
audio_layer = Dense(32, activation="relu")(audio_layer)
audio_layer = Dropout(0.2)(audio_layer)
audio_output = Dense(16, activation="relu", name="audio_output")(audio_layer)  # Final Audio output

audio_model = Model(inputs=audio_input, outputs=audio_output, name="Audio_Model")

# 🔹 3️⃣ Fusion Model (Equal Priority)
merged = Concatenate()([mcq_output, audio_output])
fusion_layer = Dense(16, activation="relu")(merged)  # Fully connected layer
final_output = Dense(1, activation="sigmoid", name="final_output")(fusion_layer)

fusion_model = Model(inputs=[mcq_input, audio_input], outputs=final_output, name="Fusion_Model")

# 🔹 Compile & Train Fusion Model
fusion_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
fusion_model.fit(
    [X_train_mcq, X_train_audio], y_train,
    validation_data=([X_test_mcq, X_test_audio], y_test),
    epochs=20, batch_size=16
)

# 🔹 Save Models
mcq_model.save("D:/Postpartum_Depression_Detection/models/mcq_model.h5")
audio_model.save("D:/Postpartum_Depression_Detection/models/audio_model.h5")
fusion_model.save("D:/Postpartum_Depression_Detection/models/fusion_model.h5")

print("✅ All models trained and saved successfully!")

# 🔹 Severity Prediction
def predict_severity(mcq_data, audio_data):
    mcq_data = scaler_mcq.transform([mcq_data]).reshape(1, -1, 1)
    audio_data = scaler_audio.transform([audio_data]).reshape(1, -1, 1)
    
    prob = fusion_model.predict([mcq_data, audio_data])[0][0]
    if prob < 0.25:
        return "No Depression"
    elif prob < 0.5:
        return "Mild"
    elif prob < 0.75:
        return "Moderate"
    else:
        return "Severe"

print("\n✅ Model Ready! Use 'predict_severity(mcq_features, audio_features)' to test.")
