#trained model
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, LSTM, Dropout, Flatten
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
DATA_CSV = "D:/Postpartum_Depression_Detection/data/integrated_dataset.csv"
df = pd.read_csv(DATA_CSV)

# Debugging: Print column types before processing
print("\n🔹 Column Names and Data Types:")
print(df.dtypes)

# 🔹 Drop unnecessary columns
df = df.drop(columns=["Timestamp", "Filename"], errors='ignore')

# 🔹 Convert 'Age' to a numeric format
def convert_age(age_str):
    """Convert age range (e.g., '35-40') to a single numeric value."""
    if isinstance(age_str, str) and "-" in age_str:
        parts = age_str.split("-")
        return (int(parts[0]) + int(parts[1])) / 2  # Take the average
    elif isinstance(age_str, (int, float)):  # Already numeric
        return age_str
    return np.nan  # Handle unexpected values

df["Age"] = df["Age"].apply(convert_age)

# Check for missing values after conversion
print("\n🔹 Missing Values After Age Conversion:")
print(df.isnull().sum())

# Drop rows with missing values (if any)
df = df.dropna()

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Convert categorical labels (if necessary)
if y.dtype == 'object':
    y = y.astype('category').cat.codes

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape for CNN + LSTM
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Build CNN + BiLSTM Model
model = Sequential([
    Conv1D(64, kernel_size=3, activation="relu", input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")  # Binary classification
])

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16)

# Save model
model_path = "D:/Postpartum_Depression_Detection/models/best_model.h5"
model.save(model_path)
print(f"✅ Model training complete. Saved to {model_path}")
