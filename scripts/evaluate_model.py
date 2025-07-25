import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
DATA_CSV = "D:/Postpartum_Depression_Detection/data/integrated_dataset.csv"
df = pd.read_csv(DATA_CSV)

# Preprocessing
df = df.drop(columns=["Timestamp", "Filename"], errors='ignore')

# Convert 'Age' to numeric
df["Age"] = df["Age"].apply(lambda x: np.mean([int(i) for i in str(x).split('-')]) if '-' in str(x) else int(x))

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load trained model
MODEL_PATH = "D:/Postpartum_Depression_Detection/models/best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Make predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Compute Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print results
print(f"\n✅ Model Evaluation:")
print(f"🔹 Accuracy: {accuracy:.4f}")
print(f"🔹 Confusion Matrix:\n{conf_matrix}")
print(f"🔹 Classification Report:\n{class_report}")

# Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Depressed", "Depressed"], yticklabels=["Not Depressed", "Depressed"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
