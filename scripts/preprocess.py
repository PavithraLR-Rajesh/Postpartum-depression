from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
df = pd.read_csv('../data/dataset.csv')  # Update path if needed

# Features & labels
X = df.drop(columns=['label'])  # Features
y = df['label']  # Labels (0: Not Depressed, 1: Depressed)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Save to files (optional)
X_train.to_csv("../data/X_train.csv", index=False)
X_test.to_csv("../data/X_test.csv", index=False)
y_train.to_csv("../data/y_train.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)

print("✅ Dataset split successfully!")
