import pandas as pd

# File Paths
POST_NATAL_CSV = "D:/Postpartum_Depression_Detection/data/post_natal.csv"
AUDIO_FEATURES_CSV = "D:/Postpartum_Depression_Detection/data/audio_extract_feature.csv"
OUTPUT_CSV = "D:/Postpartum_Depression_Detection/data/integrated_dataset.csv"

# Load datasets
post_natal_df = pd.read_csv(POST_NATAL_CSV)
audio_features_df = pd.read_csv(AUDIO_FEATURES_CSV)

# Rename 'Depression' column to 'label' in post_natal_df
if "Depression" in post_natal_df.columns:
    post_natal_df.rename(columns={"Depression": "label"}, inplace=True)

# Convert 'Depressed'/'Not Depressed' to 1/0 in audio_features_df
if "label" in audio_features_df.columns:
    audio_features_df["label"] = audio_features_df["label"].map({"Depressed": 1, "Not Depressed": 0})

# Check if label exists in both DataFrames
if "label" not in post_natal_df.columns:
    raise KeyError("Error: 'label' column is missing in post_natal.csv after renaming. Cannot merge.")

if "label" not in audio_features_df.columns:
    raise KeyError("Error: 'label' column is missing in audio_extract_feature.csv after conversion. Cannot merge.")

# Merge datasets on 'label'
merged_df = post_natal_df.merge(audio_features_df, on="label", how="inner")

# Save merged dataset
merged_df.to_csv(OUTPUT_CSV, index=False)
print("Datasets merged successfully! Saved to:", OUTPUT_CSV)
