
# 🧠 Postpartum Depression Detection App

This Flask-based web application helps detect **postpartum depression** using both **MCQ questionnaire responses** and **audio analysis**. It uses pre-trained deep learning models to make predictions based on user input.

---

## 🚀 Features

- MCQ-based depression screening using a trained neural network
- Audio analysis using speech features (MFCC, Chroma, etc.)
- Final decision based on combined MCQ and Audio model outputs
- Web interface for users to submit data
- Real-time results with predictions

---

## 🛠 Requirements

Install Python packages using:

```bash
pip install -r requirements.txt
````

Required tools:

* Python 3.8 or above
* FFmpeg (for audio conversion)

To install **FFmpeg**:

* **Windows**: Download from [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html) and add it to your system PATH.
* **Linux (Debian/Ubuntu)**: `sudo apt install ffmpeg`
* **MacOS**: `brew install ffmpeg`

---

## 📂 Project Structure

```plaintext
Postpartum-depression/
│
├── app3.py                      # Main Flask app
├── requirements.txt
├── README.md                   # This file
│
├── templates/                  # HTML templates for web app
│   ├── home.html
│   ├── questionnaire.html
│   ├── audio_analysis.html
│   └── result.html
│
├── models/                     # Pre-trained ML models
│   ├── mcq_model.h5
│   ├── audio_model.h5
│   └── fusion_model.h5
│
├── data/                       # CSV dataset for scaling
│   ├── integrated_dataset.csv
│   └── sample_audio.wav
│
├── uploads/                    # Uploaded and converted audio files
│   ├── audio.webm
│   └── audio.wav
│
├── scripts/                    # Extra scripts for training/evaluation
│   ├── evaluate_model.py
│   ├── extract_audio_features.py
│   ├── ml_predict.py
│   ├── predict_depression.py
│   └── (more...)
```

---

## 📦 How to Run the Web App

1. Make sure you have all the required models and `integrated_dataset.csv` in place.

2. Run the Flask app:

```bash
python app.py
```

3. Open your browser and go to:

```
http://127.0.0.1:5000/
```

4. Use the MCQ form or upload audio to get predictions.

---

## 🧪 How to Run CLI Testing Script

You can also test predictions directly via command line using one of your scripts (like `predict_depression.py`):

```bash
python predict_depression.py
```

* Choose:

  * `1` for MCQ-based test
  * `2` for Audio-only
  * `3` for Fusion (both MCQ & audio)
* Make sure `test_audio.wav` is placed inside the correct directory (`uploads/` or path used in script)

---

## 🔍 How It Works

* **MCQ Model**: Takes scaled questionnaire inputs and outputs a depression score.
* **Audio Model**: Extracts features like MFCCs, Chroma, Spectral Centroid, Zero-Crossing Rate using Librosa.
* **Fusion Model**: Combines both MCQ and audio data for a more reliable prediction.
* Each model outputs a value which is interpreted as:

  * `Not Depressed` if score ≤ 1
  * `Depressed` if score > 1

---

## 🧪 Experimental / Unused Code

This project includes some **test files**, **experimental scripts**, or **older model versions** that are not used directly in the main app.

These include:

* `scripts/`: training, testing, model evaluation, and feature extraction scripts
* Extra models like `best_model.h5`, `fine_tune_model.py`, etc.
* Unused datasets or audio samples in `data/`

These files are retained for:

* Debugging
* Reproducibility
* Further experimentation

You can ignore them if you're just running the app, but feel free to explore them if you're interested in model training or backend logic.

---

## 📧 Contact

Feel free to reach out if you have questions or want to contribute!

```

---


```
