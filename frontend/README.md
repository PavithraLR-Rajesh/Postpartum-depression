
# 🧠 Postpartum Depression Detection

This project helps detect **postpartum depression** from a mother's **voice recording** using **machine learning**.

---

## 📁 Project Files

* `data/audio_extract_feature.csv` – audio features and labels (Depressed / Not Depressed)
* `data/sample_audio.wav` – example voice file
* `frontend/app.py` – app code (for prediction)
* `frontend/requirements.txt` – needed Python packages

---

## 🎯 Goal

To find out if a mother is **depressed** or **not** by analyzing her **voice** using a trained model.

---

## ▶️ How to Run the Project

1. **Install packages**
   Open terminal and run:

   ```
   pip install -r frontend/requirements.txt
   ```

2. **Run the app**
   In terminal:

   ```
   cd frontend
   python app.py
   ```

3. **Use the app**
   It will open in your browser. Upload a `.wav` file to get the result.

---

## 🧠 How It Works

* Voice features are taken from audio (like pitch, energy, etc.)
* Machine learning model checks if it's **Depressed** or **Not Depressed**
* You see the result on the screen

---

## 📦 Tools Used

* Python
* Pandas, Scikit-learn
* Flask or Streamlit (for the app)
* Librosa (for audio features)

---


