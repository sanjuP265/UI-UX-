# 🎙 EmoSense AI — Smart Emotion Recognition System
### Deep Learning · 1D CNN + BiLSTM + Attention · 8 Emotions

---

## 📋 Project Overview

**Title:** Smart Emotion Recognition System Using Multimodal Learning from Speech Data  
**Model:** Deep Learning (1D CNN → BiLSTM → Attention → Classifier)  
**Emotions:** Neutral · Calm · Happy · Sad · Angry · Fearful · Disgust · Surprised  
**Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

---

## 🧠 Deep Learning Architecture

```
Audio Input (WAV/MP3/FLAC)
        ↓
Feature Extraction
  ├── MFCC (40 coeff, mean+std) = 80 features
  ├── Delta MFCC               = 80 features
  ├── Chroma STFT (12 bands)   = 24 features
  ├── Mel Spectrogram (128)    = 256 features
  ├── Zero Crossing Rate       = 2 features
  ├── RMS Energy               = 2 features
  └── Spectral (centroid/BW/rolloff) = 3 features
                                 Total: ~447 features
        ↓
1D CNN (Conv1d 64→128→256 + BatchNorm + MaxPool)
        ↓
Bi-directional LSTM (2 layers, 128 hidden, bidirectional)
        ↓
Temporal Attention Mechanism
        ↓
FC Classifier (256→128→64→8)
        ↓
Softmax → 8 Emotion Classes
```

**Why this is better than your old ML project:**
- Old: RandomForestClassifier (binary POSITIVE/NEGATIVE, random output)
- New: Deep Learning with CNN+LSTM+Attention, 8 specific emotions, real confidence scores

---

## 📦 Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset (RAVDESS)

**Option A — RAVDESS (Recommended for college project)**
1. Go to: https://zenodo.org/record/1188976
2. Download `Audio_Speech_Actors_01-24.zip`
3. Extract to: `data/ravdess/`
4. Structure should be:
```
data/ravdess/
├── Actor_01/
│   ├── 03-01-01-01-01-01-01.wav
│   ├── 03-01-02-01-01-01-01.wav
│   └── ...
├── Actor_02/
│   └── ...
...
```

**Option B — CREMA-D (Alternative)**
- Download from: https://github.com/CheyneyComputerScience/CREMA-D
- Has 7442 clips with emotion labels

**RAVDESS File Format:**
`03-01-06-01-02-01-12.wav`
- Position 3 = Emotion code:
  - 01=neutral, 02=calm, 03=happy, 04=sad
  - 05=angry, 06=fearful, 07=disgust, 08=surprised

### 3. Train the Deep Learning Model

```bash
python train_model.py
```

**Training output:**
- `models/best_model.pt` — PyTorch model weights
- `models/scaler.pkl` — Feature normalizer
- `models/label_encoder.pkl` — Label encoder
- `models/config.json` — Model config + accuracy

Training takes ~5-15 minutes on CPU. GPU is faster.

### 4. Run the Web App

```bash
python app.py
```

Open browser: `http://localhost:5000`

---

## 🗂️ Project Structure

```
EmotionDL/
├── app.py                  # Flask web app (DL inference)
├── train_model.py          # DL model training script
├── requirements.txt
├── models/                 # Generated after training
│   ├── best_model.pt
│   ├── scaler.pkl
│   ├── label_encoder.pkl
│   └── config.json
├── data/
│   └── ravdess/           # Put dataset here
│       └── Actor_XX/*.wav
├── uploads/               # Temp audio uploads
├── templates/
│   └── index.html         # Beautiful UI
└── static/
    ├── css/style.css      # Dark futuristic CSS
    └── js/
        ├── app.js         # Frontend logic
        └── neural.js      # Neural network animation
```

---

## 🎯 Features

| Feature | Old Project | New Project |
|---------|-------------|-------------|
| ML Type | Traditional ML (Random Forest) | Deep Learning (CNN+LSTM) |
| Emotions | Binary (positive/negative) + random | 8 specific emotions |
| Output | Random guess from pool | Real confidence scores |
| Features | 13 MFCC | 447 features (MFCC+Chroma+Mel+ZCR+Spectral) |
| UI | Basic form | Animated neural network UI |
| Visualizations | None | Waveform + probability bars + stats |
| Demo mode | N/A | Works without model for testing |

---

## 📊 Expected Results

On RAVDESS dataset (1440 files):
- Training accuracy: ~85-90%
- Validation accuracy: ~70-80%

To improve accuracy:
1. Use data augmentation (pitch shift, time stretch, add noise)
2. Fine-tune hyperparameters in `train_model.py`
3. Try CREMA-D or IEMOCAP for larger training data

---

## 🔧 Customization

**Change number of epochs:**  
Edit `EPOCHS = 80` in `train_model.py`

**Change emotions:**  
The system auto-adapts to whatever emotions are in RAVDESS filename codes.

**Use your own dataset:**  
Modify `load_ravdess()` in `train_model.py` to parse your own folder structure and labels.

---

## 👩‍💻 Tech Stack

- **Deep Learning:** PyTorch
- **Audio:** Librosa
- **Backend:** Flask
- **Frontend:** HTML5 + CSS3 + Vanilla JS
- **Features:** MFCC, Chroma, Mel Spectrogram, ZCR, RMS, Spectral

---

*Built as End-Semester Project — Smart Emotion Recognition System*
