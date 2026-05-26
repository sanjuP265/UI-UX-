"""
Smart Emotion Recognition System - Deep Learning Training Script
Dataset: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
Download: https://zenodo.org/record/1188976

RAVDESS filename format: 03-01-06-01-02-01-12.wav
  - Modality: 03=Audio-only
  - Vocal channel: 01=Speech
  - Emotion: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
  - Emotional intensity: 01=normal, 02=strong
  - Statement: 01-02
  - Repetition: 01-02
  - Actor: 01-24

Place RAVDESS .wav files in:  data/ravdess/Actor_XX/*.wav
"""

import os
import numpy as np
import librosa
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# ─── CONFIG ────────────────────────────────────────────────────────────────────
SAMPLE_RATE    = 22050
DURATION       = 3          # seconds
N_MFCC         = 40
N_CHROMA       = 12
N_MEL          = 128
BATCH_SIZE     = 32
EPOCHS         = 80
LEARNING_RATE  = 1e-3
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

EMOTION_EMOJIS = {
    "neutral":   "😐",
    "calm":      "😌",
    "happy":     "😄",
    "sad":       "😢",
    "angry":     "😡",
    "fearful":   "😨",
    "disgust":   "🤢",
    "surprised": "😲"
}

# ─── FEATURE EXTRACTION ────────────────────────────────────────────────────────
def extract_features(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """Extract rich audio features: MFCC + Chroma + Mel + ZCR + RMS"""
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        # Pad/trim to fixed length
        target_len = sr * duration
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        features = []

        # 1. MFCCs (40 coefficients × time → mean + std = 80)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # 2. Delta MFCCs
        delta_mfcc = librosa.feature.delta(mfcc)
        features.extend(np.mean(delta_mfcc, axis=1))
        features.extend(np.std(delta_mfcc, axis=1))

        # 3. Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        # 4. Mel Spectrogram (128 bands → mean + std = 256)
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MEL)
        mel_db = librosa.power_to_db(mel)
        features.extend(np.mean(mel_db, axis=1))
        features.extend(np.std(mel_db, axis=1))

        # 5. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        # 6. RMS Energy
        rms = librosa.feature.rms(y=audio)
        features.append(np.mean(rms))
        features.append(np.std(rms))

        # 7. Spectral features
        spec_cent = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features.append(np.mean(spec_cent))
        spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        features.append(np.mean(spec_bw))
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features.append(np.mean(rolloff))

        return np.array(features, dtype=np.float32)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


# ─── LOAD RAVDESS DATA ─────────────────────────────────────────────────────────
def load_ravdess(data_dir="data/ravdess"):
    X, y = [], []
    path = Path(data_dir)
    files = list(path.rglob("*.wav"))
    print(f"Found {len(files)} audio files")

    for f in files:
        parts = f.stem.split("-")
        if len(parts) < 7:
            continue
        emotion_code = parts[2]
        if emotion_code not in EMOTIONS:
            continue
        label = EMOTIONS[emotion_code]
        features = extract_features(str(f))
        if features is not None:
            X.append(features)
            y.append(label)

    print(f"Loaded {len(X)} samples")
    return np.array(X), np.array(y)


# ─── DATASET CLASS ─────────────────────────────────────────────────────────────
class EmotionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── DEEP LEARNING MODEL ───────────────────────────────────────────────────────
class EmotionDLModel(nn.Module):
    """
    1D CNN + BiLSTM + Attention + Classifier
    Architecture for speech emotion recognition
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),
        )

        # Bi-LSTM
        cnn_out_size = input_dim // 4  # after 2 MaxPool1d(2)
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)

        # Attention
        self.attn = nn.Linear(256, 1)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, features) → reshape for CNN: (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.cnn(x)                          # (batch, 256, len//4)
        x = x.permute(0, 2, 1)                  # (batch, len//4, 256)
        x, _ = self.lstm(x)                      # (batch, len//4, 256)

        # Attention pooling
        attn_w = torch.softmax(self.attn(x), dim=1)
        x = (x * attn_w).sum(dim=1)              # (batch, 256)

        return self.classifier(x)


# ─── TRAINING LOOP ─────────────────────────────────────────────────────────────
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            total_loss += criterion(out, y).item()
            correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)


# ─── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    print("Loading RAVDESS dataset...")

    X, y_labels = load_ravdess("data/ravdess")

    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_ds = EmotionDataset(X_train, y_train)
    test_ds  = EmotionDataset(X_test,  y_test)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
    test_dl  = DataLoader(test_ds,  BATCH_SIZE)

    num_classes = len(le.classes_)
    model = EmotionDLModel(X.shape[1], num_classes).to(DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Classes: {list(le.classes_)}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print("\n Training started...\n")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train(model, train_dl, optimizer, criterion)
        va_loss, va_acc = evaluate(model, test_dl, criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), "models/best_model.pt")
            print(f"  ✅ Epoch {epoch:3d} | Train Acc: {tr_acc:.3f} | Val Acc: {va_acc:.3f} ← SAVED")
        elif epoch % 10 == 0:
            print(f"     Epoch {epoch:3d} | Train Acc: {tr_acc:.3f} | Val Acc: {va_acc:.3f}")

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(le,     "models/label_encoder.pkl")

    cfg = {
        "input_dim":   int(X.shape[1]),
        "num_classes": int(num_classes),
        "classes":     list(le.classes_),
        "emojis":      {c: EMOTION_EMOJIS.get(c, "🎭") for c in le.classes_},
        "best_val_acc": float(best_acc),
        "history":     history
    }
    with open("models/config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"\n🎯 Best Validation Accuracy: {best_acc:.4f}")

    # Final evaluation
    model.load_state_dict(torch.load("models/best_model.pt", map_location=DEVICE))
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for X_b, y_b in test_dl:
            preds = model(X_b.to(DEVICE)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(y_b.numpy())

    print("\nClassification Report:")
    print(classification_report(all_true, all_preds, target_names=le.classes_))

    print("\nAll model files saved to models/")
    print("Run: python app.py   to start the web app")
