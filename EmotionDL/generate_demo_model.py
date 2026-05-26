"""
generate_demo_model.py
Run this to create a DEMO model so you can test the UI immediately
WITHOUT downloading any dataset.

This creates a random (untrained) model just to verify everything works.
For real predictions, train with: python train_model.py
"""
import os
import json
import numpy as np
import torch
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

os.makedirs("models", exist_ok=True)

# Must match app.py
import torch.nn as nn

class EmotionDLModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),
        )
        self.lstm = nn.LSTM(256, 128, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.3)
        self.attn = nn.Linear(256, 1)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        attn_w = torch.softmax(self.attn(x), dim=1)
        x = (x * attn_w).sum(dim=1)
        return self.classifier(x)

EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
EMOJIS   = {"neutral":"😐","calm":"😌","happy":"😄","sad":"😢",
            "angry":"😡","fearful":"😨","disgust":"🤢","surprised":"😲"}

INPUT_DIM = 447

# Create demo model (random weights)
model = EmotionDLModel(INPUT_DIM, 8)
torch.save(model.state_dict(), "models/best_model.pt")
print("✅ Demo model saved: models/best_model.pt")

# Scaler
scaler = StandardScaler()
scaler.fit(np.random.randn(100, INPUT_DIM))
joblib.dump(scaler, "models/scaler.pkl")
print("✅ Scaler saved: models/scaler.pkl")

# Label encoder
le = LabelEncoder()
le.fit(EMOTIONS)
joblib.dump(le, "models/label_encoder.pkl")
print("✅ Label encoder saved: models/label_encoder.pkl")

# Config
cfg = {
    "input_dim": INPUT_DIM,
    "num_classes": 8,
    "classes": EMOTIONS,
    "emojis": EMOJIS,
    "best_val_acc": 0.0,
    "note": "DEMO MODEL — predictions are random. Train with train_model.py for real results."
}
with open("models/config.json", "w") as f:
    json.dump(cfg, f, indent=2)
print("✅ Config saved: models/config.json")

print("\n🚀 Demo model ready! Run: python app.py")
print("   For real accuracy, download RAVDESS and run: python train_model.py")
