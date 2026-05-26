"""
Smart Emotion Recognition System - Flask Web Application
Deep Learning: 1D CNN + BiLSTM + Attention
"""
 
import os
import uuid
import json
import traceback
import numpy as np
import librosa
import joblib
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
 
app = Flask(__name__)
 
# FIX: use a separate temp folder, NOT 'uploads/' which has RAVDESS wav files in it
UPLOAD_FOLDER = "uploads_tmp"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB
 
# ─── CONSTANTS ─────────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050
DURATION    = 3
N_MFCC      = 40
N_MEL       = 128
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
EMOTION_COLORS = {
    "neutral":   "#64748b",
    "calm":      "#0ea5e9",
    "happy":     "#f59e0b",
    "sad":       "#6366f1",
    "angry":     "#ef4444",
    "fearful":   "#8b5cf6",
    "disgust":   "#10b981",
    "surprised": "#f97316"
}
 
EMOTION_DESCRIPTIONS = {
    "neutral":   "No strong emotion detected. Flat, steady speech patterns.",
    "calm":      "Relaxed and composed. Low energy, smooth vocal delivery.",
    "happy":     "Positive emotions detected. Elevated pitch and bright tone.",
    "sad":       "Low energy and subdued tone. Slow speech rate detected.",
    "angry":     "High energy, tense vocals. Strong emotion detected.",
    "fearful":   "Tense and anxious vocal patterns. Irregular speech flow.",
    "disgust":   "Negative arousal detected. Low, contemptuous vocal quality.",
    "surprised": "Sudden pitch changes detected. Unexpected vocal patterns."
}
 
# Fallback emojis — used in demo mode and as safety net
FALLBACK_EMOJIS = {
    "neutral": "😐", "calm": "😌", "happy": "😄", "sad": "😢",
    "angry":   "😡", "fearful": "😨", "disgust": "🤢", "surprised": "😲"
}
 
# ─── MODEL DEFINITION (must match train_model.py) ──────────────────────────────
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
 
 
# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
def load_model():
    cfg_path    = Path("models/config.json")
    model_path  = Path("models/best_model.pt")
    scaler_path = Path("models/scaler.pkl")
    le_path     = Path("models/label_encoder.pkl")
 
    # FIX: check ALL 4 required files, not just 2
    missing = [str(p) for p in [cfg_path, model_path, scaler_path, le_path]
               if not p.exists()]
    if missing:
        print(f"WARNING: Missing model files: {missing}")
        print("Run: python generate_demo_model.py")
        return None, None, None, None
 
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
 
        model = EmotionDLModel(cfg["input_dim"], cfg["num_classes"])
        # FIX: weights_only=False avoids FutureWarning/error on PyTorch >= 2.6
        model.load_state_dict(
            torch.load(model_path, map_location=DEVICE, weights_only=False)
        )
        model.eval()
 
        scaler = joblib.load(str(scaler_path))
        le     = joblib.load(str(le_path))
        return model, scaler, le, cfg
 
    except Exception:
        print("ERROR loading model:")
        print(traceback.format_exc())
        return None, None, None, None
 
 
model, scaler, le, cfg = load_model()
MODEL_LOADED = model is not None
 
 
# ─── FEATURE EXTRACTION ────────────────────────────────────────────────────────
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    target_len = SAMPLE_RATE * DURATION
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
 
    features = []
 
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
 
    delta_mfcc = librosa.feature.delta(mfcc)
    features.extend(np.mean(delta_mfcc, axis=1))
    features.extend(np.std(delta_mfcc, axis=1))
 
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))
 
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MEL)
    mel_db = librosa.power_to_db(mel)
    features.extend(np.mean(mel_db, axis=1))
    features.extend(np.std(mel_db, axis=1))
 
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.extend([np.mean(zcr), np.std(zcr)])
 
    rms = librosa.feature.rms(y=audio)
    features.extend([np.mean(rms), np.std(rms)])
 
    features.append(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr)))
    features.append(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
 
    return np.array(features, dtype=np.float32)
 
 
def get_audio_stats(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
    duration = len(audio) / sr
    rms = float(np.sqrt(np.mean(audio ** 2)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
    pitches, mags = librosa.piptrack(y=audio, sr=sr)
    pitch_vals = pitches[mags > np.median(mags)]
    avg_pitch = float(np.mean(pitch_vals)) if len(pitch_vals) > 0 else 0
    step = max(1, len(audio) // 100)
    waveform = audio[::step][:100].tolist()
    return {
        "duration": round(duration, 2),
        "rms":      round(rms, 4),
        "zcr":      round(zcr, 4),
        "pitch":    round(avg_pitch, 1),
        "waveform": waveform
    }
 
 
# ─── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", model_loaded=MODEL_LOADED)
 
 
@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
 
    file = request.files["audio"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
 
    allowed = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400
 
    # FIX: random unique name so no two requests ever collide
    filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}{ext}")
 
    # FIX: file.save() is now INSIDE try — any OS error returns clean JSON error
    try:
        file.save(filepath)
 
        features = extract_features(filepath)
        stats    = get_audio_stats(filepath)
 
        if MODEL_LOADED:
            feat_scaled = scaler.transform(features.reshape(1, -1))
            tensor = torch.FloatTensor(feat_scaled).to(DEVICE)
            with torch.no_grad():
                logits = model(tensor)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
 
            pred_idx   = int(np.argmax(probs))
            pred_label = le.inverse_transform([pred_idx])[0]
            confidence = float(probs[pred_idx])
            all_probs  = {
                le.inverse_transform([i])[0]: float(p)
                for i, p in enumerate(probs)
            }
            # FIX: safely get emoji; works whether cfg has 6 or 8 classes
            emoji = cfg.get("emojis", FALLBACK_EMOJIS).get(pred_label, "🎭")
 
        else:
            import random
            demo_emotions = list(EMOTION_COLORS.keys())
            pred_label = random.choice(demo_emotions)
            confidence = random.uniform(0.55, 0.92)
            probs_raw  = np.random.dirichlet(np.ones(8))
            probs_raw[demo_emotions.index(pred_label)] = confidence
            probs_raw  = probs_raw / probs_raw.sum()
            all_probs  = dict(zip(demo_emotions, probs_raw.tolist()))
            emoji      = FALLBACK_EMOJIS.get(pred_label, "🎭")
 
        result = {
            "emotion":     pred_label,
            "emoji":       emoji,
            "confidence":  round(confidence * 100, 1),
            "color":       EMOTION_COLORS.get(pred_label, "#888"),
            "description": EMOTION_DESCRIPTIONS.get(pred_label, ""),
            "all_probs":   all_probs,
            "stats":       stats,
            "demo_mode":   not MODEL_LOADED
        }
        return jsonify(result)
 
    except Exception as e:
        # FIX: print full traceback to console so you can see the real error
        print("=== PREDICT ERROR ===")
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
 
    finally:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass
 
 
@app.route("/model-info")
def model_info():
    if not MODEL_LOADED:
        return jsonify({"loaded": False})
    return jsonify({
        "loaded":    True,
        "classes":   cfg.get("classes", []),
        "accuracy":  round(cfg.get("best_val_acc", 0) * 100, 2),
        "input_dim": cfg.get("input_dim"),
        "device":    str(DEVICE)
    })
 
 
if __name__ == "__main__":
    print(f"Model loaded: {MODEL_LOADED}")
    print(f"Device: {DEVICE}")
    if not MODEL_LOADED:
        print("WARNING: No trained model found. Running in DEMO mode.")
        print("  Run: python generate_demo_model.py")
    # debug=False is CRITICAL — debug=True makes Flask watchdog restart mid-request
    app.run(debug=False, host="127.0.0.1", port=5000)