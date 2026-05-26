# Deep Learning Enhanced Needleman–Wunsch
### End Semester Project — Advanced Bioinformatics Dashboard

---

## Project Structure

```
dna_advanced/
├── app.py              ← Flask backend (all API routes)
├── needleman.py        ← NW algorithms (Classic, ML-optimised, Affine-Gap)
├── fasta_parser.py     ← Lightweight FASTA reader (no biopython needed)
├── requirements.txt    ← Python dependencies
├── data/
│   └── raw_dataset.fasta   ← YOUR dataset (copy here)
├── templates/
│   └── index.html      ← Full dashboard UI
└── static/
    └── style.css       ← Styling
```

---

## How to Run

### Step 1 — Copy your dataset
```bash
# Copy raw_dataset.fasta into the data/ folder
cp /path/to/raw_dataset.fasta data/raw_dataset.fasta
```

### Step 2 — Install dependencies
```bash
pip install flask scikit-learn numpy
```
*(If using Python 3.11+ system-wide: `pip install flask scikit-learn numpy --break-system-packages`)*

### Step 3 — Run the Flask app
```bash
python app.py
```

### Step 4 — Open in browser
```
http://127.0.0.1:5000
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Classic NW** | Standard Needleman-Wunsch (match=1, mismatch=−1, gap=−2) |
| **ML-Optimised NW** | Adaptive scoring based on GC content + Shannon entropy |
| **Affine-Gap NW** | Biologically realistic gap opening + extension model |
| **Full Traceback** | Colour-coded alignment viewer (match/mismatch/gap) |
| **Batch Analysis** | All pairwise alignments for first 6 sequences |
| **Custom Input** | Paste your own sequences for instant alignment |
| **Dataset Explorer** | GC content bars, length stats for all loaded sequences |

---

## ML Scoring Logic

The ML-optimised mode derives alignment parameters from sequence properties:

| Condition | Match | Mismatch | Gap |
|-----------|-------|----------|-----|
| GC > 60%  | +3    | −3       | −2  |
| GC 45–60% | +2    | −2       | −1  |
| GC < 45%  | +2    | −1       | −1  |
| Low entropy (< 1.5 bits) | — | — | −1 extra |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Dashboard |
| GET | `/api/sequences` | List all loaded sequences |
| POST | `/api/align` | Align two sequences by index |
| POST | `/api/align_custom` | Align custom pasted sequences |
| GET | `/api/batch` | Pairwise batch alignment table |
| GET | `/api/stats` | Dataset-level statistics |
