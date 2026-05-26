"""
Deep Learning Enhanced Needleman-Wunsch — Flask Backend
Endpoints:
  GET  /                  → Dashboard
  GET  /api/sequences     → List loaded sequences
  POST /api/align         → Run alignment (classic + ML + affine)
  POST /api/align_custom  → Custom sequences from textarea
  GET  /api/batch         → Batch pairwise comparison table
  GET  /api/stats         → Dataset-level statistics
"""

import os, time, json
from flask import Flask, render_template, jsonify, request
from needleman import (
    needleman_wunsch, needleman_wunsch_affine,
    ml_scoring_params, compute_stats, batch_align
)
from fasta_parser import parse_fasta

app = Flask(__name__)

# ─── Load Dataset Once ────────────────────────────────────────────────────────
FASTA_PATH = os.path.join(os.path.dirname(__file__), "data", "raw_dataset.fasta")
SEQUENCES = []

def load_sequences():
    global SEQUENCES
    try:
        SEQUENCES = parse_fasta(FASTA_PATH, max_seqs=20)
        print(f"[✓] Loaded {len(SEQUENCES)} sequences from {FASTA_PATH}")
    except Exception as e:
        print(f"[!] Could not load FASTA: {e}")
        SEQUENCES = []

load_sequences()

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/sequences")
def api_sequences():
    data = [
        {
            "id": sid,
            "description": desc[:80],
            "length": len(seq),
            "gc": round((seq.count("G") + seq.count("C")) / max(len(seq), 1) * 100, 1)
        }
        for sid, desc, seq in SEQUENCES
    ]
    return jsonify({"sequences": data, "total": len(data)})


@app.route("/api/align", methods=["POST"])
def api_align():
    body = request.get_json(force=True)
    idx1 = int(body.get("idx1", 0))
    idx2 = int(body.get("idx2", 1))
    seg_len = int(body.get("length", 200))
    seg_len = min(seg_len, 500)

    if idx1 >= len(SEQUENCES) or idx2 >= len(SEQUENCES):
        return jsonify({"error": "Index out of range"}), 400

    sid1, desc1, full1 = SEQUENCES[idx1]
    sid2, desc2, full2 = SEQUENCES[idx2]
    seq1 = full1[:seg_len]
    seq2 = full2[:seg_len]

    # ── Classic NW ──
    t0 = time.perf_counter()
    c_score, a1, a2, op, matrix = needleman_wunsch(seq1, seq2)
    t_classic = round((time.perf_counter() - t0) * 1000, 2)
    c_stats = compute_stats(a1, a2, op)

    # ── ML-Optimised NW ──
    match, mismatch, gap, gc_pct, entropy = ml_scoring_params(seq1, seq2)
    t0 = time.perf_counter()
    ml_score, ml_a1, ml_a2, ml_op, _ = needleman_wunsch(seq1, seq2, match, mismatch, gap)
    t_ml = round((time.perf_counter() - t0) * 1000, 2)
    ml_stats = compute_stats(ml_a1, ml_a2, ml_op)

    # ── Affine Gap NW ──
    t0 = time.perf_counter()
    aff_score = needleman_wunsch_affine(seq1, seq2, match, mismatch, gap_open=-3, gap_extend=-1)
    t_affine = round((time.perf_counter() - t0) * 1000, 2)

    improvement = round((ml_score - c_score) / max(abs(c_score), 1) * 100, 2) if c_score != 0 else 0

    return jsonify({
        "seq1": {"id": sid1, "desc": desc1[:60], "length": len(seq1), "segment": seq1[:80] + "..."},
        "seq2": {"id": sid2, "desc": desc2[:60], "length": len(seq2), "segment": seq2[:80] + "..."},
        "classic": {
            "score": c_score, "time_ms": t_classic,
            "params": {"match": 1, "mismatch": -1, "gap": -2},
            "stats": c_stats,
            "aligned1": a1[:120], "aligned2": a2[:120], "ops": op[:120],
        },
        "ml_optimised": {
            "score": ml_score, "time_ms": t_ml,
            "params": {"match": match, "mismatch": mismatch, "gap": gap},
            "stats": ml_stats,
            "gc_pct": gc_pct, "entropy": entropy,
            "aligned1": ml_a1[:120], "aligned2": ml_a2[:120], "ops": ml_op[:120],
        },
        "affine": {
            "score": aff_score, "time_ms": t_affine,
            "params": {"gap_open": -3, "gap_extend": -1},
        },
        "improvement_pct": improvement,
        "segment_length": seg_len,
    })


@app.route("/api/align_custom", methods=["POST"])
def api_align_custom():
    body = request.get_json(force=True)
    seq1 = body.get("seq1", "").upper().replace(" ", "").replace("\n", "")
    seq2 = body.get("seq2", "").upper().replace(" ", "").replace("\n", "")

    if not seq1 or not seq2:
        return jsonify({"error": "Both sequences required"}), 400
    if len(seq1) > 500 or len(seq2) > 500:
        seq1, seq2 = seq1[:500], seq2[:500]

    c_score, a1, a2, op, _ = needleman_wunsch(seq1, seq2)
    c_stats = compute_stats(a1, a2, op)
    match, mismatch, gap, gc_pct, entropy = ml_scoring_params(seq1, seq2)
    ml_score, ml_a1, ml_a2, ml_op, _ = needleman_wunsch(seq1, seq2, match, mismatch, gap)
    ml_stats = compute_stats(ml_a1, ml_a2, ml_op)
    aff_score = needleman_wunsch_affine(seq1, seq2)

    improvement = round((ml_score - c_score) / max(abs(c_score), 1) * 100, 2) if c_score != 0 else 0

    return jsonify({
        "seq1": {"id": "Custom-A", "length": len(seq1)},
        "seq2": {"id": "Custom-B", "length": len(seq2)},
        "classic": {
            "score": c_score, "stats": c_stats,
            "params": {"match": 1, "mismatch": -1, "gap": -2},
            "aligned1": a1, "aligned2": a2, "ops": op,
        },
        "ml_optimised": {
            "score": ml_score, "stats": ml_stats,
            "params": {"match": match, "mismatch": mismatch, "gap": gap},
            "gc_pct": gc_pct, "entropy": entropy,
            "aligned1": ml_a1, "aligned2": ml_a2, "ops": ml_op,
        },
        "affine": {"score": aff_score},
        "improvement_pct": improvement,
    })


@app.route("/api/batch")
def api_batch():
    length = int(request.args.get("length", 120))
    results = batch_align([(sid, seq) for sid, _, seq in SEQUENCES], length=length)
    return jsonify({"results": results})


@app.route("/api/stats")
def api_stats():
    if not SEQUENCES:
        return jsonify({"error": "No sequences loaded"}), 500
    lengths = [len(seq) for _, _, seq in SEQUENCES]
    gc_vals = [(seq.count("G") + seq.count("C")) / max(len(seq), 1) * 100 for _, _, seq in SEQUENCES]
    return jsonify({
        "total_sequences": len(SEQUENCES),
        "total_bp": sum(lengths),
        "avg_length": round(sum(lengths) / len(lengths)),
        "min_length": min(lengths),
        "max_length": max(lengths),
        "avg_gc": round(sum(gc_vals) / len(gc_vals), 2),
        "sequence_ids": [sid for sid, _, _ in SEQUENCES[:10]],
        "lengths": lengths[:10],
        "gc_vals": [round(g, 2) for g in gc_vals[:10]],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
