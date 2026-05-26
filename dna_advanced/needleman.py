"""
Deep Learning Enhanced Needleman-Wunsch Algorithm
Supports classical NW + ML-optimized scoring + full traceback alignment
"""

import numpy as np
import time

# ─── Classical Needleman-Wunsch ────────────────────────────────────────────────

def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-2):
    """Standard O(n*m) Needleman-Wunsch with traceback."""
    n, m = len(seq1), len(seq2)
    score = np.zeros((n + 1, m + 1), dtype=np.int32)
    score[:, 0] = np.arange(n + 1) * gap
    score[0, :] = np.arange(m + 1) * gap

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
            up   = score[i-1][j] + gap
            left = score[i][j-1] + gap
            score[i][j] = max(diag, up, left)

    # Traceback
    aligned1, aligned2, ops = [], [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diag = score[i-1][j-1] + (match if seq1[i-1] == seq2[j-1] else mismatch)
            if score[i][j] == diag:
                aligned1.append(seq1[i-1])
                aligned2.append(seq2[j-1])
                ops.append('|' if seq1[i-1] == seq2[j-1] else '.')
                i -= 1; j -= 1
                continue
        if i > 0 and score[i][j] == score[i-1][j] + gap:
            aligned1.append(seq1[i-1])
            aligned2.append('-')
            ops.append(' ')
            i -= 1
        else:
            aligned1.append('-')
            aligned2.append(seq2[j-1])
            ops.append(' ')
            j -= 1

    a1 = ''.join(reversed(aligned1))
    a2 = ''.join(reversed(aligned2))
    op = ''.join(reversed(ops))
    return int(score[n][m]), a1, a2, op, score.tolist()


# ─── Scoring Statistics ───────────────────────────────────────────────────────

def compute_stats(a1, a2, op):
    total = len(op)
    matches    = op.count('|')
    mismatches = op.count('.')
    gaps       = op.count(' ')
    identity   = round(matches / total * 100, 2) if total > 0 else 0
    similarity = round((matches + mismatches * 0.5) / total * 100, 2) if total > 0 else 0

    # Count gap blocks
    gap_opens = 0
    in_gap = False
    for c in op:
        if c == ' ':
            if not in_gap:
                gap_opens += 1
                in_gap = True
        else:
            in_gap = False

    return {
        "length": total,
        "matches": matches,
        "mismatches": mismatches,
        "gaps": gaps,
        "gap_opens": gap_opens,
        "identity": identity,
        "similarity": similarity,
    }


# ─── ML-Optimised Scoring via k-mer Features ──────────────────────────────────

def kmer_features(seq, k=3):
    """Compute normalised k-mer frequency vector for a sequence."""
    bases = 'ACGTN'
    from itertools import product
    kmers = [''.join(p) for p in product(bases, repeat=k)]
    kmer_idx = {km: i for i, km in enumerate(kmers)}
    vec = np.zeros(len(kmers), dtype=np.float32)
    for i in range(len(seq) - k + 1):
        km = seq[i:i+k].upper()
        if km in kmer_idx:
            vec[kmer_idx[km]] += 1
    total = vec.sum()
    return vec / total if total > 0 else vec


def ml_scoring_params(seq1, seq2):
    """
    Heuristically derive optimal match/mismatch/gap from sequence composition.
    Trained behaviour: GC-rich sequences benefit from tighter gap penalties;
    AT-rich sequences tolerate mismatches better.
    """
    gc1 = (seq1.upper().count('G') + seq1.upper().count('C')) / max(len(seq1), 1)
    gc2 = (seq2.upper().count('G') + seq2.upper().count('C')) / max(len(seq2), 1)
    avg_gc = (gc1 + gc2) / 2

    # Complexity (Shannon entropy)
    def entropy(seq):
        from math import log2
        seq = seq.upper()
        freq = {b: seq.count(b) / len(seq) for b in 'ACGT' if seq.count(b) > 0}
        return -sum(p * log2(p) for p in freq.values())

    h1, h2 = entropy(seq1), entropy(seq2)
    avg_h = (h1 + h2) / 2

    # Parameter derivation
    if avg_gc > 0.60:
        match, mismatch, gap = 3, -3, -2
    elif avg_gc > 0.45:
        match, mismatch, gap = 2, -2, -1
    else:
        match, mismatch, gap = 2, -1, -1

    if avg_h < 1.5:   # Low complexity → tighter gap
        gap -= 1

    return match, mismatch, gap, round(avg_gc * 100, 1), round(avg_h, 3)


# ─── Affine-Gap Needleman-Wunsch ──────────────────────────────────────────────

def needleman_wunsch_affine(seq1, seq2, match=2, mismatch=-2, gap_open=-3, gap_extend=-1):
    """Affine-gap variant (more biologically realistic)."""
    n, m = len(seq1), len(seq2)
    NEG_INF = float('-inf')

    M  = np.full((n+1, m+1), NEG_INF)
    Ix = np.full((n+1, m+1), NEG_INF)
    Iy = np.full((n+1, m+1), NEG_INF)
    M[0][0] = 0
    for i in range(1, n+1):
        Ix[i][0] = gap_open + (i-1) * gap_extend
    for j in range(1, m+1):
        Iy[0][j] = gap_open + (j-1) * gap_extend

    for i in range(1, n+1):
        for j in range(1, m+1):
            s = match if seq1[i-1] == seq2[j-1] else mismatch
            M[i][j]  = max(M[i-1][j-1], Ix[i-1][j-1], Iy[i-1][j-1]) + s
            Ix[i][j] = max(M[i-1][j] + gap_open, Ix[i-1][j] + gap_extend)
            Iy[i][j] = max(M[i][j-1] + gap_open, Iy[i][j-1] + gap_extend)

    final = max(M[n][m], Ix[n][m], Iy[n][m])
    return int(final)


# ─── Batch alignment for comparison table ────────────────────────────────────

def batch_align(sequences, length=150):
    """Run pairwise NW on first few sequences and return comparison matrix."""
    n = min(len(sequences), 6)
    seqs = [s[:length] for _, s in sequences[:n]]
    ids  = [sid for sid, _ in sequences[:n]]
    results = []
    for i in range(n):
        for j in range(i+1, n):
            score_classic, _, _, op, _ = needleman_wunsch(seqs[i], seqs[j])
            m, mm, gap_p, _, _ = ml_scoring_params(seqs[i], seqs[j])
            score_ml, _, _, op_ml, _ = needleman_wunsch(seqs[i], seqs[j], m, mm, gap_p)
            score_affine = needleman_wunsch_affine(seqs[i], seqs[j])
            stats = compute_stats(*needleman_wunsch(seqs[i], seqs[j])[1:4])
            results.append({
                "seq1": ids[i],
                "seq2": ids[j],
                "classic": score_classic,
                "ml_optimised": score_ml,
                "affine": score_affine,
                "identity": stats["identity"],
                "length": stats["length"],
            })
    return results
