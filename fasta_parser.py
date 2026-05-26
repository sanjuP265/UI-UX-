"""Lightweight FASTA parser — no biopython needed."""

def parse_fasta(path, max_seqs=100):
    seqs = []
    current_id, current_desc, current_seq = None, "", []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id:
                    seqs.append((current_id, current_desc, "".join(current_seq)))
                if len(seqs) >= max_seqs:
                    current_id = None
                    break
                parts = line[1:].split(None, 1)
                current_id   = parts[0]
                current_desc = parts[1] if len(parts) > 1 else ""
                current_seq  = []
            else:
                current_seq.append(line.upper())
    if current_id and len(seqs) < max_seqs:
        seqs.append((current_id, current_desc, "".join(current_seq)))
    return seqs
