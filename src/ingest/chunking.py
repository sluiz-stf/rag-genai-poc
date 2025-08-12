import re
from typing import List, Dict

def split_by_headings(text: str):
    # Heurística simples: títulos por markdown ou linhas maiúsculas longas
    parts = re.split(r"\n(?=#+\s|[A-Z][A-Z0-9 \-/]{8,}\n)", text)
    return [p.strip() for p in parts if p.strip()]

def smart_chunk(text: str, max_tokens=400, overlap_tokens=60):
    # fallback simples por frases
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], []
    cur_len = 0
    for s in sents:
        tok = len(s.split())
        if cur_len + tok > max_tokens and cur:
            chunks.append(" ".join(cur))
            # overlap
            cur = cur[-overlap_tokens//10:] if overlap_tokens else []
            cur_len = sum(len(x.split()) for x in cur)
        cur.append(s)
        cur_len += tok
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def chunk_document(doc: Dict, max_tokens=400):
    sections = split_by_headings(doc["text"]) or [doc["text"]]
    all_chunks = []
    for i, sec in enumerate(sections):
        chunks = smart_chunk(sec, max_tokens=max_tokens)
        for j, ch in enumerate(chunks):
            all_chunks.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "section": i,
                "chunk_id": f"{doc['id']}_{i}_{j}",
                "text": ch
            })
    return all_chunks