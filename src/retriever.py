"""
Hybrid retriever combining FAISS vector similarity and BM25 lexical search.
"""

import re
from src.embedder import embed_query
from src.vector_store import VectorStore

METADATA_WORDS = {"title", "author", "edition", "version", "published", "copyright", "overview", "summary"}
CLAUSE_REGEX = re.compile(r"\b(\d+(?:\.\d+)+)\b|(?:clause|section|control)\s+(\d+(?:\.\d+)*)", re.IGNORECASE)


def retrieve_chunk(query: str, model, vector_store: VectorStore, top_k: int = 8) -> list[dict]:
    """Retrieve the top-k most relevant chunks using Hybrid Search (FAISS + BM25 + RRF)."""
    chunks = vector_store.chunks
    if not chunks:
        return []

    n = len(chunks)
    q_lower = query.lower()
    q_words = set(re.findall(r"\b[a-z0-9]+\b", q_lower))
    clauses = {m.group(1) or m.group(2) for m in CLAUSE_REGEX.finditer(query) if m.group(1) or m.group(2)}

    # 1. Dense FAISS search
    dense_hits = vector_store.search_chunk(embed_query(model, query), top_k=n)
    dense_rank = {h["chunk_idx"]: rank for rank, h in enumerate(dense_hits, 1)}
    dense_sim = {h["chunk_idx"]: h["score"] for h in dense_hits}

    # 2. BM25 lexical search
    bm25 = vector_store.bm25_scores(query)
    bm25_order = sorted(range(n), key=lambda i: bm25[i], reverse=True)
    bm25_rank = {idx: rank for rank, idx in enumerate(bm25_order, 1) if bm25[idx] > 0}

    # 3. Combine scores with Reciprocal Rank Fusion (RRF)
    results = []
    for i, chunk in enumerate(chunks):
        text = chunk.get("text", "")
        is_toc = text.count(" ... ") >= 4

        score = 1.0 / (20 + dense_rank.get(i, n + 1)) + 0.05 * dense_sim.get(i, 0.0)
        if i in bm25_rank:
            score += 0.85 / (20 + bm25_rank[i])

        if not is_toc:
            for c in clauses:
                if re.search(rf"(?:^|\n)\s*{re.escape(c)}\s+[A-Z]", text):
                    score += 0.25
                elif re.search(rf"\b{re.escape(c)}\b", text):
                    score += 0.10

            if "scope" in q_words and re.search(r"(?:^|\n)\s*1\s+Scope\b", text):
                score += 0.20
            if "annex" in q_words and not clauses and re.search(r"(?:^|\n)\s*Annex\s+A\b", text, re.IGNORECASE):
                score += 0.20

        if i == 0 and (q_words & METADATA_WORDS or "reference number" in q_lower or "full title" in q_lower):
            score += 0.25

        if is_toc and "contents" not in q_lower:
            score *= 0.30

        results.append({
            "chunk_idx": i,
            "text": text,
            "metadata": chunk.get("metadata", {}),
            "score": float(score),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
