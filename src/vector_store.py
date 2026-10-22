"""
FAISS vector store with built-in BM25 lexical scoring.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path
import faiss
import numpy as np

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "does", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "say", "that", "the",
    "this", "to", "under", "was", "what", "when", "where", "which", "who", "with", "about"
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\b\d+(?:\.\d+)+\b|\b[a-z0-9]+\b", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class VectorStore:
    """FAISS vector index and chunk storage."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks: list[dict] = []

    def add_chunks(self, chunks: list[dict]) -> None:
        """Add embedded chunks to the FAISS index."""
        if not chunks:
            raise ValueError("No chunks provided.")

        vectors = np.asarray([c["embedding"] for c in chunks], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError("Embedding dimension does not match FAISS index.")

        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search_chunk(self, query_embedding, top_k: int = 5) -> list[dict]:
        """Return top-k chunks by dense cosine similarity."""
        if self.index.ntotal == 0:
            return []

        query_vec = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_vec)

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_vec, k)

        return [
            {
                "chunk_idx": int(idx),
                "text": self.chunks[idx]["text"],
                "metadata": self.chunks[idx].get("metadata", {}),
                "score": float(score),
            }
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    def bm25_scores(self, query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
        """Compute Okapi BM25 lexical scores for all chunks."""
        q_tokens = _tokenize(query)
        if not self.chunks or not q_tokens:
            return [0.0] * len(self.chunks)

        doc_tokens = [_tokenize(c.get("text", "")) for c in self.chunks]
        doc_freqs = [Counter(tokens) for tokens in doc_tokens]
        avgdl = sum(len(t) for t in doc_tokens) / len(self.chunks) or 1.0
        n_docs = len(self.chunks)

        idf = {}
        for token in set(q_tokens):
            df = sum(1 for counts in doc_freqs if token in counts)
            idf[token] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

        scores = []
        for counts, tokens in zip(doc_freqs, doc_tokens):
            dl = len(tokens)
            score = sum(
                idf[t] * (counts[t] * (k1 + 1)) / (counts[t] + k1 * (1 - b + b * dl / avgdl))
                for t in q_tokens if counts[t] > 0
            )
            scores.append(score)
        return scores

    def save(self, directory: str | Path) -> None:
        """Save FAISS index and chunk records to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))
        records = [{"text": c["text"], "metadata": c.get("metadata", {})} for c in self.chunks]
        (path / "chunks.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, file_path: str | Path, model=None):
        """Load saved FAISS index and chunks, or build them if not present."""
        doc_path = Path(file_path)
        store_path = Path("data/vector_store") / doc_path.stem
        index_file = store_path / "index.faiss"
        chunks_file = store_path / "chunks.json"

        if not (index_file.exists() and chunks_file.exists()):
            from src.load_chunks import load_chunks
            load_chunks(str(doc_path), model=model)

        index = faiss.read_index(str(index_file))
        chunks = json.loads(chunks_file.read_text(encoding="utf-8"))

        instance = cls(dimension=index.d)
        instance.index = index
        instance.chunks = chunks
        return instance
