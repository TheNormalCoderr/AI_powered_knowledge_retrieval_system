"""
Generate and persist vector index and chunks from a document.
"""

from pathlib import Path
from src.parser import parse_text
from src.chunker import chunk_document
from src.embedder import load_embedding_model, embed_chunk
from src.vector_store import VectorStore


def load_chunks(file_path: str, model=None):
    """Parse, chunk, embed, and store document in FAISS."""
    # Step 1: Parsing text
    text = parse_text(file_path=file_path)
    if not text:
        raise ValueError(
            f"No extractable text found in '{file_path}'. "
            "If this is a scanned document, ensure OCR tools (pytesseract) are installed."
        )

    # Step 2: Splitting text into chunks
    chunks = chunk_document(doc=text)
    print(f"Total chunks created: {len(chunks)}")
    if not chunks:
        raise ValueError(f"No chunks generated from '{file_path}'.")

    # Step 3: Loading the embedding model if not already provided
    if model is None:
        model = load_embedding_model()

    # Step 4: Generating embeddings
    embedded_chunks = embed_chunk(model=model, chunks=chunks, batch_size=32)
    if not embedded_chunks:
        raise ValueError(f"Failed to create embeddings for chunks in '{file_path}'.")

    # Step 5: Store embedded chunks in vector index using FAISS
    dimension = len(embedded_chunks[0]["embedding"])
    vector_store = VectorStore(dimension)
    vector_store.add_chunks(embedded_chunks)

    # Save to data/vector_store/<filename_stem>
    save_dir = Path("data/vector_store") / Path(file_path).stem
    vector_store.save(directory=str(save_dir))

    print(f"Chunks saved successfully to '{save_dir}'")
