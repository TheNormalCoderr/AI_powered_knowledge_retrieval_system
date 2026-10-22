"""
Adds a embedding to metadata of pages in chunks converting text to vector embeds
"""

MODEL_NAME = "BAAI/bge-base-en-v1.5"

def load_embedding_model():
    """Load the pretrained embedding model"""
    print("Initializing model loading...")
    import torch
    from sentence_transformers import SentenceTransformer
    
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    try:
        model = SentenceTransformer(MODEL_NAME, device=device, local_files_only=True)
    except Exception:
        print(f"Downloading '{MODEL_NAME}' (this can take a few minutes on the first run)...")
        model = SentenceTransformer(MODEL_NAME, device=device)

    print("Model loaded successfully!")
    return model

def embed_chunk(model, chunks, batch_size=32):
    """Embed chunks into vectors"""
    if not chunks:
        return []
    texts = [chunk.page_content for chunk in chunks]
    
    embeddings = model.encode(
        texts, 
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True
    )
    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append({
            "text": chunk.page_content,
            "metadata": chunk.metadata,
            "embedding": embedding.tolist()
        })

    return embedded_chunks

def embed_query(model, query:str) -> list[float]:
    """Convert a search query into a normalized embedding."""
    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    embedding = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embedding.tolist()

