"""
Pipeline entrypoint for loading documents, retrieving knowledge, and answering queries.
"""

import os
from pathlib import Path
from src.embedder import load_embedding_model
from src.vector_store import VectorStore
from src.retriever import retrieve_chunk
from src.generator import generate_answer

def run():
    # Configure document to query
    file_name = "iso27001.pdf"
    file_path = os.path.join("data/raw_pdfs", file_name)

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Source file not found at '{file_path}'")

    print("\n[1/3] Loading embedding model...")
    model = load_embedding_model()

    print(f"\n[2/3] Loading vector store for '{file_name}'...")
    vector_store = VectorStore.load(file_path=file_path, model=model)

    # Query
    query = input("Enter your query: ").strip()
    if not query:
        print("Empty query provided. Exiting.")
        return

    results = retrieve_chunk(
        query=query,
        model=model,
        vector_store=vector_store,
        top_k=8
    )
    print("\n[3/3] Generating answer...")
    answer = generate_answer(query=query, retrieved_chunks=results)
    print(f"\nAnswer:\n{answer}\n")

if __name__ == "__main__":
    run()
