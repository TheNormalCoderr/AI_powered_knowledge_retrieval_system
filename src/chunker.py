"""
Split documents into overlapping chunks and merge tiny page fragments.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_document(doc, chunk_size: int = 1000, chunk_overlap: int = 180):
    """Divide documents into overlapping chunks without leaving tiny orphan fragments."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_documents(documents=doc)

    chunks = []
    for chunk in raw_chunks:
        text = chunk.page_content.strip()
        page = chunk.metadata.get("page")

        if chunks and chunks[-1].metadata.get("page") == page and len(chunks[-1].page_content) < 100:
            chunks[-1].page_content += f"\n{text}"
        elif chunks and chunks[-1].metadata.get("page") == page and len(text) < 100:
            chunks[-1].page_content += f"\n{text}"
        else:
            chunks.append(chunk)

    return chunks
