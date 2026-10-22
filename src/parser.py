"""
Extract and clean text from PDF and text documents.
"""

import re
from pathlib import Path
import pymupdf
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    """Remove control characters, PDF watermarks, hyphenated line wraps, and TOC dot-leaders."""
    text = re.sub(r"[\u200b-\u200d\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[\xa0\u2000-\u200a\t]", " ", text)
    text = re.sub(r"[-`',]{6,}", "", text)
    text = re.sub(r"(?m)^\s*Table\s+A\.1\s*\(continued\)\s*$", "", text)
    text = re.sub(r"([a-zA-Z])-\s*\n\s*([a-z])", r"\1\2", text)
    text = re.sub(r"\.{4,}", " ... ", text)
    text = re.sub(r"[ ]{2,}", " ", text)

    lines = [line.strip() for line in text.splitlines()]
    while lines and (
        not lines[-1]
        or re.fullmatch(r"\d{1,3}|[ivxlcdm]{1,5}", lines[-1])
        or "all rights reserved" in lines[-1].lower()
    ):
        lines.pop()

    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def parse_text(file_path: str) -> list[Document]:
    """Extract text from a PDF or text file and return a list of page Documents."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    documents = []
    try:
        if path.suffix.lower() == ".pdf":
            with pymupdf.open(file_path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text = clean_text(page.get_text())
                    if text:
                        documents.append(
                            Document(
                                page_content=text,
                                metadata={"source": str(file_path), "page": page_num},
                            )
                        )
        else:
            text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": str(file_path), "page": 1},
                    )
                )
    except Exception as e:
        print(f"Extraction failed for {file_path}, Reason: {e}")

    return documents
