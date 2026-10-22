"""
Generate grounded answers with verified citations using Groq.
"""

import os
import re
import time
import random
from dotenv import load_dotenv
from groq import Groq, APIStatusError

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

SYSTEM_PROMPT = """Answer the question using ONLY the supplied context.
1. Base every claim strictly on the context. If the answer is not in the context, state clearly that it is not available.
2. Cite factual claims using source IDs in brackets like [S1], [S2].
3. Only cite source IDs present in the context. Keep the answer clear and concise."""


def _clean_citations(text: str) -> tuple[str, list[str]]:
    """Normalize citation brackets (e.g. 【S1†L1-L4】 -> [S1]) and extract cited source IDs."""
    cleaned = text.replace("\u200b", "")
    cleaned = re.sub(r"【\s*(S\d+)(?:[^】]*?)】", r"[\1]", cleaned)
    cleaned = re.sub(r"\[\s*(S\d+)†[^\]]*\]", r"[\1]", cleaned)

    cited = [m for block in re.findall(r"\[([^\]]+)\]", cleaned) for m in re.findall(r"\b(S\d+)\b", block)]
    return cleaned, cited


def generate_answer(query: str, retrieved_chunks: list[dict]) -> str:
    """Generate an answer grounded in retrieved document chunks."""
    if not retrieved_chunks:
        return "I couldn't find relevant information in the document."

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

    context_parts = []
    valid_sources = {}
    for i, chunk in enumerate(retrieved_chunks, 1):
        sid = f"S{i}"
        meta = chunk.get("metadata", {})
        valid_sources[sid] = {"source": meta.get("source", "Unknown"), "page": meta.get("page", "Unknown")}
        context_parts.append(f"[{sid}] File: {valid_sources[sid]['source']} | Page: {valid_sources[sid]['page']}\n{chunk.get('text', '')}")

    user_prompt = f"Context:\n{'\n\n---\n\n'.join(context_parts)}\n\nQuestion:\n{query}"
    client = Groq(api_key=api_key)
    response = None

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
            )
            break
        except APIStatusError as exc:
            if exc.status_code not in RETRYABLE_STATUS_CODES or attempt == MAX_RETRIES - 1:
                raise
            delay = (2 ** attempt) + random.uniform(0, 1)
            print(f"Groq HTTP {exc.status_code}. Retrying in {delay:.1f}s...")
            time.sleep(delay)

    answer = (response.choices[0].message.content or "").strip() if (response and response.choices) else ""
    if not answer:
        return "The model did not return a text answer."

    answer, cited_ids = _clean_citations(answer)
    invalid_ids = sorted(set(cited_ids) - set(valid_sources))
    if invalid_ids:
        answer += f"\n\nWarning: Unrecognized source IDs: {', '.join(invalid_ids)}"

    used_ids = [sid for sid in dict.fromkeys(cited_ids) if sid in valid_sources]
    if used_ids:
        answer += "\n\nSources:" + "".join(
            f"\n- [{sid}] Page {valid_sources[sid]['page']} — {valid_sources[sid]['source']}"
            for sid in used_ids
        )

    return answer
