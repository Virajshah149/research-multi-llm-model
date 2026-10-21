"""
PDF Ingestion and Document Processing Pipeline.
Extracts text from paper1.pdf, cleans headers/footers, drops references,
splits into semantically overlapping chunks, and deduplicates passages.
"""

import re
from pathlib import Path
from typing import List, Dict, Any
import pypdf

from config import PDF_PATH, CHUNK_SIZE, CHUNK_OVERLAP


def clean_page_text(text: str) -> str:
    """Removes journal headers, recurring footers, publisher banners, and OCR/formatting noise."""
    # Clean whole lines of standard journal boilerplate
    boilerplate_patterns = [
        r"Provided proper attribution[\s\S]*?scholarly works\.[^\n]*\n?",
        r"Available online at www\.[^\n]+",
        r"www\.elsevier\.com[^\n]*",
        r"ScienceDirect",
        r"This is an open access article under the CC[^\n]*",
        r"Peer-review under responsibility of[^\n]*",
        r"International Conference on Machine Learning[^\n]*",
        r"Data Engineering",
        r"10\.1016/j\.procs[^\n]*",
        r"1877-0509[^\n]*",
        r"1877\s*-0509[^\n]*",
        r"Published by Elsevier[^\n]*",
        r"The Authors\.",
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove running headers
    text = re.sub(
        r"\d*\s*Aditi\s+Govindu\s+et\s+al\.\s*/\s*Procedia\s+Computer\s+Science[^\n]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Author\s+name\s*/\s*Procedia\s+Computer\s+Science[^\n]*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"Procedia\s+Computer\s+Science\s+\d+\s*\(\d{4}\)\s*\d+[-–]\d+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    # Remove standalone page numbers
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    
    # Normalize whitespaces and line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def extract_and_clean_pdf(pdf_path: Path = PDF_PATH) -> List[Dict[str, Any]]:
    """
    Extracts pages from the PDF, cleans text, and truncates before the References section.
    Returns a list of page objects with text and page numbers.
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"Source PDF not found at {pdf_path}")

    reader = pypdf.PdfReader(str(pdf_path))
    pages = []
    hit_references = False

    for page_idx, page in enumerate(reader.pages):
        raw_text = page.extract_text() or ""
        cleaned = clean_page_text(raw_text)

        # Truncate when References section is encountered (standard on page 12)
        ref_match = re.search(r"\n\s*References\s*\n", cleaned, flags=re.IGNORECASE)
        if ref_match:
            cleaned = cleaned[: ref_match.start()].strip()
            hit_references = True

        if cleaned:
            pages.append({"page": page_idx + 1, "text": cleaned})

        if hit_references:
            break

    return pages


def recursive_character_chunker(
    text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Splits text using hierarchical delimiters (paragraphs -> sentences -> words)."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Find best break point within separator hierarchy
        split_at = -1
        sub_text = text[start:end]
        for sep in separators:
            idx = sub_text.rfind(sep)
            if idx != -1 and idx > chunk_overlap:
                split_at = start + idx + len(sep)
                break

        if split_at == -1:
            split_at = end

        chunk = text[start:split_at].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, split_at - chunk_overlap)

    return chunks


def build_deduplicated_chunks(
    pdf_path: Path = PDF_PATH,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """
    Loads PDF, chunks text, and applies deduplication based on normalized text content.
    Returns list of chunk dicts ready for embedding and indexing.
    """
    pages = extract_and_clean_pdf(pdf_path)
    all_chunks: List[Dict[str, Any]] = []
    seen_hashes = set()
    chunk_id = 0

    for p in pages:
        page_chunks = recursive_character_chunker(p["text"], chunk_size, chunk_overlap)
        for chunk in page_chunks:
            # Skip very short fragments
            if len(chunk.split()) < 15:
                continue

            # Deduplication key: normalized alphanumeric words
            norm_key = " ".join(re.findall(r"\w+", chunk.lower()))
            if not norm_key or norm_key in seen_hashes:
                continue

            seen_hashes.add(norm_key)
            all_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "page": p["page"],
                    "text": chunk,
                    "char_count": len(chunk),
                }
            )
            chunk_id += 1

    return all_chunks


if __name__ == "__main__":
    chunks = build_deduplicated_chunks()
    print(f"Extraction successful: {len(chunks)} deduplicated chunks created.")
    print("Sample chunk 0:")
    print(f"[Page {chunks[0]['page']}] {chunks[0]['text'][:200]}...")
