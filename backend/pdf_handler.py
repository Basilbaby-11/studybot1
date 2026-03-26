"""
pdf_handler.py — Handles PDF and text document ingestion.
Extracts text from uploaded files for the RAG pipeline.

Supported formats: .pdf, .txt, .md
"""

import re
from pathlib import Path
from typing import Optional
from backend.config import DATA_DIR


# ── PDF Extraction ─────────────────────────────────────────────────────

def extract_text_from_pdf(file_path: str | Path) -> str:
    """
    Extract all text from a PDF using pypdf.
    Falls back gracefully if a page has no extractable text.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf not installed. Run: pip install pypdf")

    reader = PdfReader(str(file_path))
    pages_text = []

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages_text.append(f"[Page {page_num + 1}]\n{text}")

    if not pages_text:
        return ""

    return "\n\n".join(pages_text)


def extract_text_from_txt(file_path: str | Path) -> str:
    """Read plain text / markdown files."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def extract_text(file_path: str | Path) -> str:
    """
    Auto-detect file type and extract text.
    Returns cleaned text string.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        raw = extract_text_from_pdf(path)
    elif suffix in (".txt", ".md"):
        raw = extract_text_from_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .pdf, .txt, or .md")

    return clean_text(raw)


# ── Text Cleaning ──────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Remove noise from extracted text:
    - collapse excessive whitespace
    - strip non-printable characters
    - fix common PDF extraction artifacts
    """
    # Remove non-printable characters
    text = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]', ' ', text)
    # Collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Collapse 2+ spaces to 1
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


# ── Save Uploaded File ─────────────────────────────────────────────────

def save_upload(filename: str, content: bytes) -> Path:
    """
    Save an uploaded file to the data/documents directory.
    Returns the saved file path.
    """
    safe_name = re.sub(r'[^\w.\-]', '_', filename)
    dest = DATA_DIR / safe_name

    with open(dest, 'wb') as f:
        f.write(content)

    return dest


# ── Metadata ──────────────────────────────────────────────────────────

def get_pdf_metadata(file_path: str | Path) -> dict:
    """Return basic metadata: page count, title, author if available."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(file_path))
        meta = reader.metadata or {}
        return {
            "pages": len(reader.pages),
            "title": meta.get("/Title", ""),
            "author": meta.get("/Author", ""),
        }
    except Exception:
        return {"pages": 0, "title": "", "author": ""}
