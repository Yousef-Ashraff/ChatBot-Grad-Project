"""
lecture_service.py — In-memory lecture context store
=====================================================

Stores a parsed PDF lecture per student in RAM.
The backend receives a base64-encoded PDF from the Flutter app (/set-lecture),
extracts plain text from it, and keeps it here.

When the student sends a question while a lecture is active, chatbot_api.chat()
calls get_lecture_context(student_id) and answers directly from the lecture
text — bypassing the LangGraph agent entirely.

No file system used — everything lives in a dict so it works on any host.

Install dependency:
    uv add pypdf        (preferred — matches pyproject.toml)
    # OR
    pip install pypdf
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# student_id → {"text": str, "name": str}
_lecture_store: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def set_lecture(student_id: str, pdf_b64: str, lecture_name: str) -> str:
    """
    Decode a base64 PDF, extract its text, cache it for the student.
    Returns the extracted plain text.
    Raises RuntimeError with a human-readable message on any failure.
    """
    # 1. Decode base64
    try:
        pdf_bytes = base64.b64decode(pdf_b64)
    except Exception as exc:
        raise RuntimeError(f"Invalid base64 payload: {exc}") from exc

    logger.info(
        "[LectureService] set_lecture: student=%s name=%r bytes=%d",
        student_id, lecture_name, len(pdf_bytes),
    )

    # 2. Extract text
    text = _extract_text(pdf_bytes)   # raises RuntimeError if no library / no text

    if not text.strip():
        raise RuntimeError(
            "Could not extract any text from the PDF. "
            "The file may be a scanned / image-only PDF. "
            "Please use a text-based PDF."
        )

    # 3. Store
    _lecture_store[student_id] = {"text": text, "name": lecture_name}
    logger.info(
        "[LectureService] stored lecture for %s: %r (%d chars extracted)",
        student_id, lecture_name, len(text),
    )
    return text


def clear_lecture(student_id: str) -> None:
    """Remove any active lecture for this student."""
    removed = _lecture_store.pop(student_id, None)
    if removed:
        logger.info("[LectureService] cleared lecture for %s", student_id)


def get_lecture_context(student_id: str) -> Optional[dict]:
    """Return {"text": str, "name": str} if a lecture is active, else None."""
    return _lecture_store.get(student_id)


# ─────────────────────────────────────────────────────────────────────────────
# PDF text extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text(pdf_bytes: bytes) -> str:
    """
    Extract all text from a PDF given as raw bytes.
    Tries pypdf first, then PyMuPDF (fitz), then pdfminer.six.
    Raises RuntimeError if no library is available.
    """
    errors = []

    # ── pypdf (preferred, listed in pyproject.toml) ───────────────────────
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n\n".join(p for p in pages if p.strip())
        logger.info("[LectureService] pypdf extracted %d chars", len(text))
        return text
    except ImportError:
        errors.append("pypdf not installed")
    except Exception as exc:
        errors.append(f"pypdf error: {exc}")

    # ── PyMuPDF / fitz (optional) ─────────────────────────────────────────
    try:
        import fitz
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = [doc[i].get_text() for i in range(doc.page_count)]
        doc.close()
        text = "\n\n".join(p for p in pages if p.strip())
        logger.info("[LectureService] fitz extracted %d chars", len(text))
        return text
    except ImportError:
        errors.append("PyMuPDF (fitz) not installed")
    except Exception as exc:
        errors.append(f"fitz error: {exc}")

    # ── pdfminer.six (optional) ───────────────────────────────────────────
    try:
        from pdfminer.high_level import extract_text_to_fp
        from pdfminer.layout import LAParams
        output = io.StringIO()
        extract_text_to_fp(io.BytesIO(pdf_bytes), output, laparams=LAParams())
        text = output.getvalue()
        logger.info("[LectureService] pdfminer extracted %d chars", len(text))
        return text
    except ImportError:
        errors.append("pdfminer.six not installed")
    except Exception as exc:
        errors.append(f"pdfminer error: {exc}")

    raise RuntimeError(
        "No PDF library available. Run: uv add pypdf\n"
        f"Attempted: {'; '.join(errors)}"
    )