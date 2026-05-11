"""
app/routes/documents.py
POST /upload/arabic  — indexes an Arabic PDF (native text extraction)
POST /upload/english — indexes an English PDF (OCR extraction)
POST /reset          — wipes the index so you can start fresh
GET  /status         — returns how many articles are currently indexed
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.models.schemas import UploadResponse
from app.services import parser, vector_store

router = APIRouter()

# Noise phrases specific to Egypt's Arabic constitution PDF
_AR_NOISE = ["دستور جمهورية مصر العربية", "ديباجة وثيقة الدستور"]


# ── Arabic upload ─────────────────────────────────────────────────────────────

@router.post("/upload/arabic", response_model=UploadResponse)
async def upload_arabic(file: UploadFile = File(...)) -> UploadResponse:
    """Parse an Arabic PDF and APPEND its articles to the shared index."""
    pdf_bytes = await _read_pdf(file)

    try:
        articles_ar = parser.parse_arabic_pdf(pdf_bytes, noise_phrases=_AR_NOISE)
        if not articles_ar:
            raise HTTPException(status_code=422, detail="No Arabic articles found in this PDF.")
        vector_store.add_to_index(articles_ar)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc

    stats = vector_store.index_stats()
    return UploadResponse(
        status      ="ok",
        message     =f"Arabic PDF indexed. Index now has {stats['total']} total articles.",
        pages       =0,
        articles_en =stats["en"],
        articles_ar =stats["ar"],
    )


# ── English upload ────────────────────────────────────────────────────────────

@router.post("/upload/english", response_model=UploadResponse)
async def upload_english(file: UploadFile = File(...)) -> UploadResponse:
    """Parse an English PDF (via OCR) and APPEND its articles to the shared index."""
    pdf_bytes = await _read_pdf(file)

    try:
        articles_en = parser.parse_english_pdf(pdf_bytes, gpu=False)
        if not articles_en:
            raise HTTPException(status_code=422, detail="No English articles found in this PDF.")
        vector_store.add_to_index(articles_en)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc

    stats = vector_store.index_stats()
    return UploadResponse(
        status      ="ok",
        message     =f"English PDF indexed. Index now has {stats['total']} total articles.",
        pages       =0,
        articles_en =stats["en"],
        articles_ar =stats["ar"],
    )


# ── Legacy single upload (kept for backward compatibility) ────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Auto-detect language from filename and route to the correct parser.
    Files with 'arabic' or 'ar' in the name → Arabic parser.
    Everything else → English OCR parser.
    """
    pdf_bytes  = await _read_pdf(file)
    name_lower = (file.filename or "").lower()
    is_arabic  = "arabic" in name_lower or "_ar" in name_lower or "ar_" in name_lower

    try:
        if is_arabic:
            articles = parser.parse_arabic_pdf(pdf_bytes, noise_phrases=_AR_NOISE)
        else:
            articles = parser.parse_english_pdf(pdf_bytes, gpu=False)

        if not articles:
            raise HTTPException(status_code=422, detail="No articles found in this PDF.")

        vector_store.add_to_index(articles)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc

    stats = vector_store.index_stats()
    return UploadResponse(
        status      ="ok",
        message     =f"Indexed as {'Arabic' if is_arabic else 'English'}. Total articles: {stats['total']}.",
        pages       =0,
        articles_en =stats["en"],
        articles_ar =stats["ar"],
    )


# ── Reset ─────────────────────────────────────────────────────────────────────

@router.post("/reset")
def reset_index() -> dict:
    """Wipe the entire index. Upload both PDFs again after this."""
    vector_store.reset_index()
    return {"status": "ok", "message": "Index cleared."}


# ── Status ────────────────────────────────────────────────────────────────────

@router.get("/status")
def index_status() -> dict:
    stats = vector_store.index_stats()
    return {
        "ready"      : vector_store.index_ready(),
        "total"      : stats["total"],
        "articles_en": stats["en"],
        "articles_ar": stats["ar"],
    }


# ── Shared helper ─────────────────────────────────────────────────────────────

async def _read_pdf(file: UploadFile) -> bytes:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    return pdf_bytes