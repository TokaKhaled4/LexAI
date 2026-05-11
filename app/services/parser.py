"""
app/services/parser.py
PDF text extraction and article chunking for both English and Arabic.
Pure functions — no global state.

English path: reads from pre-extracted text file (extracted_text_en.txt)
              placed in the same directory as main.py.
              The uploaded PDF is accepted by the route but ignored for extraction.
              OCR code is kept below but commented out.
Arabic path:  native PyMuPDF text extraction (unchanged).
"""

from __future__ import annotations
import re
import unicodedata
import os

import fitz          # PyMuPDF


# ══════════════════════════════════════════════════════════════════════════════
# ENGLISH — pre-extracted text file
# ══════════════════════════════════════════════════════════════════════════════

# Path to the pre-extracted OCR text file.
# Place extracted_text_en.txt next to main.py (i.e. inside the lexai/ folder).
_EN_TEXT_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),  # → lexai/
    "extracted_text_en.txt",
)


def _load_english_text() -> str:
    """Read the pre-extracted English text file from disk."""
    if not os.path.exists(_EN_TEXT_FILE):
        raise FileNotFoundError(
            f"Pre-extracted English text not found at: {_EN_TEXT_FILE}\n"
            "Place 'extracted_text_en.txt' inside the lexai/ folder."
        )
    with open(_EN_TEXT_FILE, "r", encoding="utf-8") as f:
        return f.read()


# ── OCR extraction (commented out — kept for reference) ──────────────────────
# def extract_text_ocr(pdf_bytes: bytes, gpu: bool = False) -> str:
#     """Render every page at 300 DPI and run EasyOCR over it."""
#     import easyocr
#     import numpy as np
#
#     reader = easyocr.Reader(["en"], gpu=gpu)
#     doc = fitz.open(stream=pdf_bytes, filetype="pdf")
#     raw_text = ""
#
#     for page in doc:
#         mat = fitz.Matrix(300 / 72, 300 / 72)
#         pix = page.get_pixmap(matrix=mat)
#         img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
#             pix.height, pix.width, pix.n
#         )
#         if pix.n == 4:
#             img = img[:, :, :3]
#
#         results = reader.readtext(img, detail=0, paragraph=True)
#         raw_text += "\n".join(results) + "\n"
#
#     doc.close()
#     return raw_text
# ─────────────────────────────────────────────────────────────────────────────


def clean_ocr_english(text: str) -> str:
    text = re.sub(
        r'\bArtic1e\b|\bArticie\b|\bARTIC1E\b|\bArt1cle\b',
        "Article", text, flags=re.IGNORECASE
    )
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def split_articles_english(text: str) -> list[dict]:
    pattern = r'(Article\s*\(\s*\d+\s*\))'
    parts   = re.split(pattern, text, flags=re.IGNORECASE)

    articles: list[dict] = []
    for i in range(1, len(parts), 2):
        header = parts[i].strip()
        body   = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if len(body) < 20:
            continue
        num_match = re.search(r'\d+', header)
        num = int(num_match.group()) if num_match else -1
        articles.append({
            "article_id" : f"en_{num}",
            "article_num": num,
            "header"     : header,
            "text"       : f"{header} {body}",
            "lang"       : "en",
        })

    seen: dict[int, dict] = {}
    for art in articles:
        n = art["article_num"]
        if n not in seen or len(art["text"]) > len(seen[n]["text"]):
            seen[n] = art

    return sorted(seen.values(), key=lambda x: x["article_num"])


def parse_english_pdf(pdf_bytes: bytes, gpu: bool = False) -> list[dict]:
    """
    Parse English articles from the pre-extracted text file.
    pdf_bytes is accepted so the route signature stays the same but ignored.
    gpu is kept for API compatibility but unused.
    """
    raw  = _load_english_text()
    text = clean_ocr_english(raw)
    return split_articles_english(text)


# ══════════════════════════════════════════════════════════════════════════════
# ARABIC — native PDF text extraction
# ══════════════════════════════════════════════════════════════════════════════

_AR_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

_PAT_WITH_PAREN = re.compile(
    r"م\s?ا\s?د\s?ة\s*[\(\)]\s*"
    r"((?:[٠-٩]|\d)(?:[\n\s]*(?:[٠-٩]|\d))*)"
    r"\s*[\(\)]"
)
_PAT_NO_OPEN_PAREN = re.compile(r"مادة(\d+)\n\)")


def _extract_text_native(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    raw = "\n".join(page.get_text() for page in doc)
    doc.close()
    return raw


def _clean_arabic(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    for ch in '\u202a\u202b\u202c\u202d\u202e\u200e\u200f\u2066\u2067\u2068\u2069':
        text = text.replace(ch, '')
    text = re.sub(r'[\u0610-\u061A\u064B-\u065F]', '', text)

    text = text.replace("اإل", "الإ")
    text = text.replace("األ", "الأ")
    text = text.replace("اآل", "الآ")

    text = re.sub(r'\bال\b',  'لا',  text)
    text = re.sub(r'\bوال\b', 'ولا', text)
    text = re.sub(r'\bإال\b', 'إلا', text)
    text = re.sub(r'\bأال\b', 'ألا', text)
    text = re.sub(r'\bفال\b', 'فلا', text)

    text = re.sub(r'\b([بتثجحخدذرزسشصضطظعغفقكلمنهي])\s+', r'\1', text)
    text = re.sub(r'\s+([،\.؛:؟!])', r'\1', text)
    text = re.sub(r'([،\.؛:؟!])(?=[^\s\d])', r'\1 ', text)
    return re.sub(r' {2,}', ' ', text).strip()


def _normalize_num(raw: str) -> int:
    d = "".join(raw.split())
    if re.fullmatch(r"[٠-٩]+", d):
        d = d[::-1]
    return int(d.translate(_AR_INDIC))


def _normalize_body(body: str) -> str:
    body = re.sub(r"([\u0600-\u06FF])\n([ىا])(?=[\n ،.،؟!]|$)", r"\1\2", body)
    body = re.sub(r"(?<!\n)\n(?!\n)", " ", body)
    return re.sub(r" +", " ", body).strip()


def _chunk_arabic(text: str, noise_phrases: list[str] | None = None) -> list[dict]:
    noise_phrases = noise_phrases or []
    seen: dict[int, int] = {}

    for m in _PAT_WITH_PAREN.finditer(text):
        try:
            n = _normalize_num(m.group(1))
            if n not in seen:
                seen[n] = m.start()
        except ValueError:
            continue

    for m in _PAT_NO_OPEN_PAREN.finditer(text):
        try:
            n = int(m.group(1))
            if n not in seen:
                seen[n] = m.start()
        except ValueError:
            continue

    headers = sorted((offset, num) for num, offset in seen.items())
    articles: list[dict] = []

    for i, (start, num) in enumerate(headers):
        end     = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        raw_body = text[start:end]
        raw_body = re.sub(r"^م\s?ا\s?د\s?ة[\s\S]{0,20}?\)\s*", "", raw_body).strip()
        if len(raw_body) < 10:
            continue

        body = _normalize_body(raw_body)
        for noise in noise_phrases:
            body = body.replace(noise, "")

        articles.append({
            "article_id" : f"ar_{num}",
            "article_num": num,
            "header"     : f"مادة ({num})",
            "text"       : "\u200f" + f"مادة ({num})\n" + body.strip(),
            "lang"       : "ar",
        })

    articles.sort(key=lambda a: a["article_num"])
    return articles


def parse_arabic_pdf(
    pdf_bytes: bytes,
    noise_phrases: list[str] | None = None,
) -> list[dict]:
    raw  = _extract_text_native(pdf_bytes)
    text = _clean_arabic(raw)
    return _chunk_arabic(text, noise_phrases)