# ─────────────────────────────────────────────────────────────────────────────
# LexAI — Dockerfile
# Place this file inside:  NLP Project\Dockerfile
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ── System libraries ──────────────────────────────────────────────────────────
# libgl1 + libglib2.0-0  → required by OpenCV (used inside EasyOCR)
# libgomp1               → required by faiss-cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory inside the container ────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ───────────────────────────────────────────────
# Copy requirements first so Docker caches this layer (faster rebuilds)
COPY requirements.txt .

# Strip inline comments before installing (pip can choke on them)
RUN sed 's/#.*//' requirements.txt > requirements_clean.txt \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements_clean.txt

# ── Pre-download ML models at build time ─────────────────────────────────────
# Baked into the image → container starts instantly, works offline
RUN python - <<'EOF'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
print("Models downloaded successfully.")
EOF

# ── Copy application source code ──────────────────────────────────────────────
COPY main.py .
COPY app/  ./app/
COPY static/ ./static/

# ── Copy the pre-extracted English text (replaces OCR at runtime) ─────────────
COPY extracted_text_en.txt .

# ── Expose API port ───────────────────────────────────────────────────────────
EXPOSE 8000

# ── Launch the server ─────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
