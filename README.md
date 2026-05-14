# LexAI — Constitutional Document Intelligence
A bilingual RAG (Retrieval-Augmented Generation) system that answers natural-language questions over Egypt's Constitution in Arabic and English. This project includes document parsing, semantic chunking, vector-based retrieval, cross-encoder reranking, and a FastAPI backend — deployed as an interactive web application.

## Overview
Understanding constitutional law is complex and language-sensitive. Using Egypt's Constitution in both Arabic and English, this project aims to:
- Answer **natural-language queries** in Arabic or English (RAG pipeline)
- **Retrieve the most relevant constitutional articles** using semantic search
- **Generate grounded, cited answers** via state-of-the-art LLMs

## Dataset
- **Source**: Egypt's Constitution (2014) — Arabic & English versions
- **Arabic articles**: 223 chunks
- **English articles**: 247 chunks
- **Total indexed**: 470 article-level chunks
- **Chunk unit**: One constitutional article = one chunk

## Preprocessing Steps
- Arabic text extraction using PyMuPDF with Unicode normalisation (NFKC)
- English text loaded from pre-extracted OCR file (`extracted_text_en.txt`)
- Article boundary detection via regex (`Article (N)` / `المادة رقم (N)`)
- Noise phrase removal and ligature repair for Arabic text
- Deduplication by article ID (`en_N` / `ar_N`)
- Minimum body length filtering (removes header-only noise)

## Retrieval Pipeline
- **Bi-encoder**: `paraphrase-multilingual-mpnet-base-v2` — encodes Arabic + English into a shared vector space
- **Index**: FAISS `IndexFlatIP` with L2-normalised vectors (cosine similarity)
- **Reranker**: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` — two-stage retrieval (top-50 → rerank → top-5)
- **Language detection**: automatic query routing based on Arabic character ratio

## Models Used
### LLM Backends (via Groq API)
- **`qwen/qwen3-32b`** — primary model, strong multilingual & Arabic legal reasoning
- **`llama-3.3-70b-versatile`** — English summarisation and reasoning

### Retrieval Models
- **Bi-encoder**: paraphrase-multilingual-mpnet-base-v2
- **Cross-encoder**: mmarco-mMiniLMv2-L12-H384-v1 (reranking)

## Docker
- Containerised with a single `Dockerfile` using `python:3.11-slim`
- ML models baked into the image at build time (offline-ready)
- `docker-compose.yml` maps port `8000` and reads `GROQ_API_KEY` from `.env`
- Start everything with `docker compose up --build`

## Deployment
Built and deployed on **Hugging Face Spaces** using FastAPI + static HTML frontend.

### App Features
- PDF upload (Arabic & English)
- Natural-language query interface
- Grounded answers with article citations
- Auto language detection (Arabic / English)
- Index status monitoring

🔗 [Launch the App](https://huggingface.co/spaces/tokakhaled24/LexAI/)

---
