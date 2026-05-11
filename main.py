"""
LexAI — FastAPI entry point
Run with: uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes import documents, queries

app = FastAPI(
    title="LexAI",
    description="Constitutional Document Intelligence — RAG over Arabic & English PDFs",
    version="1.0.0",
)

# ── CORS (allow the HTML file served from disk / ngrok) ──────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(documents.router, prefix="", tags=["Documents"])
app.include_router(queries.router,   prefix="", tags=["Queries"])

# ── Static files (serves index.html at /static/index.html) ───────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return {"status": "ok", "message": "LexAI is running. POST /upload then POST /query."}
