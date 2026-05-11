"""
app/models/schemas.py
Pydantic request / response schemas used across routes and services.
"""

from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field


# ── Upload ────────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    status: str
    message: str
    pages: int
    articles_en: int
    articles_ar: int


# ── Query ─────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question to answer")
    model: Literal["qwen", "llama"] = Field(
        default="qwen",
        description="Which LLM backend to use: 'qwen' or 'llama'",
    )
    lang: Literal["auto", "en", "ar"] = Field(
        default="auto",
        description="Language filter. 'auto' detects from query.",
    )


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    model_used: str
    lang_detected: str
