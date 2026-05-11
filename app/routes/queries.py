"""
app/routes/queries.py
POST /query — retrieve relevant articles and generate an LLM answer.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models.schemas import QueryRequest, QueryResponse
from app.services import llm, vector_store

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest) -> QueryResponse:
    """
    Retrieve top-k articles via FAISS + CrossEncoder reranking,
    then generate an answer with the chosen LLM model.
    """
    if not vector_store.index_ready():
        raise HTTPException(
            status_code=400,
            detail="No document indexed yet. Upload a PDF first.",
        )

    # ── Language detection ────────────────────────────────────────────────
    lang = request.lang
    if lang == "auto":
        lang = llm.detect_lang(request.query)

    # ── Retrieval ─────────────────────────────────────────────────────────
    try:
        results = vector_store.retrieve_rerank(
            query       = request.query,
            lang_filter = lang,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not results:
        return QueryResponse(
            answer        ="No relevant articles found for your query.",
            citations     =[],
            model_used    =request.model,
            lang_detected =lang,
        )

    # ── Build context string and citation list ────────────────────────────
    context   = "\n\n".join(doc for doc, _meta, _score in results)
    citations = [meta["article_id"] for _doc, meta, _score in results]

    # ── Generate answer ───────────────────────────────────────────────────
    try:
        answer = llm.generate_answer(
            query     = request.query,
            context   = context,
            citations = citations,
            lang      = lang,
            model_key = request.model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc

    return QueryResponse(
        answer        =answer,
        citations     =citations,
        model_used    =request.model,
        lang_detected =lang,
    )
