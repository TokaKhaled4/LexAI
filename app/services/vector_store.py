"""
app/services/vector_store.py
Singleton that holds the FAISS index, embeddings, and reranker.
Call add_to_index() after each upload — articles are APPENDED, not replaced.
Call reset_index() to wipe everything and start fresh.
"""

from __future__ import annotations
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

# ── Model names ───────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
RERANKER_MODEL   = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# ── Lazy-loaded singletons ────────────────────────────────────────────────────
_embed_model: SentenceTransformer | None = None
_reranker:    CrossEncoder        | None = None

# ── Index state (grows with each upload) ─────────────────────────────────────
_index:     faiss.IndexFlatIP | None = None
_documents: list[str]                = []
_metadatas: list[dict]               = []


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


# ── Public API ────────────────────────────────────────────────────────────────

def add_to_index(articles: list[dict]) -> None:
    """
    Encode articles and APPEND them to the existing FAISS index.
    Safe to call multiple times (once per uploaded PDF).
    Duplicate article_ids are skipped automatically.
    """
    global _index, _documents, _metadatas

    if not articles:
        return

    embed = _get_embed_model()

    # ── Deduplicate: skip article_ids already in the index ───────────────
    existing_ids = {m["article_id"] for m in _metadatas}
    new_articles = [a for a in articles if a["article_id"] not in existing_ids]

    if not new_articles:
        return

    new_docs  = [a["text"] for a in new_articles]
    new_metas = [
        {
            "article_id" : a["article_id"],
            "article_num": a["article_num"],
            "lang"       : a["lang"],
        }
        for a in new_articles
    ]

    embeddings = embed.encode(
        new_docs,
        show_progress_bar=True,
        batch_size=32,
        convert_to_numpy=True,
    )

    safe = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(safe)

    # ── Create index on first call, otherwise append ──────────────────────
    if _index is None:
        _index = faiss.IndexFlatIP(safe.shape[1])

    _index.add(safe)
    _documents.extend(new_docs)
    _metadatas.extend(new_metas)


def reset_index() -> None:
    """Wipe the index so a fresh session can start."""
    global _index, _documents, _metadatas
    _index     = None
    _documents = []
    _metadatas = []


def index_stats() -> dict:
    """Return counts per language currently in the index."""
    en = sum(1 for m in _metadatas if m.get("lang") == "en")
    ar = sum(1 for m in _metadatas if m.get("lang") == "ar")
    return {"total": len(_metadatas), "en": en, "ar": ar}


def retrieve_rerank(
    query: str,
    top_k: int = 10,
    final_k: int = 5,
    lang_filter: str | None = None,
) -> list[tuple[str, dict, float]]:
    """
    Returns up to *final_k* (doc_text, metadata, score) tuples,
    reranked by CrossEncoder.
    """
    if _index is None or not _documents:
        raise RuntimeError("Index is empty — upload a document first.")

    embed   = _get_embed_model()
    reranker = _get_reranker()

    q_vec = embed.encode([query], convert_to_numpy=True)
    q_vec = np.array(q_vec, dtype=np.float32)
    faiss.normalize_L2(q_vec)

    search_k = top_k * 5 if lang_filter else top_k
    _, indices = _index.search(q_vec, search_k)

    candidate_docs:  list[str]  = []
    candidate_metas: list[dict] = []

    for idx in indices[0]:
        if idx == -1:
            continue
        meta = _metadatas[idx]
        if lang_filter and meta.get("lang") != lang_filter:
            continue
        candidate_docs.append(_documents[idx])
        candidate_metas.append(meta)
        if len(candidate_docs) == top_k:
            break

    if not candidate_docs:
        return []

    scores = reranker.predict([(query, doc) for doc in candidate_docs])
    ranked = sorted(
        zip(candidate_docs, candidate_metas, scores),
        key=lambda x: x[2],
        reverse=True,
    )
    return ranked[:final_k]


def index_ready() -> bool:
    return _index is not None and _index.ntotal > 0