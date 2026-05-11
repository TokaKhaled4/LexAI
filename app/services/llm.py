"""
app/services/llm.py
Groq-backed LLM service.  Supports "qwen" and "llama" model keys.
"""

from __future__ import annotations
import os
import re

from groq import Groq

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_CONFIGS: dict[str, dict] = {
    "qwen": {
        "name"       : "qwen/qwen3-32b",
        "description": "Strong multilingual (Arabic + English), best for RAG reasoning",
    },
    "llama": {
        "name"       : "llama-3.3-70b-versatile",
        "description": "Best reasoning + summarisation, very strong general model",
    },
}

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = "your_api_key"
        _client = Groq(api_key=api_key)
    return _client


# ── Prompt builders ───────────────────────────────────────────────────────────

def _build_prompt(
    query: str,
    context: str,
    lang: str,
    model_key: str,
) -> list[dict]:
    if model_key == "llama":
        if lang == "ar":
            system = (
                "أنت مساعد قانوني دقيق. "
                "استخدم فقط السياق المقدم لك. "
                "لا تضف أي معلومات خارج السياق. "
                "أجب في فقرة واحدة قصيرة فقط مع ذكر أرقام المواد."
            )
            user = f"السياق:\n{context}\n\nالسؤال: {query}\n\nأجب باختصار وبالاعتماد فقط على السياق:"
        else:
            system = (
                "You are a strict legal assistant. "
                "Use ONLY the provided context. "
                "Do NOT hallucinate or add external knowledge. "
                "Answer in ONE short paragraph and cite article numbers."
            )
            user = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer strictly using the context:"
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]

    # ── Qwen ──
    no_think = "\n/no_think\n"
    if lang == "ar":
        system = (
            "أنت مساعد قانوني. القواعد الصارمة:\n"
            "1. استخدم السياق المقدم فقط.\n"
            "2. لا يُسمح باستخدام المعرفة الخارجية.\n"
            "3. فقرة واحدة فقط.\n"
            "4. اذكر أرقام المواد داخل النص.\n"
            "5. لا تتجاوز 100 كلمة."
        )
        user = f"السياق:\n{context}\n\nالسؤال: {query}{no_think}أجب باختصار وبالاعتماد فقط على السياق:"
    else:
        system = (
            "You are a legal assistant. Strict rules:\n"
            "1. Use ONLY the given context.\n"
            "2. No external knowledge allowed.\n"
            "3. One paragraph only.\n"
            "4. Cite article numbers inline.\n"
            "5. Max 100 words."
        )
        user = f"Context:\n{context}\n\nQuestion: {query}{no_think}Answer strictly based on context:"

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


# ── Public API ────────────────────────────────────────────────────────────────

def detect_lang(query: str) -> str:
    """Returns 'ar' if >30 % of query chars are Arabic, else 'en'."""
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', query))
    return "ar" if arabic_chars > len(query) * 0.3 else "en"


def generate_answer(
    query: str,
    context: str,
    citations: list[str],
    lang: str,
    model_key: str = "qwen",
) -> str:
    """
    Call the Groq API and return a cleaned answer string.
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model_key '{model_key}'. Choose from {list(MODEL_CONFIGS)}")

    client     = _get_client()
    model_name = MODEL_CONFIGS[model_key]["name"]
    messages   = _build_prompt(query, context, lang, model_key)

    response = client.chat.completions.create(
        model      = model_name,
        messages   = messages,
        temperature= 0.1,
        max_tokens = 300,
        top_p      = 0.9,
    )

    answer = response.choices[0].message.content or ""

    # Strip any leftover <think>…</think> blocks (Qwen reasoning traces)
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
    answer = answer.split("\n\n")[0].strip()
    return answer
