"""
LLM provider abstraction.
Set LLM_PROVIDER=gemini  (default) to use Google Gemini (free tier).
Set LLM_PROVIDER=openai          to use OpenAI GPT-4.
"""
from fastapi import HTTPException

from app.config import settings
from app.logger import get_logger

log = get_logger("llm")


def _require(key: str | None, name: str) -> str:
    if not key:
        log.error("%s API key is not configured", name)
        raise HTTPException(status_code=500, detail=f"{name} API key not configured")
    return key


# ── embeddings ────────────────────────────────────────────────────────────────

def embed(texts: list[str]) -> list[list[float]]:
    log.info("Embedding %d text(s) using provider=%s", len(texts), settings.llm_provider)
    try:
        if settings.llm_provider == "openai":
            from openai import OpenAI
            key = _require(settings.openai_api_key, "OpenAI")
            resp = OpenAI(api_key=key).embeddings.create(
                model=settings.openai_embedding_model, input=texts
            )
            result = [item.embedding for item in resp.data]
        else:
            from google import genai
            key = _require(settings.gemini_api_key, "Gemini")
            client = genai.Client(api_key=key)
            result = [
                client.models.embed_content(
                    model=settings.gemini_embedding_model,
                    contents=t,
                ).embeddings[0].values
                for t in texts
            ]
        log.info("Embedding complete — %d vector(s) returned", len(result))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Embedding failed (provider=%s): %s", settings.llm_provider, exc, exc_info=True)
        raise


# ── chat ──────────────────────────────────────────────────────────────────────

def chat(system: str, user: str) -> str:
    log.info("Chat request — provider=%s | user_msg_len=%d", settings.llm_provider, len(user))
    try:
        if settings.llm_provider == "openai":
            from openai import OpenAI
            key = _require(settings.openai_api_key, "OpenAI")
            completion = OpenAI(api_key=key).chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            reply = completion.choices[0].message.content
        else:
            from google import genai
            from google.genai import types
            key = _require(settings.gemini_api_key, "Gemini")
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model=settings.gemini_model,
                contents=user,
                config=types.GenerateContentConfig(system_instruction=system),
            )
            reply = response.text

        log.info("Chat response received — reply_len=%d", len(reply))
        return reply
    except HTTPException:
        raise
    except Exception as exc:
        log.error("Chat failed (provider=%s): %s", settings.llm_provider, exc, exc_info=True)
        raise
