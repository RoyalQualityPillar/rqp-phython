from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from openai import OpenAI

from app.config import settings


router = APIRouter()


class ChatRequest(BaseModel):
    message: str


@router.get("/health", summary="Health check")
async def health_check() -> dict:
    """Simple health endpoint for liveness/readiness probes."""
    return {
        "status": "ok",
        "service": "rqp-chatbot-backend",
        "time": datetime.now(timezone.utc).isoformat(),
    }


@router.get("/live", summary="Liveness probe")
async def live() -> dict:
    return {"status": "live"}


@router.get("/ready", summary="Readiness probe")
async def ready() -> dict:
    # In real code, check DB, external services, etc. here.
    return {"status": "ready"}


@router.post("/chatbot/dummy", summary="Dummy chatbot endpoint")
async def chatbot_dummy(payload: dict) -> dict:
    """Endpoint for the chatbot UI to call.

    If an OpenAI API key is configured and a `message` field is
    present in the payload, it will return an LLM response.
    Otherwise it falls back to a fixed dummy reply.
    """

    # Default dummy reply
    reply_text = "This is a dummy response from the chatbot backend."
    openai_used = False
    openai_error: str | None = None

    if settings.openai_api_key and isinstance(payload, dict) and "message" in payload:
        try:
            client = OpenAI(api_key=settings.openai_api_key)
            completion = client.chat.completions.create(
                model=settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for the RQP chatbot UI.",
                    },
                    {"role": "user", "content": str(payload["message"])},
                ],
            )
            reply_text = completion.choices[0].message.content
            openai_used = True
        except Exception as exc:
            # On any error, keep the dummy reply so the UI still works,
            # but record the error for debugging.
            openai_error = str(exc)

    return {
        "reply": reply_text,
        "received": payload,
        "openai_configured": bool(settings.openai_api_key),
        "openai_used": openai_used,
        "openai_error": openai_error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@router.post("/chatbot/openai", summary="Chat via OpenAI LLM")
async def chatbot_openai(body: ChatRequest) -> dict:
    """Chat endpoint backed by OpenAI.

    Expects JSON: {"message": "..."} and returns {"reply": "..."}.
    """
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    try:
        client = OpenAI(api_key=settings.openai_api_key)
        completion = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant for the RQP chatbot UI."},
                {"role": "user", "content": body.message},
            ],
        )
        reply = completion.choices[0].message.content
    except Exception as exc:  # pragma: no cover - generic external error
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "reply": reply,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
