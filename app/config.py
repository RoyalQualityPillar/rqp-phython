from typing import List

from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings


# Ensure .env in the project root is loaded
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


class Settings(BaseSettings):
    """Application settings / macro values.

    All values can be overridden using environment variables
    or a .env file in the project root.
    """

    # General
    app_name: str = "RQP Chatbot Backend"
    app_version: str = "0.1.0"
    debug: bool = True

    # CORS
    cors_origins: List[str] = [
        "http://localhost:4200",
        "http://localhost:4210",
        "http://localhost:4300",
        "http://127.0.0.1:4200",
        "http://127.0.0.1:4210",
        "http://127.0.0.1:4300",
    ]

    # LLM provider: "gemini" or "openai"
    llm_provider: str = "gemini"

    # OpenAI (GPT-4 — paid)
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dim: int = 1536

    # Google Gemini (free tier)
    gemini_api_key: str | None = None
    gemini_model: str = "models/gemini-2.5-flash"
    gemini_embedding_model: str = "models/gemini-embedding-001"
    gemini_embedding_dim: int = 3072

    # PostgreSQL
    db_host: str = "localhost"
    db_port: int = 5432
    db_user: str = "postgres"
    db_password: str = "myWizard@123"
    db_name: str = "rqp_chat"
    db_schema: str = "rqp"

    class Config:
        # We already load .env explicitly above, but keep this for clarity.
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
