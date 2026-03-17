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

    # OpenAI configuration
    openai_api_key: str | None = None
    # Use a commonly available model by default
    openai_model: str = "gpt-4o-mini"

    # Example macro-style values (customize as needed)
    # api_timeout_seconds: int = 30
    # some_macro_flag: bool = False

    class Config:
        # We already load .env explicitly above, but keep this for clarity.
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
