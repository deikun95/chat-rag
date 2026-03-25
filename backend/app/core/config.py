from pydantic_settings import BaseSettings
from pathlib import Path

# Resolve .env path relative to this file, not CWD
_ENV_FILE = Path(__file__).resolve().parent.parent.parent.parent / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API keys
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # App
    app_env: str = "development"
    max_upload_size_mb: int = 20
    allowed_origins: str = "http://localhost:3000"

    # ChromaDB
    chroma_persist_dir: str = "./chroma_data"

    # Rate limiting
    rate_limit_rpm: int = 20

    # Computed
    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",")]

    model_config = {
        "env_file": str(_ENV_FILE),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton — import this everywhere
settings = Settings()
