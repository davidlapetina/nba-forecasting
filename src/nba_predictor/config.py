from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "nba_predictor")
    postgres_user: str = os.getenv("POSTGRES_USER", "nba")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "nba")
    model_dir: Path = Path(os.getenv("MODEL_DIR", "./models"))
    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    classifier_model: str = os.getenv("CLASSIFIER_MODEL", "lightgbm")
    timesfm_model_version: str = os.getenv("TIMESFM_MODEL_VERSION", "timesfm-2.5-200m")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    database_url_override: str | None = os.getenv("DATABASE_URL")

    @property
    def database_url(self) -> str:
        if self.database_url_override:
            return self.database_url_override
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
