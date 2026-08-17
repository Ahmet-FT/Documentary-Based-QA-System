"""
config.py
=========
Merkezi yapılandırma — .env dosyasından tüm ayarları yükler.

Tüm modüller bu dosyadan ayarlarını alır:
    from app.config import settings

    settings.OLLAMA_MODEL       → "llama3.1:8b"
    settings.EMBEDDING_MODEL    → "intfloat/multilingual-e5-large"
    ...
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# .env dosyasını yükle (proje kökünden)
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=os.path.abspath(_env_path))


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int = 0) -> int:
    val = os.getenv(key)
    return int(val) if val else default


def _env_float(key: str, default: float = 0.0) -> float:
    val = os.getenv(key)
    return float(val) if val else default


@dataclass(frozen=True)
class Settings:
    """Tüm uygulama ayarlarını tutar."""

    # LLM (Ollama)
    OLLAMA_MODEL: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "llama3.1:8b"))
    OLLAMA_BASE_URL: str = field(default_factory=lambda: _env("OLLAMA_BASE_URL", "http://localhost:11434"))
    OLLAMA_TIMEOUT: int = field(default_factory=lambda: _env_int("OLLAMA_TIMEOUT", 120))
    OLLAMA_TEMPERATURE: float = field(default_factory=lambda: _env_float("OLLAMA_TEMPERATURE", 0.1))
    OLLAMA_NUM_CTX: int = field(default_factory=lambda: _env_int("OLLAMA_NUM_CTX", 4096))
    OLLAMA_TOP_P: float = field(default_factory=lambda: _env_float("OLLAMA_TOP_P", 0.9))
    OLLAMA_REPEAT_PENALTY: float = field(default_factory=lambda: _env_float("OLLAMA_REPEAT_PENALTY", 1.1))
    OLLAMA_NUM_GPU: int = field(default_factory=lambda: _env_int("OLLAMA_NUM_GPU", -1))

    # Embedding
    EMBEDDING_MODEL: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "intfloat/multilingual-e5-large"))
    EMBEDDING_DEVICE: str = field(default_factory=lambda: _env("EMBEDDING_DEVICE", ""))
    EMBEDDING_BATCH_SIZE: int = field(default_factory=lambda: _env_int("EMBEDDING_BATCH_SIZE", 32))

    # VectorStore
    CHROMA_COLLECTION: str = field(default_factory=lambda: _env("CHROMA_COLLECTION", "source_citation_qa"))
    CHROMA_PERSIST_DIR: str = field(default_factory=lambda: _env("CHROMA_PERSIST_DIR", "./chroma_db"))

    # Sunucu
    SERVER_HOST: str = field(default_factory=lambda: _env("SERVER_HOST", "0.0.0.0"))
    SERVER_PORT: int = field(default_factory=lambda: _env_int("SERVER_PORT", 8000))

    # API Limitleri
    MAX_FILE_SIZE_MB: int = field(default_factory=lambda: _env_int("MAX_FILE_SIZE_MB", 50))
    MAX_TEXT_LENGTH: int = field(default_factory=lambda: _env_int("MAX_TEXT_LENGTH", 500_000))
    MAX_QUERY_LENGTH: int = field(default_factory=lambda: _env_int("MAX_QUERY_LENGTH", 2000))


# Singleton — tüm modüller bunu import eder
settings = Settings()
