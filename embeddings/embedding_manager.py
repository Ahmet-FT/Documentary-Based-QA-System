"""
embedding_manager.py
====================
intfloat/multilingual-e5-large tabanlı embedding yöneticisi.

Türkçe ve çok dilli belgeler için önerilen model.

Özellikler:
    - Lazy loading: model ilk kullanımda yüklenir
    - Prefix yönetimi: e5 modeli query/passage prefix gerektirir
    - LlamaIndex Settings entegrasyonu
    - Toplu embed desteği (batch)

Kullanım:
    from app.embeddings import EmbeddingManager

    em = EmbeddingManager()
    em.configure_llamaindex()               # LlamaIndex global ayarı

    vecs = em.embed_passages(["Metin..."])  # Döküman vektörleri
    qvec = em.embed_query("Soru nedir?")    # Sorgu vektörü
"""

from __future__ import annotations

from typing import List

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from app.config import settings as app_settings


# ---------------------------------------------------------------------------
# Sabitler (.env'den yüklenir)
# ---------------------------------------------------------------------------

DEFAULT_MODEL   = app_settings.EMBEDDING_MODEL
QUERY_PREFIX    = "query: "
PASSAGE_PREFIX  = "passage: "
EMBED_DIM       = 1024      # multilingual-e5-large çıktı boyutu
DEFAULT_BATCH   = app_settings.EMBEDDING_BATCH_SIZE


# ---------------------------------------------------------------------------
# EmbeddingManager
# ---------------------------------------------------------------------------

class EmbeddingManager:
    """
    multilingual-e5-large embedding yöneticisi.

    Args:
        model_name  : HuggingFace model id. Varsayılan: intfloat/multilingual-e5-large
        device      : 'cpu' | 'cuda' | 'mps'. None → otomatik seçim.
        batch_size  : Toplu işlem boyutu.
        cache_dir   : Model cache dizini. None → HuggingFace varsayılanı.
    """

    def __init__(
        self,
        model_name: str  = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int  = DEFAULT_BATCH,
        cache_dir: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device     = device
        self.batch_size = batch_size
        self.cache_dir  = cache_dir
        self._embed_model: HuggingFaceEmbedding | None = None

    # ------------------------------------------------------------------
    # Lazy model yükleme
    # ------------------------------------------------------------------

    @property
    def embed_model(self) -> HuggingFaceEmbedding:
        """İlk erişimde modeli yükler (lazy loading)."""
        if self._embed_model is None:
            self._embed_model = self._load()
        return self._embed_model

    def _load(self) -> HuggingFaceEmbedding:
        kwargs: dict = {
            "model_name": self.model_name,
            "embed_batch_size": self.batch_size,
        }
        if self.device:
            kwargs["device"] = self.device
        if self.cache_dir:
            kwargs["cache_folder"] = self.cache_dir
        return HuggingFaceEmbedding(**kwargs)

    # ------------------------------------------------------------------
    # LlamaIndex entegrasyonu
    # ------------------------------------------------------------------

    def configure_llamaindex(self) -> None:
        """
        LlamaIndex global Settings'e bu embedding modelini atar.
        Pipeline başlangıcında bir kez çağırın.
        """
        Settings.embed_model = self.embed_model
        Settings.embed_batch_size = self.batch_size

    # ------------------------------------------------------------------
    # Embedding API
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> List[float]:
        """
        Tek bir sorgu metnini vektöre dönüştürür.
        E5 modeli için 'query: ' prefix'i otomatik eklenir.

        Args:
            query: Kullanıcı sorusu / arama metni.

        Returns:
            1024 boyutlu float listesi.
        """
        prefixed = f"{QUERY_PREFIX}{query}"
        return self.embed_model.get_text_embedding(prefixed)

    def embed_passage(self, text: str) -> List[float]:
        """
        Tek bir döküman parçasını vektöre dönüştürür.
        E5 modeli için 'passage: ' prefix'i otomatik eklenir.
        """
        prefixed = f"{PASSAGE_PREFIX}{text}"
        return self.embed_model.get_text_embedding(prefixed)

    def embed_passages(self, texts: List[str]) -> List[List[float]]:
        """
        Döküman parçaları listesini toplu olarak vektöre dönüştürür.

        Args:
            texts: Döküman metinleri listesi.

        Returns:
            Her metin için 1024 boyutlu vektör listesi.
        """
        prefixed = [f"{PASSAGE_PREFIX}{t}" for t in texts]
        return self.embed_model.get_text_embedding_batch(
            prefixed,
            show_progress=True,
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        loaded = "yüklendi" if self._embed_model is not None else "lazy"
        return (
            f"EmbeddingManager("
            f"model='{self.model_name}', "
            f"dim={EMBED_DIM}, "
            f"batch={self.batch_size}, "
            f"status={loaded})"
        )

    @staticmethod
    def embed_dim() -> int:
        """Model çıktı vektör boyutu."""
        return EMBED_DIM
