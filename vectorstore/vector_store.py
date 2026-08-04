"""
vector_store.py
===============
ChromaDB tabanlı vektör deposu.

Sorumluluklar:
    - Chunk listesini embedding'e dönüştürüp ChromaDB'ye kaydetmek
    - Koleksiyon yönetimi (oluştur / var olanı aç / sıfırla)
    - Semantik arama (embed_query → ChromaDB query → QueryResult)
    - Kaynak filtresi ile metadata bazlı arama
    - İstatistik ve durum sorgulama

Veri modeli (ChromaDB dökümanı başına):
    id          : "<file_name>_chunk_<chunk_index>"
    document    : chunk metni
    embedding   : 1024 boyutlu float vektör
    metadata    : chunk'ın tüm metadata alanları (str/int/float/bool)

Kullanım:
    from app.vectorstore import VectorStore
    from app.embeddings import EmbeddingManager

    em = EmbeddingManager()
    vs = VectorStore(embed_manager=em)

    vs.add_chunks(chunks)
    results = vs.query("Türkiye'nin başkenti neredir?", n=3)
    for r in results:
        print(r.text, r.score, r.metadata["source"])
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from llama_index.core import Document

from app.embeddings import EmbeddingManager


# ---------------------------------------------------------------------------
# Sabitler
# ---------------------------------------------------------------------------

DEFAULT_COLLECTION = "source_citation_qa"
DEFAULT_PERSIST_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "chroma_db"
)


# ---------------------------------------------------------------------------
# Veri sınıfları
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    """Tek bir arama sonucunu temsil eder."""
    text:     str
    score:    float           # Cosine benzerlik (0–1), yüksek = daha benzer
    metadata: Dict[str, Any]
    doc_id:   str


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    ChromaDB üzerinde persist edilebilir vektör deposu.

    Args:
        embed_manager   : EmbeddingManager örneği. None → otomatik oluşturulur.
        collection_name : ChromaDB koleksiyon adı.
        persist_dir     : Veri kalıcılığı için dizin yolu.
        reset           : True → varsa eski koleksiyonu siler, sıfırdan başlar.
    """

    def __init__(
        self,
        embed_manager: Optional[EmbeddingManager] = None,
        collection_name: str = DEFAULT_COLLECTION,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        reset: bool = False,
    ) -> None:
        self.embed_manager   = embed_manager or EmbeddingManager()
        self.collection_name = collection_name
        self.persist_dir     = os.path.abspath(persist_dir)

        self._client     = self._init_client()
        self._collection = self._init_collection(reset)

    # ------------------------------------------------------------------
    # Başlatma
    # ------------------------------------------------------------------

    def _init_client(self) -> chromadb.PersistentClient:
        os.makedirs(self.persist_dir, exist_ok=True)
        return chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )

    def _init_collection(self, reset: bool) -> chromadb.Collection:
        if reset:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass

        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},   # cosine benzerlik
        )

    # ------------------------------------------------------------------
    # Yazma
    # ------------------------------------------------------------------

    def add_chunks(
        self,
        chunks: List[Document],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> int:
        """
        Chunk listesini embed edip ChromaDB'ye kaydeder.

        Aynı ID'ye sahip dökümanlar güncellenir (upsert).
        Zaten mevcut olan chunk'lar embed maliyeti oluşturmaz.

        Args:
            chunks        : LlamaIndex Document nesneleri listesi.
            batch_size    : Toplu işlem boyutu.
            show_progress : İlerleme mesajı yazdır.

        Returns:
            Eklenen/güncellenen kayıt sayısı.
        """
        if not chunks:
            return 0

        total = len(chunks)
        added = 0

        for start in range(0, total, batch_size):
            batch = chunks[start: start + batch_size]

            texts     = [c.text for c in batch]
            ids       = [self._make_id(c, start + i) for i, c in enumerate(batch)]
            metadatas = [self._sanitize_metadata(c.metadata) for c in batch]

            # Embedding (passage prefix otomatik eklenir)
            embeddings = self.embed_manager.embed_passages(texts)

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )
            added += len(batch)

            if show_progress:
                print(f"  [VectorStore] {added}/{total} chunk eklendi", end="\r")

        if show_progress:
            print(f"  [VectorStore] {added}/{total} chunk eklendi ✓      ")

        return added

    # ------------------------------------------------------------------
    # Sorgulama
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        n: int = 5,
        source_filter: Optional[str] = None,
    ) -> List[QueryResult]:
        """
        Doğal dil sorusunu vektöre dönüştürüp en yakın chunk'ları getirir.

        Args:
            query_text    : Kullanıcı sorusu.
            n             : Döndürülecek sonuç sayısı.
            source_filter : Belirli bir dosyaya göre filtrele (file_name).

        Returns:
            QueryResult listesi, benzerlik skoruna göre azalan sırada.
        """
        query_vec = self.embed_manager.embed_query(query_text)

        where = {"file_name": source_filter} if source_filter else None

        raw = self._collection.query(
            query_embeddings=[query_vec],
            n_results=min(n, self._collection.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results: List[QueryResult] = []
        ids       = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw["distances"][0]

        for doc_id, text, meta, dist in zip(ids, documents, metadatas, distances):
            # ChromaDB cosine space → distance = 1 - similarity
            score = 1.0 - dist
            results.append(QueryResult(
                text=text,
                score=round(score, 4),
                metadata=meta,
                doc_id=doc_id,
            ))

        return results

    # ------------------------------------------------------------------
    # Yönetim
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Koleksiyondaki toplam chunk sayısı."""
        return self._collection.count()

    def clear(self) -> None:
        """Tüm koleksiyonu temizler (sıfırlar)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._init_collection(reset=False)

    def stats(self) -> Dict[str, Any]:
        """Koleksiyon istatistikleri."""
        return {
            "collection_name": self.collection_name,
            "persist_dir":     self.persist_dir,
            "total_chunks":    self.count(),
            "embed_model":     self.embed_manager.model_name,
            "embed_dim":       EmbeddingManager.embed_dim(),
        }

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------

    @staticmethod
    def _make_id(chunk: Document, fallback_idx: int) -> str:
        """
        Her chunk için deterministik, benzersiz ID üretir.
        Format: <temizlenmiş_dosyaadı>_chunk_<index>
        """
        file_name = chunk.metadata.get("file_name", f"doc_{fallback_idx}")
        chunk_idx = chunk.metadata.get("chunk_index", fallback_idx)
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", file_name)
        return f"{safe_name}_chunk_{chunk_idx}"

    @staticmethod
    def _sanitize_metadata(meta: dict) -> dict:
        """
        ChromaDB yalnızca str/int/float/bool metadata değerlerini kabul eder.
        Diğer tipleri string'e dönüştürür.
        """
        sanitized = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            else:
                sanitized[k] = str(v)
        return sanitized

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"VectorStore("
            f"collection='{self.collection_name}', "
            f"chunks={self.count()}, "
            f"persist='{self.persist_dir}')"
        )
