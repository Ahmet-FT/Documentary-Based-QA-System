"""
ingestion_pipeline.py
=====================
Doküman yükleme → Chunking → Embedding → ChromaDB kayıt
adımlarını tek bir pipeline altında birleştirir.

Kullanım:
    from app.ingestion_pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    stats = pipeline.ingest("Documents/rapor.pdf")
    print(stats)
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from llama_index.core import Document

from app.chunkers.chunker import Chunker, ChunkMode
from app.embeddings import EmbeddingManager
from app.loaders import DocumentLoader
from app.retriever import Retriever
from app.vectorstore import VectorStore


class IngestionPipeline:
    """
    Uçtan uca doküman ingestion pipeline'ı.

    Aşamalar:
        1. Load   → DocumentLoader ile dokümanı yükle
        2. Chunk  → Chunker ile parçala
        3. Embed  → EmbeddingManager ile vektörleştir
        4. Store  → VectorStore (ChromaDB) ile kaydet

    Args:
        chunk_mode      : Parçalama stratejisi. Varsayılan: RECURSIVE
        chunk_size      : Token sayısı (RECURSIVE) veya karakter sayısı.
        chunk_overlap   : Örtüşme boyutu.
        embed_manager   : EmbeddingManager. None → varsayılan model.
        vector_store    : VectorStore. None → varsayılan ChromaDB.
    """

    def __init__(
        self,
        chunk_mode: ChunkMode | str = ChunkMode.RECURSIVE,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        embed_manager: Optional[EmbeddingManager] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        self.loader        = DocumentLoader()
        self.chunker       = Chunker(
            mode=chunk_mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.embed_manager = embed_manager or EmbeddingManager()
        self.vector_store  = vector_store or VectorStore(
            embed_manager=self.embed_manager
        )
        self._retriever: Optional[Retriever] = None

    @property
    def retriever(self) -> Retriever:
        """Pipeline'ın VectorStore'una bağlı Retriever (lazy)."""
        if self._retriever is None:
            self._retriever = Retriever(
                vector_store=self.vector_store,
                embed_manager=self.embed_manager,
            )
        return self._retriever

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest(self, file_path: str, show_progress: bool = True) -> Dict[str, Any]:
        """
        Tek bir dosyayı pipeline'dan geçirir.

        Returns:
            İstatistik sözlüğü: file, pages, chunks, elapsed_sec
        """
        t0 = time.perf_counter()

        if show_progress:
            print(f"\n{'='*55}")
            print(f"  Ingestion: {file_path}")
            print(f"{'='*55}")

        # 1. Yükle
        if show_progress:
            print("  [1/3] Doküman yükleniyor...")
        docs = self.loader.load(file_path)
        if show_progress:
            print(f"        {len(docs)} sayfa/bölüm yüklendi")

        # 2. Parçala
        if show_progress:
            print(f"  [2/3] Chunking ({self.chunker.mode.value})...")
        chunks = self.chunker.chunk(docs)
        if show_progress:
            print(f"        {len(chunks)} chunk üretildi")

        # 3. Embed + Kaydet
        if show_progress:
            print("  [3/3] Embedding + ChromaDB'ye kayıt...")
        added = self.vector_store.add_chunks(chunks, show_progress=show_progress)

        elapsed = round(time.perf_counter() - t0, 2)

        stats = {
            "file":        file_path,
            "pages":       len(docs),
            "chunks":      len(chunks),
            "stored":      added,
            "elapsed_sec": elapsed,
        }

        if show_progress:
            print(f"\n  ✅ Tamamlandı — {len(chunks)} chunk, {elapsed}s")

        return stats

    def ingest_many(
        self, file_paths: List[str], show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Birden fazla dosyayı sırayla ingest eder."""
        return [self.ingest(fp, show_progress) for fp in file_paths]

    def search(self, query: str, n: int = 5) -> list:
        """Pipeline'ın VectorStore'una kısa yol sorgusu."""
        return self.vector_store.query(query, n=n)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"IngestionPipeline("
            f"chunker={self.chunker!r}, "
            f"store={self.vector_store!r})"
        )
