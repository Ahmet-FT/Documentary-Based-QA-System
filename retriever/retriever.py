"""
retriever.py
============
Vektör veritabanından semantik benzerlik araması yapar.

Bu modül VectorStore üzerine oturan bir katmandır:
    - Top-k sonuç getirme
    - Benzerlik eşiği filtreleme
    - Kaynak (dosya) bazlı filtreleme
    - Sonuçları zengin metadata ile döndürme
    - Sonuç özetleri ve raporlama

Kullanım (basit):
    from app.retriever import Retriever

    retriever = Retriever()
    results = retriever.retrieve("Yapay zekanın önemi nedir?", top_k=5)
    for r in results:
        print(r)

Kullanım (pipeline ile):
    from app.ingestion_pipeline import IngestionPipeline

    pipeline = IngestionPipeline()
    pipeline.ingest("Documents/rapor.pdf")

    retriever = Retriever(vector_store=pipeline.vector_store)
    results   = retriever.retrieve("Soru metni", top_k=3)
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from app.embeddings import EmbeddingManager
from app.vectorstore import VectorStore
from app.vectorstore.vector_store import QueryResult


# ---------------------------------------------------------------------------
# Veri sınıfı
# ---------------------------------------------------------------------------

@dataclass
class RetrievalResult:
    """
    Tek bir retrieval sonucunu temsil eder.

    Attributes:
        rank        : Sıralama (1'den başlar)
        text        : Chunk metni
        score       : Cosine benzerlik skoru (0-1 arası, yüksek = daha benzer)
        metadata    : Chunk'a ait tüm metadata alanları
        doc_id      : ChromaDB'deki benzersiz ID
        source      : Kaynak dosya adı (metadata'dan kolayca erişim)
        page        : Sayfa numarası (varsa)
        chunk_index : Chunk sıra numarası (varsa)
    """
    rank:        int
    text:        str
    score:       float
    metadata:    Dict[str, Any]
    doc_id:      str
    source:      str      = field(init=False)
    page:        Any      = field(init=False)
    chunk_index: int      = field(init=False)

    def __post_init__(self):
        self.source      = self.metadata.get("file_name", "bilinmiyor")
        self.page        = self.metadata.get(
            "page_number",
            self.metadata.get("page_label", self.metadata.get("page", "?"))
        )
        self.chunk_index = int(self.metadata.get("chunk_index", -1))

    # ------------------------------------------------------------------
    # Görüntüleme
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        preview = textwrap.shorten(self.text, width=120, placeholder="...")
        return (
            f"[#{self.rank}] Skor: {self.score:.4f} | "
            f"Kaynak: {self.source} | Sayfa: {self.page} | "
            f"Chunk: {self.chunk_index}\n"
            f"  → {preview}"
        )

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(rank={self.rank}, score={self.score:.4f}, "
            f"source='{self.source}', chunk_index={self.chunk_index})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Sonucu sözlük olarak döndürür (JSON/API çıktısı için)."""
        return {
            "rank":        self.rank,
            "score":       self.score,
            "text":        self.text,
            "source":      self.source,
            "page":        self.page,
            "chunk_index": self.chunk_index,
            "doc_id":      self.doc_id,
            "metadata":    self.metadata,
        }


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class Retriever:
    """
    Semantik benzerlik tabanlı doküman parça getirici.

    VectorStore üzerine oturan bir retrieval katmanı sağlar.
    Pipeline'dan bağımsız kullanılabildiği gibi IngestionPipeline
    ile entegre de çalışır.

    Args:
        vector_store    : VectorStore örneği. None → varsayılan ChromaDB.
        embed_manager   : EmbeddingManager. None → VectorStore'unkini kullanır.
        default_top_k   : retrieve() çağrısında varsayılan sonuç sayısı.
        min_score       : Bu eşiğin altındaki sonuçlar filtrelenir (0.0 = filtre yok).
    """

    def __init__(
        self,
        vector_store:  Optional[VectorStore]    = None,
        embed_manager: Optional[EmbeddingManager] = None,
        default_top_k: int   = 5,
        min_score:     float = 0.0,
    ) -> None:
        self.embed_manager = embed_manager or EmbeddingManager()
        self.vector_store  = vector_store or VectorStore(
            embed_manager=self.embed_manager
        )
        self.default_top_k = default_top_k
        self.min_score     = min_score

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query:         str,
        top_k:         Optional[int]  = None,
        min_score:     Optional[float] = None,
        source_filter: Optional[Union[str, List[str]]]  = None,
    ) -> List[RetrievalResult]:
        """
        Sorguya en yakın top-k chunk'ı getirir.

        Args:
            query         : Kullanıcı sorusu veya arama metni.
            top_k         : Kaç sonuç isteniyor (None → default_top_k).
            min_score     : Benzerlik eşiği (None → self.min_score).
            source_filter : Dosya adı (str) veya dosya adları listesi (list).

        Returns:
            RetrievalResult listesi, benzerlik skoruna göre azalan sırada.
        """
        k     = top_k     if top_k     is not None else self.default_top_k
        score = min_score if min_score is not None else self.min_score

        raw_results: List[QueryResult] = self.vector_store.query(
            query_text=query,
            n=k,
            source_filter=source_filter,
        )

        results: List[RetrievalResult] = []
        for rank, qr in enumerate(raw_results, start=1):
            if qr.score < score:
                continue
            results.append(RetrievalResult(
                rank=rank,
                text=qr.text,
                score=qr.score,
                metadata=qr.metadata,
                doc_id=qr.doc_id,
            ))

        return results

    def retrieve_with_report(
        self,
        query:         str,
        top_k:         Optional[int]  = None,
        min_score:     Optional[float] = None,
        source_filter: Optional[Union[str, List[str]]]  = None,
        print_output:  bool = True,
    ) -> List[RetrievalResult]:
        """
        retrieve() ile aynıdır; ek olarak konsola biçimlendirilmiş rapor yazdırır.

        Returns:
            RetrievalResult listesi.
        """
        results = self.retrieve(
            query=query,
            top_k=top_k,
            min_score=min_score,
            source_filter=source_filter,
        )

        if print_output:
            self._print_report(query, results)

        return results

    # ------------------------------------------------------------------
    # Yardımcı
    # ------------------------------------------------------------------

    def _print_report(self, query: str, results: List[RetrievalResult]) -> None:
        """Konsola biçimlendirilmiş arama raporu yazdırır."""
        sep = "─" * 60
        print(f"\n{'='*60}")
        print(f"  🔍 Sorgu : {query}")
        print(f"  📊 Sonuç : {len(results)} chunk bulundu")
        print(f"{'='*60}")

        if not results:
            print("  ❌ Sonuç bulunamadı.")
            return

        for r in results:
            print(f"\n{sep}")
            print(f"  Sıra      : #{r.rank}")
            print(f"  Benzerlik : {r.score:.4f}  ({r.score*100:.1f}%)")
            print(f"  Kaynak    : {r.source}")
            print(f"  Sayfa     : {r.page}")
            print(f"  Chunk     : #{r.chunk_index}")
            print(f"  Metin     :")
            wrapped = textwrap.fill(r.text, width=70, initial_indent="    ", subsequent_indent="    ")
            print(wrapped)

        print(f"\n{sep}")
        avg_score = sum(r.score for r in results) / len(results)
        print(f"  Ort. Benzerlik : {avg_score:.4f}  ({avg_score*100:.1f}%)")
        print(f"  En Yüksek     : {results[0].score:.4f}")
        print(f"  En Düşük      : {results[-1].score:.4f}")
        print(f"{'='*60}\n")

    def stats(self) -> Dict[str, Any]:
        """VectorStore istatistiklerini döndürür."""
        return self.vector_store.stats()

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Retriever("
            f"top_k={self.default_top_k}, "
            f"min_score={self.min_score}, "
            f"store={self.vector_store!r})"
        )
