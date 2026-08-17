"""
qa_engine.py
============
Kaynak göstermeli soru-cevap motoru.

Retriever'dan gelen chunk'ları prompt'a yerleştirir ve Ollama LLM'den
grounded (kaynak tabanlı) cevap üretir.

Akış:
    Soru → Retriever (Top-K chunk) → Prompt Builder → LLM → Cevap + Kaynaklar

Kullanım:
    from app.qa_engine import QAEngine

    qa = QAEngine()
    result = qa.ask("Yapay zekanın etik sorunları nelerdir?")
    print(result.answer)
    print(result.sources)

Veya pipeline ile:
    from app.ingestion_pipeline import IngestionPipeline
    from app.qa_engine import QAEngine

    pipeline = IngestionPipeline()
    pipeline.ingest("rapor.pdf")

    qa = QAEngine(retriever=pipeline.retriever)
    result = qa.ask("Raporda ne anlatılıyor?")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.llm import LLMManager
from app.llm.prompt_templates import build_qa_prompt, get_system_prompt
from app.retriever import Retriever, RetrievalResult


# ---------------------------------------------------------------------------
# Veri sınıfı
# ---------------------------------------------------------------------------

@dataclass
class QAResult:
    """
    Soru-cevap sonucunu temsil eder.

    Attributes:
        query           : Kullanıcı sorusu.
        answer          : LLM'in ürettiği cevap.
        sources         : Kullanılan kaynakların özet listesi.
        retrieval_results : Ham retrieval sonuçları.
        prompt_tokens   : Prompt token sayısı.
        response_tokens : Üretilen token sayısı.
        duration_sec    : Toplam süre (saniye).
        llm_model       : Kullanılan LLM model adı.
    """
    query:             str
    answer:            str
    sources:           List[Dict[str, Any]]
    retrieval_results: List[RetrievalResult]
    prompt_tokens:     int   = 0
    response_tokens:   int   = 0
    duration_sec:      float = 0.0
    llm_model:         str   = ""

    @property
    def has_answer(self) -> bool:
        """Cevap bulundu mu (yoksa 'bulunamadı' mı)."""
        not_found_markers = [
            "bulunamamıştır",
            "bulunamadı",
            "yer almamaktadır",
            "mevcut değildir",
            "bilgi bulunmamaktadır",
        ]
        lower = self.answer.lower()
        return not any(m in lower for m in not_found_markers)

    def to_dict(self) -> Dict[str, Any]:
        """Sonucu sözlük olarak döndürür."""
        return {
            "query":           self.query,
            "answer":          self.answer,
            "has_answer":      self.has_answer,
            "sources":         self.sources,
            "prompt_tokens":   self.prompt_tokens,
            "response_tokens": self.response_tokens,
            "duration_sec":    round(self.duration_sec, 2),
            "llm_model":       self.llm_model,
        }

    def __str__(self) -> str:
        return (
            f"Soru: {self.query}\n"
            f"Cevap: {self.answer}\n"
            f"Kaynaklar: {len(self.sources)} adet\n"
            f"Süre: {self.duration_sec:.2f}s"
        )


# ---------------------------------------------------------------------------
# QAEngine
# ---------------------------------------------------------------------------

class QAEngine:
    """
    Kaynak göstermeli soru-cevap motoru.

    Retriever + LLM'i birleştirerek grounded cevap üretir.

    Args:
        retriever       : Retriever örneği. None → varsayılan oluşturulur.
        llm_manager     : LLMManager örneği. None → varsayılan (llama3.1:8b).
        default_top_k   : Retrieval'da varsayılan sonuç sayısı.
        min_score       : Retrieval için minimum benzerlik eşiği.
        max_context_chunks : LLM'e gönderilecek maksimum chunk sayısı.
    """

    def __init__(
        self,
        retriever:          Optional[Retriever]  = None,
        llm_manager:        Optional[LLMManager] = None,
        default_top_k:      int   = 5,
        min_score:          float = 0.0,
        max_context_chunks: int   = 5,
    ) -> None:
        self.retriever          = retriever or Retriever()
        self.llm                = llm_manager or LLMManager()
        self.default_top_k      = default_top_k
        self.min_score          = min_score
        self.max_context_chunks = max_context_chunks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        query: str,
        top_k: Optional[int]   = None,
        min_score: Optional[float] = None,
        source_filter: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> QAResult:
        """
        Kullanıcı sorusunu cevaplar.

        Akış:
            1. Retriever ile en ilgili chunk'ları getir
            2. Chunk'ları prompt'a yerleştir
            3. Ollama LLM'den cevap üret
            4. Sonuçları yapılandır

        Args:
            query         : Kullanıcı sorusu.
            top_k         : Kaç chunk getirilsin (None → default_top_k).
            min_score     : Min. benzerlik eşiği (None → self.min_score).
            source_filter : Belirli bir dosyaya filtrele.
            temperature   : LLM sıcaklığı geçersiz kılma.

        Returns:
            QAResult — Cevap, kaynaklar ve metadata.

        Raises:
            ConnectionError : Ollama erişilemezse.
        """
        t0 = time.perf_counter()

        k = top_k if top_k is not None else self.default_top_k
        score_threshold = min_score if min_score is not None else self.min_score

        # 1. Retrieve
        retrieval_results = self.retriever.retrieve(
            query=query,
            top_k=k,
            min_score=score_threshold,
            source_filter=source_filter,
        )

        # 2. Context hazırlama — sonuçları en fazla max_context_chunks ile sınırla
        limited_results = retrieval_results[:self.max_context_chunks]

        contexts = [
            {
                "text":   r.text,
                "source": r.source,
                "page":   r.page,
                "score":  r.score,
            }
            for r in limited_results
        ]

        # 3. Prompt oluştur
        system_prompt = get_system_prompt()
        user_prompt   = build_qa_prompt(query=query, contexts=contexts)

        # 4. LLM'den cevap al
        llm_response = self.llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
        )

        elapsed = time.perf_counter() - t0

        # 5. Kaynak listesi
        sources = [
            {
                "kaynak_no": i + 1,
                "dosya":     r.source,
                "sayfa":     r.page,
                "skor":      round(r.score, 4),
                "chunk_idx": r.chunk_index,
            }
            for i, r in enumerate(limited_results)
        ]

        return QAResult(
            query=query,
            answer=llm_response.text,
            sources=sources,
            retrieval_results=retrieval_results,
            prompt_tokens=llm_response.prompt_tokens,
            response_tokens=llm_response.response_tokens,
            duration_sec=elapsed,
            llm_model=llm_response.model,
        )

    def health_check(self) -> Dict[str, Any]:
        """
        Tüm bileşenlerin durumunu kontrol eder.

        Returns:
            {
                "ollama_server": bool,
                "ollama_model": bool,
                "model_name": str,
                "vectorstore_chunks": int,
                "error": str | None,
            }
        """
        llm_health = self.llm.health_check()

        try:
            vs_stats = self.retriever.stats()
            chunk_count = vs_stats.get("total_chunks", 0)
        except Exception:
            chunk_count = -1

        return {
            "ollama_server":      llm_health["server"],
            "ollama_model":       llm_health["model"],
            "model_name":         llm_health["model_name"],
            "vectorstore_chunks": chunk_count,
            "error":              llm_health.get("error"),
        }

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"QAEngine("
            f"llm={self.llm.model}, "
            f"top_k={self.default_top_k}, "
            f"retriever={self.retriever!r})"
        )
