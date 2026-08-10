"""
qa_engine package
=================
Retriever + LLM entegrasyonu ile kaynak göstermeli soru-cevap motoru.

Dışa aktarılan sınıflar:
    QAEngine  — Soru → Retrieve → LLM → Cevap + Kaynaklar
    QAResult  — Cevap veri sınıfı
"""

from app.qa_engine.qa_engine import QAEngine, QAResult

__all__ = ["QAEngine", "QAResult"]
