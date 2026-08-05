"""
retriever package
=================
Vektör veritabanından benzerlik araması yapan modüller.

Dışa aktarılan sınıflar:
    Retriever       — Temel top-k benzerlik araması
    RetrievalResult — Arama sonucu veri sınıfı
"""

from app.retriever.retriever import Retriever, RetrievalResult

__all__ = ["Retriever", "RetrievalResult"]
