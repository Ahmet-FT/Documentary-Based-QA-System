"""
app.vectorstore
===============
ChromaDB tabanlı vektör depolama ve sorgulama.

Kullanım:
    from app.vectorstore import VectorStore

    vs = VectorStore()
    vs.add_chunks(chunks)                  # chunk listesi ekle
    results = vs.query("sorum", n=5)       # semantik arama
"""

from app.vectorstore.vector_store import VectorStore

__all__ = ["VectorStore"]
