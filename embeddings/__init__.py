"""
app.embeddings
==============
Türkçe / çok dilli embedding desteği.
Varsayılan model: intfloat/multilingual-e5-large
"""

from app.embeddings.embedding_manager import EmbeddingManager

__all__ = ["EmbeddingManager"]
