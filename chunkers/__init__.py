"""
app/chunkers/__init__.py
========================
Önerilen kullanım:
    from app.chunkers import Chunker, ChunkMode

    chunker = Chunker(mode=ChunkMode.RECURSIVE, chunk_size=512, chunk_overlap=128)
    chunks  = chunker.chunk(documents)

Tüm modlar: FIXED | OVERLAP | PARAGRAPH | SENTENCE | RECURSIVE
"""

from app.chunkers.chunker import Chunker, ChunkMode

# Geriye dönük uyumluluk
from app.chunkers.unified_chunker import UnifiedChunker
from app.chunkers.fixed_size_chunker import FixedSizeChunker
from app.chunkers.overlap_chunker import OverlapChunker
from app.chunkers.paragraph_chunker import ParagraphChunker

__all__ = [
    # Birincil arayüz
    "Chunker",
    "ChunkMode",
    # Eski arayüzler (geriye dönük uyumluluk)
    "UnifiedChunker",
    "FixedSizeChunker",
    "OverlapChunker",
    "ParagraphChunker",
]
