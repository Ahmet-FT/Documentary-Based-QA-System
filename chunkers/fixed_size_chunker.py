"""
Sabit Boyutlu Chunker (Fixed-Size Chunker)
==========================================
Metni belirli bir karakter sayısına böler.
Overlap yoktur; her parça birbirinden bağımsızdır.

Kullanım:
    chunker = FixedSizeChunker(chunk_size=500)
    chunks = chunker.chunk(documents)
"""

from typing import List
from llama_index.core import Document
from app.loaders.text_cleaner import full_clean


class FixedSizeChunker:
    """
    Metni sabit karakter uzunluğunda parçalara böler.

    Args:
        chunk_size (int): Her parçanın maksimum karakter sayısı. Varsayılan: 500.
    """

    def __init__(self, chunk_size: int = 500):
        if chunk_size <= 0:
            raise ValueError("chunk_size pozitif bir tam sayı olmalıdır.")
        self.chunk_size = chunk_size

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Document listesini alır, her belgeyi chunk_size karakterlik parçalara böler.

        Args:
            documents: LlamaIndex Document nesneleri listesi.

        Returns:
            Parçalanmış Document nesneleri listesi.
        """
        chunks: List[Document] = []

        for doc in documents:
            text = full_clean(doc.text)

            if not text:
                continue

            # Metni chunk_size'lık parçalara böl
            positions = range(0, len(text), self.chunk_size)
            total_chunks = len(list(positions))

            for chunk_idx, start in enumerate(range(0, len(text), self.chunk_size)):
                end = start + self.chunk_size
                chunk_text = text[start:end].strip()

                if not chunk_text:
                    continue

                metadata = {
                    **doc.metadata,
                    "chunk_index": chunk_idx,
                    "chunk_total": (len(text) + self.chunk_size - 1) // self.chunk_size,
                    "chunk_strategy": "fixed_size",
                    "chunk_size": self.chunk_size,
                    "char_start": start,
                    "char_end": min(end, len(text)),
                }

                chunks.append(Document(text=chunk_text, metadata=metadata))

        return chunks

    def __repr__(self) -> str:
        return f"FixedSizeChunker(chunk_size={self.chunk_size})"
