"""
Overlap (Kayan Pencere) Chunker
================================
Metni chunk_size karakterlik parçalara böler,
ancak ardışık parçalar arasında overlap_size kadar
karakter örtüşmesi bırakır.

Bu yöntem, bir parçanın sonundaki bağlamın
bir sonraki parçanın başına taşınmasını sağlar —
böylece cümle ortasında kesilen bir bilgi parçası
her iki taraftaki chunk'ta da görünür hale gelir.

Kullanım:
    chunker = OverlapChunker(chunk_size=500, overlap_size=100)
    chunks = chunker.chunk(documents)
"""

from typing import List
from llama_index.core import Document
from app.loaders.text_cleaner import full_clean


class OverlapChunker:
    """
    Kayan pencere ile örtüşen parçalama yapan chunker.

    Args:
        chunk_size (int):   Her parçanın karakter sayısı. Varsayılan: 500.
        overlap_size (int): Ardışık parçalar arasındaki örtüşme miktarı. Varsayılan: 100.
    """

    def __init__(self, chunk_size: int = 500, overlap_size: int = 100):
        if chunk_size <= 0:
            raise ValueError("chunk_size pozitif bir tam sayı olmalıdır.")
        if overlap_size < 0:
            raise ValueError("overlap_size negatif olamaz.")
        if overlap_size >= chunk_size:
            raise ValueError("overlap_size, chunk_size'dan küçük olmalıdır.")

        self.chunk_size = chunk_size
        self.overlap_size = overlap_size
        self._step = chunk_size - overlap_size

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Document listesini overlap'li parçalara böler.

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

            chunk_idx = 0
            start = 0

            while start < len(text):
                end = start + self.chunk_size
                chunk_text = text[start:end].strip()

                if chunk_text:
                    metadata = {
                        **doc.metadata,
                        "chunk_index": chunk_idx,
                        "chunk_strategy": "overlap",
                        "chunk_size": self.chunk_size,
                        "overlap_size": self.overlap_size,
                        "char_start": start,
                        "char_end": min(end, len(text)),
                    }
                    chunks.append(Document(text=chunk_text, metadata=metadata))
                    chunk_idx += 1

                start += self._step

                # Son adımda tekrar etmeyi önle
                if start >= len(text):
                    break

        # Toplam chunk sayısını metadata'ya yaz
        _set_total(chunks)
        return chunks

    def __repr__(self) -> str:
        return (
            f"OverlapChunker(chunk_size={self.chunk_size}, "
            f"overlap_size={self.overlap_size})"
        )


def _set_total(chunks: List[Document]) -> None:
    """Her Document'ın metadata'sına chunk_total değerini yazar."""
    total = len(chunks)
    for chunk in chunks:
        chunk.metadata["chunk_total"] = total
