"""
Paragraf / Bölüm Bazlı Chunker (Paragraph/Section Chunker)
============================================================
Metni boş satırlarla ayrılmış paragraflara veya
başlık kalıplarıyla ayrılmış bölümlere böler.

Semantik bölünme önceliklidir:
  1. Bölüm başlıkları tespit edilirse (Tanım, 1. Giriş, ## Başlık vb.) bölümlere ayrılır.
  2. Başlık yoksa paragraf bazlı bölünme uygulanır.
  3. Paragraf küçükse (min_chunk_size altındaysa) bir sonrakiyle birleştirilir.
  4. Paragraf büyükse (max_chunk_size üzerindeyse) sabit boyutlu alt parçalara bölünür.

Kullanım:
    chunker = ParagraphChunker(min_chunk_size=100, max_chunk_size=1000)
    chunks = chunker.chunk(documents)
"""

import re
from typing import List
from llama_index.core import Document
from app.loaders.text_cleaner import full_clean


# Bölüm başlığı olarak kabul edilen kalıplar
_SECTION_PATTERNS = [
    # Markdown başlıkları: ## Başlık
    re.compile(r"^#{1,6}\s+.+", re.MULTILINE),
    # Numaralı bölümler: "1. Giriş", "2.3 Yöntem"
    re.compile(r"^\d+(\.\d+)*[\.\)]\s+[A-ZÇŞĞÜÖİa-zçşğüöı].+", re.MULTILINE),
    # Tamamı büyük harf olan kısa başlıklar (en az 3, en fazla 60 karakter)
    re.compile(r"^[A-ZÇŞĞÜÖİ][A-ZÇŞĞÜÖİ\s]{2,58}$", re.MULTILINE),
]


class ParagraphChunker:
    """
    Paragraf veya bölüm bazlı semantik chunker.

    Args:
        min_chunk_size (int): Bir parçanın birleştirme için minimum karakter sayısı. Varsayılan: 100.
        max_chunk_size (int): Bir parçanın alt parçalara bölüneceği maksimum karakter sayısı. Varsayılan: 1000.
    """

    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 1000):
        if min_chunk_size <= 0:
            raise ValueError("min_chunk_size pozitif olmalıdır.")
        if max_chunk_size <= min_chunk_size:
            raise ValueError("max_chunk_size, min_chunk_size'dan büyük olmalıdır.")

        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Document listesini paragraf/bölüm bazlı parçalara böler.

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

            raw_segments = self._split_into_segments(text)
            merged = self._merge_small_segments(raw_segments)
            final_segments = self._split_large_segments(merged)

            for chunk_idx, segment in enumerate(final_segments):
                segment = segment.strip()
                if not segment:
                    continue

                metadata = {
                    **doc.metadata,
                    "chunk_index": chunk_idx,
                    "chunk_total": len(final_segments),
                    "chunk_strategy": "paragraph",
                    "min_chunk_size": self.min_chunk_size,
                    "max_chunk_size": self.max_chunk_size,
                    "char_count": len(segment),
                }
                chunks.append(Document(text=segment, metadata=metadata))

        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _split_into_segments(self, text: str) -> List[str]:
        """
        Metni önce bölüm başlıklarına, bulamazsa paragraflara göre böler.
        """
        # Bölüm başlığı var mı kontrol et
        if self._has_section_headers(text):
            return self._split_by_sections(text)
        return self._split_by_paragraphs(text)

    @staticmethod
    def _has_section_headers(text: str) -> bool:
        """Metinde bölüm başlığı kalıbı var mı?"""
        for pattern in _SECTION_PATTERNS:
            if pattern.search(text):
                return True
        return False

    @staticmethod
    def _split_by_paragraphs(text: str) -> List[str]:
        """Çift boş satırla ayrılmış paragrafları döner."""
        paragraphs = re.split(r"\n\s*\n", text)
        return [p.strip() for p in paragraphs if p.strip()]

    @staticmethod
    def _split_by_sections(text: str) -> List[str]:
        """
        Bölüm başlığı kalıplarından önce bölünme noktası ekleyerek
        metni semantik bölümlere ayırır.
        """
        # Tüm başlık eşleşmelerini bul
        split_positions = set()
        for pattern in _SECTION_PATTERNS:
            for match in pattern.finditer(text):
                split_positions.add(match.start())

        if not split_positions:
            return [text.strip()]

        # Sıralı pozisyonlardan dilimleri oluştur
        positions = sorted(split_positions)
        segments = []
        prev = 0
        for pos in positions:
            segment = text[prev:pos].strip()
            if segment:
                segments.append(segment)
            prev = pos

        # Son segment
        last = text[prev:].strip()
        if last:
            segments.append(last)

        return segments if segments else [text.strip()]

    def _merge_small_segments(self, segments: List[str]) -> List[str]:
        """
        min_chunk_size'dan küçük segmentleri bir sonrakiyle birleştirir.
        """
        if not segments:
            return []

        merged = []
        buffer = segments[0]

        for segment in segments[1:]:
            if len(buffer) < self.min_chunk_size:
                buffer = buffer + "\n\n" + segment
            else:
                merged.append(buffer)
                buffer = segment

        merged.append(buffer)
        return merged

    def _split_large_segments(self, segments: List[str]) -> List[str]:
        """
        max_chunk_size'dan büyük segmentleri sabit boyutlu alt parçalara böler.
        """
        result = []
        for segment in segments:
            if len(segment) <= self.max_chunk_size:
                result.append(segment)
            else:
                # Büyük segmenti max_chunk_size'lık parçalara böl
                start = 0
                while start < len(segment):
                    result.append(segment[start : start + self.max_chunk_size])
                    start += self.max_chunk_size
        return result

    def __repr__(self) -> str:
        return (
            f"ParagraphChunker(min_chunk_size={self.min_chunk_size}, "
            f"max_chunk_size={self.max_chunk_size})"
        )
