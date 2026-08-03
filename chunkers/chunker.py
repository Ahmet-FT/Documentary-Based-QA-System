"""
chunker.py
==========
Tüm chunking stratejilerini tek bir modülde sunar.

Proje için önerilen yaklaşım:
    ChunkMode.RECURSIVE | chunk_size=512 | chunk_overlap=128

Desteklenen modlar (ChunkMode):
    FIXED      — Sabit karakter penceresi, overlap yok
    OVERLAP    — Kayan pencere + overlap (karakter bazlı)
    PARAGRAPH  — Paragraf/bölüm bazlı semantik bölünme
    SENTENCE   — LlamaIndex SentenceSplitter (cümle sınırı korumalı)
    RECURSIVE  — LlamaIndex SentenceSplitter + yapısal fallback zinciri
                 paragraf → cümle → kelime → karakter (önerilen)

Kullanım:
    from app.chunkers.chunker import Chunker, ChunkMode

    chunker = Chunker(mode=ChunkMode.RECURSIVE, chunk_size=512, chunk_overlap=128)
    chunks  = chunker.chunk(documents)
    print(chunker.describe())
"""

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from app.loaders.text_cleaner import full_clean


# ---------------------------------------------------------------------------
# Mod tanımları
# ---------------------------------------------------------------------------

class ChunkMode(str, Enum):
    FIXED     = "fixed"      # Sabit karakter penceresi
    OVERLAP   = "overlap"    # Kayan pencere + overlap
    PARAGRAPH = "paragraph"  # Paragraf/bölüm semantik
    SENTENCE  = "sentence"   # LlamaIndex: cümle sınırı korumalı
    RECURSIVE = "recursive"  # LlamaIndex: yapısal recursive + overlap (önerilen)


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------

class Chunker:
    """
    Tek bir arayüzden tüm chunking stratejilerine erişim sağlar.

    Args:
        mode (ChunkMode | str):
            Parçalama stratejisi. Varsayılan: ChunkMode.RECURSIVE
        chunk_size (int):
            FIXED/OVERLAP/PARAGRAPH → karakter sayısı.
            SENTENCE/RECURSIVE      → token sayısı (LlamaIndex tokenizer).
            Varsayılan: 512
        chunk_overlap (int):
            Ardışık chunk'lar arasındaki örtüşme.
            FIXED ve PARAGRAPH modunda kullanılmaz.
            Varsayılan: 128
        min_chunk_size (int):
            PARAGRAPH modunda küçük segmentleri birleştirme eşiği.
            Varsayılan: 100
        paragraph_separator (str):
            SENTENCE/RECURSIVE için paragraf ayracı. Varsayılan: "\\n\\n"
    """

    def __init__(
        self,
        mode: ChunkMode | str = ChunkMode.RECURSIVE,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 100,
        paragraph_separator: str = "\n\n",
    ):
        self.mode               = ChunkMode(mode)
        self.chunk_size         = chunk_size
        self.chunk_overlap      = chunk_overlap
        self.min_chunk_size     = min_chunk_size
        self.paragraph_separator = paragraph_separator

        # LlamaIndex splitter — sadece SENTENCE/RECURSIVE için oluşturulur
        self._splitter: Optional[SentenceSplitter] = None
        if self.mode in (ChunkMode.SENTENCE, ChunkMode.RECURSIVE):
            self._splitter = self._build_splitter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Document listesini seçili stratejiye göre parçalar.

        Her sonuç Document'ının metadata'sına şunlar eklenir:
            chunk_index    : Bu chunk'ın sırası (0-tabanlı)
            chunk_total    : Toplam chunk sayısı
            chunk_strategy : Kullanılan strateji adı
            chunk_size     : Konfigüre edilen boyut
            chunk_overlap  : Konfigüre edilen overlap (0 ise yok)
            char_count     : Bu chunk'ın karakter sayısı
            word_count     : Bu chunk'ın kelime sayısı

        Args:
            documents: LlamaIndex Document nesneleri listesi.

        Returns:
            Parçalanmış Document nesneleri listesi.
        """
        dispatch = {
            ChunkMode.FIXED:     self._chunk_fixed,
            ChunkMode.OVERLAP:   self._chunk_overlap,
            ChunkMode.PARAGRAPH: self._chunk_paragraph,
            ChunkMode.SENTENCE:  lambda d: self._chunk_llama(d, "sentence"),
            ChunkMode.RECURSIVE: lambda d: self._chunk_llama(d, "recursive"),
        }
        return dispatch[self.mode](documents)

    def describe(self) -> str:
        """Mevcut konfigürasyonu açıklayan tek satırlık özet."""
        descs = {
            ChunkMode.FIXED:
                "Sabit karakter penceresi, overlap yok",
            ChunkMode.OVERLAP:
                f"Kayan pencere, overlap={self.chunk_overlap} karakter",
            ChunkMode.PARAGRAPH:
                f"Paragraf/bolum semantik, min={self.min_chunk_size} maks={self.chunk_size} karakter",
            ChunkMode.SENTENCE:
                "LlamaIndex SentenceSplitter — cumle siniri korumal, overlap destekli",
            ChunkMode.RECURSIVE:
                "LlamaIndex SentenceSplitter — yapısal: paragraf > cumle > kelime > karakter (ONERILIR)",
        }
        return (
            f"Chunker(mode={self.mode.value}, "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap}) | "
            f"{descs[self.mode]}"
        )

    # ------------------------------------------------------------------
    # LlamaIndex tabanlı parçalama
    # ------------------------------------------------------------------

    def _build_splitter(self) -> SentenceSplitter:
        """
        SENTENCE  → standart cümle sonu regex
        RECURSIVE → kademeli fallback: cümle > virgül > kelime > karakter
        """
        if self.mode == ChunkMode.RECURSIVE:
            secondary_regex = r"(?<=[.?!。])\s+|(?<=;)\s+|(?<=,)\s+|\s+|."
        else:
            secondary_regex = r"(?<=[.?!。])\s+"

        return SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            paragraph_separator=self.paragraph_separator,
            secondary_chunking_regex=secondary_regex,
        )

    def _chunk_llama(self, documents: List[Document], label: str) -> List[Document]:
        """LlamaIndex SentenceSplitter ile parçalar; Node → Document dönüşümü yapar."""
        cleaned: List[Document] = []
        for doc in documents:
            text = full_clean(doc.text)
            if text:
                cleaned.append(Document(text=text, metadata=doc.metadata.copy()))

        if not cleaned:
            return []

        nodes = self._splitter.get_nodes_from_documents(cleaned)

        chunks: List[Document] = []
        total = len(nodes)
        for idx, node in enumerate(nodes):
            metadata = {
                **node.metadata,
                "chunk_index":    idx,
                "chunk_total":    total,
                "chunk_strategy": label,
                "chunk_size":     self.chunk_size,
                "chunk_overlap":  self.chunk_overlap,
                "char_count":     len(node.text),
                "word_count":     len(node.text.split()),
            }
            chunks.append(Document(text=node.text, metadata=metadata))

        return chunks

    # ------------------------------------------------------------------
    # Pure-Python stratejiler
    # ------------------------------------------------------------------

    def _chunk_fixed(self, documents: List[Document]) -> List[Document]:
        chunks: List[Document] = []
        for doc in documents:
            text = full_clean(doc.text)
            if not text:
                continue
            total = (len(text) + self.chunk_size - 1) // self.chunk_size
            for idx, start in enumerate(range(0, len(text), self.chunk_size)):
                piece = text[start: start + self.chunk_size].strip()
                if not piece:
                    continue
                chunks.append(Document(
                    text=piece,
                    metadata={
                        **doc.metadata,
                        "chunk_index":    idx,
                        "chunk_total":    total,
                        "chunk_strategy": "fixed",
                        "chunk_size":     self.chunk_size,
                        "chunk_overlap":  0,
                        "char_count":     len(piece),
                        "word_count":     len(piece.split()),
                    },
                ))
        return chunks

    def _chunk_overlap(self, documents: List[Document]) -> List[Document]:
        step   = self.chunk_size - self.chunk_overlap
        chunks: List[Document] = []
        for doc in documents:
            text = full_clean(doc.text)
            if not text:
                continue
            idx, start = 0, 0
            while start < len(text):
                piece = text[start: start + self.chunk_size].strip()
                if piece:
                    chunks.append(Document(
                        text=piece,
                        metadata={
                            **doc.metadata,
                            "chunk_index":    idx,
                            "chunk_strategy": "overlap",
                            "chunk_size":     self.chunk_size,
                            "chunk_overlap":  self.chunk_overlap,
                            "char_count":     len(piece),
                            "word_count":     len(piece.split()),
                        },
                    ))
                    idx += 1
                start += step
        total = len(chunks)
        for c in chunks:
            c.metadata["chunk_total"] = total
        return chunks

    def _chunk_paragraph(self, documents: List[Document]) -> List[Document]:
        chunks: List[Document] = []
        for doc in documents:
            text = full_clean(doc.text)
            if not text:
                continue
            segments = self._split_para(text)
            segments = self._merge_small(segments)
            segments = self._split_large(segments)
            total = len([s for s in segments if s.strip()])
            real_idx = 0
            for seg in segments:
                seg = seg.strip()
                if not seg:
                    continue
                chunks.append(Document(
                    text=seg,
                    metadata={
                        **doc.metadata,
                        "chunk_index":    real_idx,
                        "chunk_total":    total,
                        "chunk_strategy": "paragraph",
                        "chunk_size":     self.chunk_size,
                        "chunk_overlap":  0,
                        "char_count":     len(seg),
                        "word_count":     len(seg.split()),
                    },
                ))
                real_idx += 1
        return chunks

    # ------------------------------------------------------------------
    # Paragraf yardımcıları
    # ------------------------------------------------------------------

    def _split_para(self, text: str) -> List[str]:
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    def _merge_small(self, segs: List[str]) -> List[str]:
        if not segs:
            return []
        merged, buf = [], segs[0]
        for seg in segs[1:]:
            if len(buf) < self.min_chunk_size:
                buf += "\n\n" + seg
            else:
                merged.append(buf)
                buf = seg
        merged.append(buf)
        return merged

    def _split_large(self, segs: List[str]) -> List[str]:
        result = []
        for seg in segs:
            if len(seg) <= self.chunk_size:
                result.append(seg)
            else:
                start = 0
                while start < len(seg):
                    result.append(seg[start: start + self.chunk_size])
                    start += self.chunk_size
        return result

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Chunker(mode='{self.mode.value}', "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )
