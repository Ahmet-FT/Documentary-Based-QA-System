"""
unified_chunker.py
==================
Tüm chunking stratejilerini tek bir modülde ve tek bir arayüzle sunar.

Desteklenen modlar (ChunkMode enum):
  FIXED        — Sabit karakter uzunluğu (elle yazılmış, LlamaIndex bağımsız)
  OVERLAP      — Kayan pencere + overlap (elle yazılmış, LlamaIndex bağımsız)
  PARAGRAPH    — Paragraf/bölüm bazlı semantik (elle yazılmış)
  SENTENCE     — LlamaIndex SentenceSplitter: cümle sınırına saygılı, overlap destekli
  RECURSIVE    — LlamaIndex SentenceSplitter ile yapısal recursive + overlap birleşimi

Kullanım:
    from app.chunkers.unified_chunker import UnifiedChunker, ChunkMode

    chunker = UnifiedChunker(
        mode=ChunkMode.RECURSIVE,
        chunk_size=512,
        chunk_overlap=128,
    )
    chunks = chunker.chunk(documents)
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
    FIXED     = "fixed"      # Sabit karakter penceresi, overlap yok
    OVERLAP   = "overlap"    # Kayan pencere + overlap
    PARAGRAPH = "paragraph"  # Paragraf/bölüm bazlı semantik
    SENTENCE  = "sentence"   # LlamaIndex SentenceSplitter (cümle sınırı korumalı)
    RECURSIVE = "recursive"  # Yapısal recursive + overlap (LlamaIndex SentenceSplitter tabanlı)


# ---------------------------------------------------------------------------
# Ana sınıf
# ---------------------------------------------------------------------------

class UnifiedChunker:
    """
    Tek arayüzden tüm chunking stratejilerine erişim sağlar.

    Args:
        mode (ChunkMode | str):
            Kullanılacak strateji. Varsayılan: ChunkMode.RECURSIVE
        chunk_size (int):
            Her chunk'ın hedef boyutu.
            FIXED/OVERLAP/PARAGRAPH → karakter sayısı
            SENTENCE/RECURSIVE      → token sayısı (LlamaIndex tokenizer)
            Varsayılan: 512
        chunk_overlap (int):
            Ardışık chunk'lar arasındaki örtüşme miktarı.
            FIXED modunda kullanılmaz.
            Varsayılan: 128
        min_chunk_size (int):
            PARAGRAPH modunda küçük segmentleri birleştirme eşiği.
            Varsayılan: 100
        paragraph_separator (str):
            SENTENCE/RECURSIVE modunda paragraf ayracı.
            Varsayılan: "\\n\\n"
    """

    def __init__(
        self,
        mode: ChunkMode | str = ChunkMode.RECURSIVE,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 100,
        paragraph_separator: str = "\n\n",
    ):
        self.mode = ChunkMode(mode)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.paragraph_separator = paragraph_separator

        # LlamaIndex tabanlı splitter'lar için nesneyi önceden hazırla
        self._llama_splitter: Optional[SentenceSplitter] = None
        if self.mode in (ChunkMode.SENTENCE, ChunkMode.RECURSIVE):
            self._llama_splitter = self._build_llama_splitter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, documents: List[Document]) -> List[Document]:
        """
        Document listesini seçili stratejiye göre parçalar.

        Args:
            documents: LlamaIndex Document nesneleri listesi.

        Returns:
            Parçalanmış ve metadata zenginleştirilmiş Document listesi.
        """
        if self.mode == ChunkMode.FIXED:
            return self._chunk_fixed(documents)
        elif self.mode == ChunkMode.OVERLAP:
            return self._chunk_overlap(documents)
        elif self.mode == ChunkMode.PARAGRAPH:
            return self._chunk_paragraph(documents)
        elif self.mode == ChunkMode.SENTENCE:
            return self._chunk_llama(documents, strategy_label="sentence")
        elif self.mode == ChunkMode.RECURSIVE:
            return self._chunk_llama(documents, strategy_label="recursive")
        else:
            raise ValueError(f"Bilinmeyen mod: {self.mode}")

    def describe(self) -> str:
        """Mevcut konfigürasyonu açıklayan tek satırlık özet."""
        base = (
            f"UnifiedChunker(mode={self.mode.value}, "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )
        if self.mode == ChunkMode.RECURSIVE:
            base += (
                " | Yapisal: paragraf > cumle > kelime > karakter "
                "(LlamaIndex SentenceSplitter + recursive fallback)"
            )
        elif self.mode == ChunkMode.SENTENCE:
            base += " | Cumle siniri korumal, overlap destekli (LlamaIndex SentenceSplitter)"
        elif self.mode == ChunkMode.PARAGRAPH:
            base += f" | min_chunk={self.min_chunk_size}, max_chunk={self.chunk_size}"
        return base

    # ------------------------------------------------------------------
    # LlamaIndex tabanlı splitter
    # ------------------------------------------------------------------

    def _build_llama_splitter(self) -> SentenceSplitter:
        """
        SentenceSplitter'ı yapılandırır.

        RECURSIVE modunda secondary_chunking_regex ile cümle→kelime→karakter
        kademeli fallback zinciri kurulur (recursive davranış).
        """
        if self.mode == ChunkMode.RECURSIVE:
            # Kademeli ayraç zinciri: paragraf → cümle sonu → virgül/noktalı virgül → kelime → karakter
            secondary_regex = r"(?<=[.?!。])\s+|(?<=;)\s+|(?<=,)\s+|\s+|."
        else:
            # Standart cümle ayracı
            secondary_regex = r"(?<=[.?!。])\s+"

        return SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            paragraph_separator=self.paragraph_separator,
            secondary_chunking_regex=secondary_regex,
        )

    def _chunk_llama(
        self, documents: List[Document], strategy_label: str
    ) -> List[Document]:
        """
        LlamaIndex SentenceSplitter ile parçalama.
        Splitter TextNode döner; bunları Document'a dönüştürürüz.
        """
        cleaned_docs = []
        for doc in documents:
            clean = full_clean(doc.text)
            if clean:
                cleaned_docs.append(Document(text=clean, metadata=doc.metadata))

        if not cleaned_docs:
            return []

        nodes = self._llama_splitter.get_nodes_from_documents(cleaned_docs)

        chunks: List[Document] = []
        for idx, node in enumerate(nodes):
            metadata = {
                **node.metadata,
                "chunk_index":    idx,
                "chunk_total":    len(nodes),
                "chunk_strategy": strategy_label,
                "chunk_size":     self.chunk_size,
                "chunk_overlap":  self.chunk_overlap,
                "char_count":     len(node.text),
            }
            chunks.append(Document(text=node.text, metadata=metadata))

        return chunks

    # ------------------------------------------------------------------
    # Elle yazılmış stratejiler (bağımlılık gerektirmez)
    # ------------------------------------------------------------------

    def _chunk_fixed(self, documents: List[Document]) -> List[Document]:
        chunks: List[Document] = []
        for doc in documents:
            text = full_clean(doc.text)
            if not text:
                continue
            total = (len(text) + self.chunk_size - 1) // self.chunk_size
            for idx, start in enumerate(range(0, len(text), self.chunk_size)):
                chunk_text = text[start : start + self.chunk_size].strip()
                if not chunk_text:
                    continue
                chunks.append(Document(
                    text=chunk_text,
                    metadata={
                        **doc.metadata,
                        "chunk_index":    idx,
                        "chunk_total":    total,
                        "chunk_strategy": "fixed",
                        "chunk_size":     self.chunk_size,
                        "chunk_overlap":  0,
                        "char_count":     len(chunk_text),
                    },
                ))
        return chunks

    def _chunk_overlap(self, documents: List[Document]) -> List[Document]:
        step = self.chunk_size - self.chunk_overlap
        chunks: List[Document] = []
        for doc in documents:
            text = full_clean(doc.text)
            if not text:
                continue
            idx = 0
            start = 0
            while start < len(text):
                chunk_text = text[start : start + self.chunk_size].strip()
                if chunk_text:
                    chunks.append(Document(
                        text=chunk_text,
                        metadata={
                            **doc.metadata,
                            "chunk_index":    idx,
                            "chunk_strategy": "overlap",
                            "chunk_size":     self.chunk_size,
                            "chunk_overlap":  self.chunk_overlap,
                            "char_count":     len(chunk_text),
                        },
                    ))
                    idx += 1
                start += step
                if start >= len(text):
                    break
        # Toplam chunk sayısını güncelle
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
            segments = self._split_paragraphs(text)
            segments = self._merge_small(segments)
            segments = self._split_large(segments)
            for idx, seg in enumerate(segments):
                seg = seg.strip()
                if not seg:
                    continue
                chunks.append(Document(
                    text=seg,
                    metadata={
                        **doc.metadata,
                        "chunk_index":    idx,
                        "chunk_total":    len(segments),
                        "chunk_strategy": "paragraph",
                        "chunk_size":     self.chunk_size,
                        "chunk_overlap":  0,
                        "char_count":     len(seg),
                    },
                ))
        return chunks

    # ------------------------------------------------------------------
    # Paragraf yardımcıları
    # ------------------------------------------------------------------

    def _split_paragraphs(self, text: str) -> List[str]:
        parts = re.split(r"\n\s*\n", text)
        return [p.strip() for p in parts if p.strip()]

    def _merge_small(self, segments: List[str]) -> List[str]:
        if not segments:
            return []
        merged = []
        buf = segments[0]
        for seg in segments[1:]:
            if len(buf) < self.min_chunk_size:
                buf = buf + "\n\n" + seg
            else:
                merged.append(buf)
                buf = seg
        merged.append(buf)
        return merged

    def _split_large(self, segments: List[str]) -> List[str]:
        result = []
        for seg in segments:
            if len(seg) <= self.chunk_size:
                result.append(seg)
            else:
                start = 0
                while start < len(seg):
                    result.append(seg[start : start + self.chunk_size])
                    start += self.chunk_size
        return result

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"UnifiedChunker(mode='{self.mode.value}', "
            f"chunk_size={self.chunk_size}, "
            f"chunk_overlap={self.chunk_overlap})"
        )
