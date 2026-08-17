"""
app/loaders/__init__.py
=======================
Önerilen kullanım:
    from app.loaders import DocumentLoader

    loader = DocumentLoader()
    docs   = loader.load("rapor.pdf")

Geriye dönük uyumluluk için load_document() fonksiyonu da mevcuttur.
"""

from app.loaders.document_loader import DocumentLoader
from app.loaders.text_cleaner import full_clean

# Geriye dönük uyumluluk — eski testler ve kod hâlâ çalışır
from app.loaders.pdf_loader import PDFLoader
from app.loaders.docx_loader import DocxLoader
from app.loaders.txt_loader import TxtLoader
import os


def load_document(file_path: str):
    """
    Geriye dönük uyumluluk fonksiyonu.
    Yeni kodlarda DocumentLoader().load() kullanılmalıdır.
    """
    return DocumentLoader().load(file_path)


__all__ = [
    "DocumentLoader",
    "full_clean",
    "load_document",
    "PDFLoader",
    "DocxLoader",
    "TxtLoader",
]
