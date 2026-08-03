import os
import fitz  # PyMuPDF
from typing import List
from llama_index.core import Document
from app.loaders.base_loader import BaseLoader
from app.loaders.text_cleaner import full_clean


class PDFLoader(BaseLoader):
    """
    PDF dosyalarını sayfa bazlı okuyan ve temizleyen yükleyici.

    Her sayfa PyMuPDF ile ham metin olarak çıkarılır,
    ardından text_cleaner.full_clean() ile:
      - Unicode normalizasyonu
      - Tire bölünmesi onarımı
      - Soft newline temizleme
      - Gereksiz boşluk/satır gürültüsü temizleme
    uygulanır.
    """

    def load(self, file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

        documents = []
        file_name = os.path.basename(file_path)

        doc = fitz.open(file_path)

        for page_idx, page in enumerate(doc):
            raw_text = page.get_text()

            # Temizlik pipeline'ını uygula
            clean = full_clean(raw_text)

            # Temizleme sonrası boş sayfaları atla
            if not clean:
                continue

            documents.append(
                Document(
                    text=clean,
                    metadata={
                        "file_name": file_name,
                        "page_number": page_idx + 1,  # 1-indexed sayfa no
                        "total_pages": len(doc),
                    },
                )
            )

        doc.close()
        return documents
