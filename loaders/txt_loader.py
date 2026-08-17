import os
from typing import List
from llama_index.core import Document
from app.loaders.base_loader import BaseLoader
from app.loaders.text_cleaner import full_clean


class TxtLoader(BaseLoader):
    """TXT / Markdown dosyalarını okuyup temizleyen yükleyici."""

    def load(self, file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

        file_name = os.path.basename(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()

        # Temizlik pipeline'ını uygula
        clean = full_clean(raw_text)

        if not clean:
            return []

        return [
            Document(
                text=clean,
                metadata={
                    "file_name": file_name,
                    "page_number": 1,
                },
            )
        ]
