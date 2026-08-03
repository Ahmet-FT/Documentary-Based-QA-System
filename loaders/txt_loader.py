import os
from typing import List
from llama_index.core import Document
from app.loaders.base_loader import BaseLoader

class TxtLoader(BaseLoader):
    """TXT dosyalarını okuyan yükleyici."""
    
    def load(self, file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
            
        file_name = os.path.basename(file_path)
        
        with open(file_path, "r", encoding="utf-8") as f:
            text_content = f.read()
            
        return [
            Document(
                text=text_content,
                metadata={
                    "file_name": file_name,
                    "page_number": 1
                }
            )
        ]
