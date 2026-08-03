import os
import docx
from typing import List
from llama_index.core import Document
from app.loaders.base_loader import BaseLoader

class DocxLoader(BaseLoader):
    """Word (.docx) dosyalarını okuyan yükleyici."""
    
    def load(self, file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
            
        file_name = os.path.basename(file_path)
        doc = docx.Document(file_path)
        
        # Tüm paragraflardaki metinleri ham şekilde birleştir
        text_content = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        
        # Word belgelerinde sayfa kavramı sabit olmadığından sayfa no: 1 verilir
        return [
            Document(
                text=text_content,
                metadata={
                    "file_name": file_name,
                    "page_number": 1
                }
            )
        ]
