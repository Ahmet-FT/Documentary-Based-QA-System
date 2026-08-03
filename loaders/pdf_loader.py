import os
import fitz  # PyMuPDF
from typing import List
from llama_index.core import Document
from app.loaders.base_loader import BaseLoader

class PDFLoader(BaseLoader):
    """PDF dosyalarını sayfa bazlı okuyan yükleyici."""
    
    def load(self, file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")
            
        documents = []
        file_name = os.path.basename(file_path)
        
        # PDF dosyasını aç
        doc = fitz.open(file_path)
        
        for page_idx, page in enumerate(doc):
            text = page.get_text()
            
            # LlamaIndex Document nesnesi oluştur
            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": file_name,
                        "page_number": page_idx + 1,  # 1-indexed sayfa no
                        "total_pages": len(doc)
                    }
                )
            )
            
        doc.close()
        return documents
