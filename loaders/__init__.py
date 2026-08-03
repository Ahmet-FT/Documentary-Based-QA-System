import os
from typing import List
from llama_index.core import Document
from app.loaders.pdf_loader import PDFLoader
from app.loaders.docx_loader import DocxLoader
from app.loaders.txt_loader import TxtLoader

def load_document(file_path: str) -> List[Document]:
    """
    Dosya uzantısına göre doğru yükleyiciyi otomatik seçer ve metin çıkarımını yapar.
    
    Args:
        file_path (str): Okunacak dosya yolu.
        
    Returns:
        List[Document]: LlamaIndex Document nesneleri listesi.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        loader = PDFLoader()
    elif ext == ".docx":
        loader = DocxLoader()
    elif ext in [".txt", ".md"]:
        loader = TxtLoader()
    else:
        raise ValueError(f"Desteklenmeyen dosya formatı: {ext}")
        
    return loader.load(file_path)
