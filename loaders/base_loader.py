from abc import ABC, abstractmethod
from typing import List
from llama_index.core import Document

class BaseLoader(ABC):
    """
    Tüm doküman yükleyiciler (loaders) için ortak ata (parent) sınıf.
    Her alt sınıf bu sınıftan türeyecek ve 'load' metodunu override edecektir.
    """
    
    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """
        Dosyayı okur ve LlamaIndex Document nesnelerinden oluşan bir liste döner.
        
        Args:
            file_path (str): Okunacak dosyanın yolu.
            
        Returns:
            List[Document]: LlamaIndex Document nesneleri listesi.
        """
        pass
