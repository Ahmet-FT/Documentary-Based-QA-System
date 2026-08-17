"""
llm package
===========
Ollama tabanlı yerel LLM yönetimi.

Dışa aktarılan sınıflar:
    LLMManager — Ollama LLM bağlantısı ve metin üretimi
"""

from app.llm.llm_manager import LLMManager

__all__ = ["LLMManager"]
