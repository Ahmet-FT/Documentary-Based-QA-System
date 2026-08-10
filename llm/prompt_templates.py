"""
prompt_templates.py
===================
Kaynak tabanlı soru-cevap için prompt şablonları.

Kurallar:
    1. LLM yalnızca verilen kaynaklara dayanarak cevap verir.
    2. Kaynakta bulunmayan bilgi için bunu açıkça belirtir.
    3. Her iddiayı kaynak numarası ile destekler.
    4. Türkçe cevap üretir (sorgu dili ne olursa olsun).

Kullanım:
    from app.llm.prompt_templates import build_qa_prompt

    prompt = build_qa_prompt(
        query="Yapay zeka nedir?",
        contexts=[
            {"text": "AI metni...", "source": "rapor.pdf", "page": "3"},
        ],
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# System Prompt — LLM'in genel davranış kuralları
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Sen bir kaynak göstermeli soru-cevap asistanısın. Görevin, yalnızca sana \
verilen doküman parçalarına dayanarak soruları cevaplamaktır.

### KESİN KURALLAR ###

1. **YALNIZCA** verilen kaynaklardaki bilgileri kullan. Kendi bilgini, \
   tahminini veya dış kaynakları kesinlikle KULLANMA.

2. Cevabında her iddiayı ilgili kaynak numarası ile belirt. \
   Kaynak gösterim formatı: [Kaynak N] (örn. [Kaynak 1], [Kaynak 2]).

3. Eğer soru, verilen kaynaklarda YOKSA veya yetersizse, şunu yaz:
   "Bu sorunun cevabı verilen dokümanlarda bulunamamıştır."
   Kesinlikle uydurma veya tahmin YAPMA.

4. Birden fazla kaynaktan bilgi birleştiriliyorsa, her cümle veya \
   paragrafta hangi kaynaktan geldiğini ayrı ayrı belirt.

5. Cevabı Türkçe ver. Açık, anlaşılır ve yapılandırılmış ol.

6. Cevabın sonunda kullandığın kaynakları listele:
   --- Kaynaklar ---
   [Kaynak 1] dosya_adi.pdf, Sayfa: 3
   [Kaynak 2] rapor.docx, Sayfa: 7
"""


# ---------------------------------------------------------------------------
# Context formatlama
# ---------------------------------------------------------------------------

def _format_context_block(
    contexts: List[Dict[str, Any]],
    max_chars_per_chunk: int = 1500,
) -> str:
    """
    Retrieval sonuçlarını numaralı kaynak bloklarına dönüştürür.

    Args:
        contexts : Her biri en az 'text', 'source', 'page' anahtarlarını
                   içeren sözlük listesi.
        max_chars_per_chunk : Chunk metninin kesilme sınırı.

    Returns:
        Biçimlendirilmiş metin bloğu.
    """
    if not contexts:
        return "(Kaynak bulunamadı.)"

    blocks: list[str] = []
    for i, ctx in enumerate(contexts, start=1):
        text   = ctx.get("text", "").strip()
        source = ctx.get("source", "bilinmiyor")
        page   = ctx.get("page", "?")
        score  = ctx.get("score")

        # Uzun chunk'ları kırp
        if len(text) > max_chars_per_chunk:
            text = text[:max_chars_per_chunk] + "..."

        header = f"[Kaynak {i}] — {source}, Sayfa: {page}"
        if score is not None:
            header += f" (benzerlik: {score:.2f})"

        blocks.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Prompt oluşturucu
# ---------------------------------------------------------------------------

def build_qa_prompt(
    query: str,
    contexts: List[Dict[str, Any]],
    max_chars_per_chunk: int = 1500,
    language: str = "Türkçe",
) -> str:
    """
    Retrieval sonuçları ve kullanıcı sorgusundan LLM prompt'u oluşturur.

    Args:
        query               : Kullanıcı sorusu.
        contexts            : Retriever sonuçları (dict listesi).
        max_chars_per_chunk  : Chunk başına maks. karakter.
        language            : Cevap dili.

    Returns:
        LLM'e gönderilecek tam prompt metni.
    """
    context_block = _format_context_block(contexts, max_chars_per_chunk)

    user_prompt = f"""\
### VERİLEN KAYNAKLAR ###

{context_block}

### KULLANICI SORUSU ###

{query}

### TALİMAT ###

Yukarıdaki kaynaklara dayanarak soruyu {language} olarak cevapla. \
Her iddiayı [Kaynak N] formatında kaynak göstererek destekle. \
Cevabın sonunda kullandığın kaynakların listesini ver. \
Eğer soru kaynaklarda yoksa "Bu sorunun cevabı verilen dokümanlarda \
bulunamamıştır." yaz."""

    return user_prompt


def get_system_prompt() -> str:
    """System prompt'u döndürür."""
    return SYSTEM_PROMPT
