"""
llm_manager.py
==============
Ollama tabanlı yerel LLM yöneticisi.

Ollama'nın REST API'si üzerinden llama3.1:8b (veya başka model)
ile metin üretimi yapar. Bağlantı kontrolü, model doğrulama
ve hata yönetimi sağlar.

Kullanım:
    from app.llm import LLMManager

    llm = LLMManager()                       # varsayılan: llama3.1:8b
    response = llm.generate("Merhaba!")      # basit metin üretimi
    response = llm.chat(system, user)        # system + user prompt

Gereksinimler:
    - Ollama kurulu ve çalışır durumda (https://ollama.com)
    - Model çekilmiş: ollama pull llama3.1:8b
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import urllib.request
import urllib.error

from app.config import settings


# ---------------------------------------------------------------------------
# Sabitler (.env'den yüklenir)
# ---------------------------------------------------------------------------

DEFAULT_MODEL          = settings.OLLAMA_MODEL
DEFAULT_BASE_URL       = settings.OLLAMA_BASE_URL
DEFAULT_TIMEOUT        = settings.OLLAMA_TIMEOUT
DEFAULT_TEMPERATURE    = settings.OLLAMA_TEMPERATURE
DEFAULT_NUM_CTX        = settings.OLLAMA_NUM_CTX
DEFAULT_TOP_P          = settings.OLLAMA_TOP_P
DEFAULT_NUM_GPU        = settings.OLLAMA_NUM_GPU
DEFAULT_REPEAT_PENALTY = settings.OLLAMA_REPEAT_PENALTY


# ---------------------------------------------------------------------------
# Yanıt veri sınıfı
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    """
    LLM yanıtını temsil eder.

    Attributes:
        text            : Üretilen metin.
        model           : Kullanılan model adı.
        total_duration  : Toplam süre (nanosaniye).
        prompt_tokens   : Prompt token sayısı.
        response_tokens : Üretilen token sayısı.
        done            : Üretim tamamlandı mı.
    """
    text:            str
    model:           str            = ""
    total_duration:  int            = 0
    prompt_tokens:   int            = 0
    response_tokens: int            = 0
    done:            bool           = True
    raw:             Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        """Toplam süreyi saniye olarak döndürür."""
        return self.total_duration / 1e9 if self.total_duration else 0.0

    def __str__(self) -> str:
        return self.text


# ---------------------------------------------------------------------------
# LLMManager
# ---------------------------------------------------------------------------

class LLMManager:
    """
    Ollama tabanlı yerel LLM yöneticisi.

    Args:
        model           : Ollama model adı. Varsayılan: llama3.1:8b
        base_url        : Ollama API adresi. Varsayılan: http://localhost:11434
        temperature     : Üretim sıcaklığı (0.0-1.0). Düşük = daha deterministik.
        num_ctx         : Context window boyutu (token).
        top_p           : Nucleus sampling eşiği.
        repeat_penalty  : Tekrar cezası (1.0 = ceza yok).
        timeout         : İstek zaman aşımı (saniye).
    """

    def __init__(
        self,
        model:          str   = DEFAULT_MODEL,
        base_url:       str   = DEFAULT_BASE_URL,
        temperature:    float = DEFAULT_TEMPERATURE,
        num_ctx:        int   = DEFAULT_NUM_CTX,
        top_p:          float = DEFAULT_TOP_P,
        repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
        num_gpu:        int   = DEFAULT_NUM_GPU,
        timeout:        int   = DEFAULT_TIMEOUT,
    ) -> None:
        self.model          = model
        self.base_url       = base_url.rstrip("/")
        self.temperature    = temperature
        self.num_ctx        = num_ctx
        self.top_p          = top_p
        self.repeat_penalty = repeat_penalty
        self.num_gpu        = num_gpu
        self.timeout        = timeout

    # ------------------------------------------------------------------
    # Bağlantı kontrolü
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Ollama sunucusuna erişilebilir mi kontrol eder."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def is_model_pulled(self) -> bool:
        """Belirtilen model Ollama'da mevcut mu kontrol eder."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "") for m in data.get("models", [])]
                # "llama3.1:8b" veya "llama3.1:8b-..." gibi eşleşmeler
                return any(
                    m == self.model or m.startswith(f"{self.model}-")
                    for m in models
                )
        except Exception:
            return False

    def health_check(self) -> Dict[str, Any]:
        """
        Ollama durumunu kontrol eder.

        Returns:
            {"server": bool, "model": bool, "model_name": str, "error": str|None}
        """
        result: Dict[str, Any] = {
            "server": False,
            "model": False,
            "model_name": self.model,
            "error": None,
        }

        if not self.is_available():
            result["error"] = (
                "Ollama sunucusuna erişilemiyor. "
                "Ollama'nın çalıştığından emin olun: https://ollama.com"
            )
            return result

        result["server"] = True

        if not self.is_model_pulled():
            result["error"] = (
                f"'{self.model}' modeli bulunamadı. "
                f"Şu komutla çekin: ollama pull {self.model}"
            )
            return result

        result["model"] = True
        return result

    # ------------------------------------------------------------------
    # Metin üretimi — /api/generate
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
    ) -> LLMResponse:
        """
        /api/generate endpoint'i ile metin üretir.

        Args:
            prompt      : Kullanıcı prompt'u.
            system      : Opsiyonel system prompt.
            temperature : Opsiyonel sıcaklık geçersiz kılma.
            num_ctx     : Opsiyonel context window geçersiz kılma.

        Returns:
            LLMResponse — Üretilen metin ve metadata.

        Raises:
            ConnectionError : Ollama erişilemezse.
            RuntimeError    : API hatası.
        """
        payload: Dict[str, Any] = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature":    temperature or self.temperature,
                "num_ctx":        num_ctx or self.num_ctx,
                "top_p":          self.top_p,
                "repeat_penalty": self.repeat_penalty,
                "num_gpu":        self.num_gpu,
            },
        }

        if system:
            payload["system"] = system

        return self._request("/api/generate", payload)

    # ------------------------------------------------------------------
    # Chat — /api/chat
    # ------------------------------------------------------------------

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: Optional[float] = None,
        num_ctx: Optional[int] = None,
    ) -> LLMResponse:
        """
        /api/chat endpoint'i ile sohbet formatında metin üretir.

        Args:
            system_prompt : Sistem talimatları.
            user_prompt   : Kullanıcı mesajı (kaynaklar + soru).
            temperature   : Opsiyonel sıcaklık geçersiz kılma.
            num_ctx       : Opsiyonel context window geçersiz kılma.

        Returns:
            LLMResponse — Üretilen metin ve metadata.
        """
        payload: Dict[str, Any] = {
            "model":  self.model,
            "stream": False,
            "messages": [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt},
            ],
            "options": {
                "temperature":    temperature or self.temperature,
                "num_ctx":        num_ctx or self.num_ctx,
                "top_p":          self.top_p,
                "repeat_penalty": self.repeat_penalty,
                "num_gpu":        self.num_gpu,
            },
        }

        return self._request("/api/chat", payload)

    # ------------------------------------------------------------------
    # HTTP yardımcı
    # ------------------------------------------------------------------

    def _request(self, endpoint: str, payload: Dict[str, Any]) -> LLMResponse:
        """Ollama API'ye POST isteği yapar ve yanıtı parse eder."""
        url  = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Ollama sunucusuna bağlanılamadı ({self.base_url}). "
                f"Ollama'nın çalıştığından emin olun.\n"
                f"Hata: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Ollama API hatası: {e}") from e

        # /api/generate → response, /api/chat → message.content
        if "message" in body:
            text = body["message"].get("content", "")
        else:
            text = body.get("response", "")

        return LLMResponse(
            text=text.strip(),
            model=body.get("model", self.model),
            total_duration=body.get("total_duration", 0),
            prompt_tokens=body.get("prompt_eval_count", 0),
            response_tokens=body.get("eval_count", 0),
            done=body.get("done", True),
            raw=body,
        )

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "bağlı" if self.is_available() else "bağlantı yok"
        return (
            f"LLMManager("
            f"model='{self.model}', "
            f"url='{self.base_url}', "
            f"temp={self.temperature}, "
            f"status={status})"
        )
