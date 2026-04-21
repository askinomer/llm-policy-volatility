"""
OllamaClient — Yerel Ollama LLM API sarmalayıcısı.

Apple Silicon (M4 Pro) üzerinde Llama 3 veya benzeri modelleri
Ollama REST API aracılığıyla çağırır. Ollama yoksa MockNLP'ye fallback yapar.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Ollama REST API (/api/generate) üzerinden LLM çağrısı yapan istemci.

    Parameters
    ----------
    base_url : str
        Ollama sunucu adresi (varsayılan: http://localhost:11434).
    model : str
        Kullanılacak model adı (varsayılan: llama3).
    temperature : float
        Yaratıcılık parametresi. Düşük = deterministik.
    max_tokens : int
        Maksimum token sayısı.
    timeout : int
        HTTP istek zaman aşımı (saniye).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3",
        temperature: float = 0.1,
        max_tokens: int = 512,
        timeout: int = 120,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._available: Optional[bool] = None

    # ------------------------------------------------------------------
    # Sağlık kontrolü
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Ollama sunucusunun çalışıp çalışmadığını kontrol eder."""
        if self._available is not None:
            return self._available

        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            self._available = resp.status_code == 200
            if self._available:
                models = [m["name"] for m in resp.json().get("models", [])]
                logger.info("Ollama aktif. Modeller: %s", models)
                if not any(self._model in m for m in models):
                    logger.warning("Model '%s' Ollama'da bulunamadı. "
                                   "Mevcut: %s", self._model, models)
            return self._available
        except (requests.ConnectionError, requests.Timeout):
            self._available = False
            logger.warning("Ollama sunucusuna bağlanılamadı: %s", self._base_url)
            return False

    # ------------------------------------------------------------------
    # Metin üretimi
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system: str = "",
        json_mode: bool = True,
    ) -> dict:
        """
        Ollama /api/generate endpoint'ine istek gönderir.

        Parameters
        ----------
        prompt : str
            Kullanıcı prompt'u.
        system : str
            System prompt (rol tanımı).
        json_mode : bool
            True ise format="json" gönderilir (yapısal çıktı).

        Returns
        -------
        dict
            Başarılı: {"success": True, "response": str, "duration_ms": float}
            Başarısız: {"success": False, "error": str}
        """
        if not self.is_available():
            return {"success": False, "error": "Ollama sunucusu erişilemez."}

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
            },
        }

        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        start = time.perf_counter()
        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            elapsed = (time.perf_counter() - start) * 1000

            return {
                "success": True,
                "response": data.get("response", ""),
                "duration_ms": elapsed,
                "model": data.get("model", self._model),
                "eval_count": data.get("eval_count", 0),
            }

        except requests.HTTPError as exc:
            logger.error("Ollama HTTP hatası: %s", exc)
            return {"success": False, "error": f"HTTP {resp.status_code}: {exc}"}
        except requests.Timeout:
            logger.error("Ollama zaman aşımı (%ds)", self._timeout)
            return {"success": False, "error": "Timeout"}
        except Exception as exc:
            logger.error("Ollama beklenmedik hata: %s", exc)
            return {"success": False, "error": str(exc)}

    def generate_json(
        self,
        prompt: str,
        system: str = "",
    ) -> Optional[dict]:
        """
        JSON yanıt üretir ve parse eder.

        Returns
        -------
        dict | None
            Başarılı ise parse edilmiş JSON dict, değilse None.
        """
        result = self.generate(prompt, system, json_mode=True)

        if not result["success"]:
            logger.warning("LLM çağrısı başarısız: %s", result["error"])
            return None

        raw = result["response"].strip()
        try:
            parsed = json.loads(raw)
            return parsed
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse hatası: %s\nHam yanıt: %s", exc, raw[:200])
            return self._try_extract_json(raw)

    @staticmethod
    def _try_extract_json(text: str) -> Optional[dict]:
        """JSON bloğunu metinden bulmaya çalışır (LLM bazen ek metin ekler)."""
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass

        return None

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "aktif" if self._available else ("pasif" if self._available is False else "?")
        return (f"OllamaClient(model='{self._model}', "
                f"url='{self._base_url}', status={status})")
