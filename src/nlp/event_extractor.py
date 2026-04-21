"""
EventExtractor — Ana NLP orkestratör sınıfı.

Ollama varsa LLM kullanır, yoksa MockNLP'ye otomatik düşer.
Sonuçları doğrular (validate), DataFrame'e dönüştürür ve DB'ye kaydeder.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from ..pipeline.database import DatabaseManager
from .llm_client import OllamaClient
from .mock_nlp import MockNLP
from .prompts import build_system_prompt, build_user_prompt, EVENT_TYPES

logger = logging.getLogger(__name__)


class EventExtractor:
    """
    Haber metinlerinden yapılandırılmış olay ve belirsizlik skoru çıkaran sınıf.

    Akış:
        1. Ollama (Llama 3) mevcutsa → LLM ile analiz
        2. Değilse → MockNLP ile rule-based analiz
        3. Sonuçları validate et
        4. DB'ye kaydet

    Parameters
    ----------
    db : DatabaseManager
        Veritabanı bağlantısı.
    config : dict
        nlp bölümü (model, ollama_base_url, fallback, temperature, max_tokens).
    force_mock : bool
        True ise Ollama kontrolü yapılmadan doğrudan MockNLP kullanılır.
    """

    def __init__(
        self,
        db: DatabaseManager,
        config: dict,
        force_mock: bool = False,
    ) -> None:
        self._db = db
        self._config = config
        self._force_mock = force_mock

        self._ollama = OllamaClient(
            base_url=config.get("ollama_base_url", "http://localhost:11434"),
            model=config.get("model", "llama3"),
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 512),
        )
        self._mock = MockNLP()
        self._system_prompt = build_system_prompt()
        self._use_llm = False

        if not force_mock and self._ollama.is_available():
            self._use_llm = True
            logger.info("LLM modu aktif: %s", self._ollama)
        else:
            logger.info("MockNLP modu aktif (Ollama erişilemez veya force_mock=True).")

    # ------------------------------------------------------------------
    # Ana analiz fonksiyonları
    # ------------------------------------------------------------------

    def extract_single(self, text: str, source: str = "", date: str = "") -> dict:
        """
        Tek bir metin için olay çıkarımı yapar.

        Returns
        -------
        dict
            event_type, uncertainty_score, impact_direction, confidence,
            key_phrases, reasoning, model_used
        """
        if self._use_llm:
            result = self._llm_analyze(text, source, date)
            if result is not None:
                return result
            logger.warning("LLM başarısız, MockNLP'ye fallback.")

        return self._mock.analyze(text, source, date)

    def extract_batch(self, articles: list[dict]) -> list[dict]:
        """
        Birden fazla haber için toplu analiz.

        Parameters
        ----------
        articles : list[dict]
            Her dict en az şunları içermeli: id, content, source, date

        Returns
        -------
        list[dict]
            Her biri validate edilmiş skor dict'i.
        """
        if not articles:
            return []

        if self._use_llm:
            results = []
            for article in articles:
                result = self.extract_single(
                    text=article.get("content", ""),
                    source=article.get("source", ""),
                    date=article.get("date", ""),
                )
                result["news_id"] = article.get("id")
                result["date"] = article.get("date", "")
                results.append(result)
        else:
            results = self._mock.analyze_batch(articles)

        validated = [self._validate_result(r) for r in results]
        logger.info("Batch analiz tamamlandı: %d metin (%s).",
                     len(validated),
                     "LLM" if self._use_llm else "MockNLP")
        return validated

    def extract_and_save(self, articles: list[dict]) -> int:
        """
        Analiz yapar ve sonuçları DB'ye kaydeder.

        Returns
        -------
        int
            Kaydedilen skor sayısı.
        """
        results = self.extract_batch(articles)

        db_records = []
        for r in results:
            raw_date = r.get("date", "")
            if hasattr(raw_date, "strftime"):
                raw_date = raw_date.strftime("%Y-%m-%d")
            db_records.append({
                "news_id": r.get("news_id"),
                "date": str(raw_date),
                "event_type": r["event_type"],
                "uncertainty_score": r["uncertainty_score"],
                "impact_direction": r["impact_direction"],
                "model_used": r.get("model_used", "unknown"),
                "raw_llm_response": str(r),
            })

        return self._db.insert_scores(db_records)

    # ------------------------------------------------------------------
    # LLM analizi
    # ------------------------------------------------------------------

    def _llm_analyze(self, text: str, source: str, date: str) -> Optional[dict]:
        """Ollama üzerinden LLM ile analiz yapar."""
        prompt = build_user_prompt(text, source, date)
        parsed = self._ollama.generate_json(prompt, self._system_prompt)

        if parsed is None:
            return None

        parsed["model_used"] = self._config.get("model", "llama3")
        return parsed

    # ------------------------------------------------------------------
    # Doğrulama (Validation)
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_result(result: dict) -> dict:
        """
        LLM veya MockNLP çıktısını doğrular ve normalize eder.
        Eksik/hatalı alanları varsayılan değerlerle doldurur.
        """
        validated = {}

        score = result.get("uncertainty_score", 0.5)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0.5
        validated["uncertainty_score"] = round(np.clip(score, 0.0, 1.0), 4)

        direction = result.get("impact_direction", 0)
        try:
            direction = int(direction)
        except (TypeError, ValueError):
            direction = 0
        validated["impact_direction"] = max(-1, min(1, direction))

        event = result.get("event_type", "diger")
        validated["event_type"] = event if event in EVENT_TYPES else "diger"

        confidence = result.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        validated["confidence"] = round(np.clip(confidence, 0.0, 1.0), 2)

        validated["key_phrases"] = result.get("key_phrases", [])[:5]
        validated["reasoning"] = str(result.get("reasoning", ""))[:500]
        validated["model_used"] = result.get("model_used", "unknown")
        validated["news_id"] = result.get("news_id")
        validated["date"] = result.get("date", "")

        return validated

    # ------------------------------------------------------------------
    # DataFrame dönüşümü
    # ------------------------------------------------------------------

    def results_to_dataframe(self, results: list[dict]) -> pd.DataFrame:
        """Skor listesini zaman serisi DataFrame'ine dönüştürür."""
        df = pd.DataFrame(results)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)
            df.sort_index(inplace=True)
        return df

    # ------------------------------------------------------------------
    # İstatistikler
    # ------------------------------------------------------------------

    def summary(self, results: list[dict]) -> dict:
        """Analiz sonuçlarının özet istatistiklerini döner."""
        if not results:
            return {}

        scores = [r["uncertainty_score"] for r in results]
        directions = [r["impact_direction"] for r in results]
        events = [r["event_type"] for r in results]

        from collections import Counter
        event_dist = dict(Counter(events).most_common())

        return {
            "total": len(results),
            "avg_uncertainty": round(np.mean(scores), 4),
            "std_uncertainty": round(np.std(scores), 4),
            "min_uncertainty": round(min(scores), 4),
            "max_uncertainty": round(max(scores), 4),
            "avg_impact": round(np.mean(directions), 2),
            "negative_pct": round(sum(1 for d in directions if d < 0) / len(directions) * 100, 1),
            "event_distribution": event_dist,
            "model": "LLM" if self._use_llm else "MockNLP",
        }

    def __repr__(self) -> str:
        mode = "LLM" if self._use_llm else "MockNLP"
        return f"EventExtractor(mode={mode}, model={self._config.get('model', '?')})"
