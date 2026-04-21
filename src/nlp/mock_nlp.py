"""
MockNLP — LLM olmadan rule-based belirsizlik skoru üretici.

Ollama erişilemez olduğunda veya test/demo modunda kullanılır.
Ağırlıklı keyword sözlüğü ile deterministik skor hesaplar.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

KEYWORD_SCORES: dict[str, tuple[float, int]] = {
    # (belirsizlik_skoru, impact_direction)
    # Yüksek belirsizlik — negatif etki
    "tedirgin": (0.85, -1), "gerilim": (0.90, -1), "belirsizlik": (0.80, -1),
    "kriz": (0.95, -1), "çatışma": (0.92, -1), "baskı": (0.75, -1),
    "risk": (0.78, -1), "eridi": (0.88, -1), "düşüş": (0.72, -1),
    "şahin": (0.70, -1), "temkinli": (0.68, -1), "endişe": (0.82, -1),
    "tırmandı": (0.88, -1), "olumsuz": (0.76, -1), "daraltma": (0.74, -1),
    "zayıfladı": (0.73, -1), "sorgulanıyor": (0.71, -1), "çıkış": (0.74, -1),
    "enflasyon": (0.65, -1), "açık": (0.62, -1), "küçültme": (0.67, -1),
    # Düşük belirsizlik — pozitif etki
    "stabil": (0.15, 1), "yolunda": (0.12, 1), "olumlu": (0.18, 1),
    "güçlü": (0.20, 1), "iyileşiyor": (0.10, 1), "rekor": (0.14, 1),
    "rahatlatıcı": (0.08, 1), "büyüme": (0.22, 1), "teyit": (0.16, 1),
    "kontrol": (0.19, 1), "toparlanma": (0.13, 1), "iyimser": (0.11, 1),
}

EVENT_KEYWORD_MAP: dict[str, list[str]] = {
    "faiz_artisi": ["faiz", "artırdı", "sıkılaştırma", "baz puan"],
    "faiz_indirimi": ["faiz", "indirdi", "gevşeme", "indirim"],
    "enflasyon_aciklamasi": ["enflasyon", "tüfe", "fiyat artışı"],
    "doviz_mudahalesi": ["döviz", "kur", "swap", "rezerv", "müdahale"],
    "jeopolitik_kriz": ["jeopolitik", "gerilim", "çatışma", "sınır", "savaş"],
    "secim_belirsizligi": ["seçim", "oy", "koalisyon", "siyasi"],
    "kredi_notu_degisikligi": ["kredi notu", "derecelendirme", "moody", "fitch"],
    "mali_politika": ["bütçe", "vergi", "mali", "harcama"],
    "merkez_bankasi_aciklamasi": ["merkez bankası", "tcmb", "fed", "powell", "ppk"],
    "istihdam_verisi": ["istihdam", "işsizlik", "iş gücü"],
    "buyume_verisi": ["büyüme", "gsyh", "gdp", "sanayi üretimi"],
}


class MockNLP:
    """
    Rule-based NLP motoru. LLM'in yapacağı işi keyword matching ile simüle eder.

    Parameters
    ----------
    noise_std : float
        Skora eklenecek Gaussian gürültü standart sapması (varsayılan 0.03).
    """

    def __init__(self, noise_std: float = 0.03) -> None:
        self._noise_std = noise_std

    def analyze(self, text: str, source: str = "unknown", date: str = "") -> dict:
        """
        Tek bir metin için belirsizlik analizi yapar.

        Returns
        -------
        dict
            LLM çıktısıyla aynı formatta:
            event_type, uncertainty_score, impact_direction, confidence,
            key_phrases, reasoning, model_used
        """
        text_lower = text.lower()

        matched_scores = []
        matched_impacts = []
        key_phrases = []

        for keyword, (score, direction) in KEYWORD_SCORES.items():
            if keyword in text_lower:
                matched_scores.append(score)
                matched_impacts.append(direction)
                key_phrases.append(keyword)

        if matched_scores:
            base_score = float(np.mean(matched_scores))
            noise = np.random.normal(0, self._noise_std)
            uncertainty = float(np.clip(base_score + noise, 0.0, 1.0))
            impact = int(np.sign(np.mean(matched_impacts)))
            confidence = min(0.5 + len(matched_scores) * 0.1, 0.95)
        else:
            uncertainty = float(np.random.uniform(0.35, 0.55))
            impact = 0
            confidence = 0.3
            key_phrases = ["belirsiz"]

        event_type = self._detect_event_type(text_lower)

        return {
            "event_type": event_type,
            "uncertainty_score": round(uncertainty, 4),
            "impact_direction": impact,
            "confidence": round(confidence, 2),
            "key_phrases": key_phrases[:3],
            "reasoning": f"Rule-based: {len(matched_scores)} keyword eşleşti.",
            "model_used": "mock",
        }

    def analyze_batch(self, articles: list[dict]) -> list[dict]:
        """
        Birden fazla haber için toplu analiz.

        Parameters
        ----------
        articles : list[dict]
            Her dict: content (veya text), source, date
        """
        results = []
        for article in articles:
            text = article.get("content") or article.get("text", "")
            result = self.analyze(
                text=text,
                source=article.get("source", "unknown"),
                date=article.get("date", ""),
            )
            result["news_id"] = article.get("id")
            raw_date = article.get("date", "")
            if hasattr(raw_date, "strftime"):
                raw_date = raw_date.strftime("%Y-%m-%d")
            result["date"] = str(raw_date)
            results.append(result)

        logger.info("MockNLP: %d metin analiz edildi.", len(results))
        return results

    @staticmethod
    def _detect_event_type(text_lower: str) -> str:
        """Metindeki anahtar kelimelere göre olay türü belirler."""
        best_match = "diger"
        best_count = 0

        for event_type, keywords in EVENT_KEYWORD_MAP.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            if count > best_count:
                best_count = count
                best_match = event_type

        return best_match
