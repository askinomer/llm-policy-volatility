"""
LLM Prompt Şablonları — Yapılandırılmış JSON çıktısı için system/user prompt'lar.

Ollama (Llama 3) veya uyumlu herhangi bir LLM ile kullanılmak üzere tasarlanmıştır.
Prompt'lar kesin JSON schema zorlayarak hallucination'ı minimize eder.
"""

from __future__ import annotations

EVENT_TYPES = [
    "faiz_artisi",
    "faiz_indirimi",
    "enflasyon_aciklamasi",
    "doviz_mudahalesi",
    "jeopolitik_kriz",
    "secim_belirsizligi",
    "ticaret_savasi",
    "kredi_notu_degisikligi",
    "mali_politika",
    "merkez_bankasi_aciklamasi",
    "istihdam_verisi",
    "buyume_verisi",
    "diger",
]

SYSTEM_PROMPT = """Sen bir finansal metin analiz uzmanısın. Görevin, verilen haber metninden
ekonomik politika olaylarını çıkarmak ve belirsizlik seviyesini ölçmektir.

KURALLAR:
1. Yanıtını YALNIZCA aşağıdaki JSON formatında ver, başka hiçbir şey yazma.
2. uncertainty_score: Metindeki politik/ekonomik belirsizlik seviyesi (0.00 = kesin, 1.00 = çok belirsiz).
3. impact_direction: Piyasa üzerindeki beklenen etki (-1 = negatif, 0 = nötr, 1 = pozitif).
4. event_type: Aşağıdaki listeden EN UYGUN olanı seç:
   {event_types}
5. confidence: Analizine ne kadar güvendiğin (0.0 - 1.0).
6. key_phrases: Kararını etkileyen en önemli 3 kelime/ifade.

JSON FORMATI:
{{
    "event_type": "string",
    "uncertainty_score": float,
    "impact_direction": int,
    "confidence": float,
    "key_phrases": ["string", "string", "string"],
    "reasoning": "string (1 cümle)"
}}"""

USER_PROMPT_TEMPLATE = """Aşağıdaki haber metnini analiz et:

KAYNAK: {source}
TARİH: {date}
METİN: {text}

JSON yanıtını ver:"""

BATCH_USER_PROMPT_TEMPLATE = """Aşağıdaki {count} haber metnini TEK TEK analiz et.
Her biri için ayrı bir JSON nesnesi üret ve sonuçları bir JSON dizisi (array) olarak döndür.

{articles}

JSON dizisi yanıtını ver:"""


def build_system_prompt() -> str:
    """Event type listesiyle doldurulmuş system prompt döner."""
    return SYSTEM_PROMPT.format(event_types=", ".join(EVENT_TYPES))


def build_user_prompt(text: str, source: str = "unknown", date: str = "") -> str:
    """Tek bir haber için user prompt oluşturur."""
    return USER_PROMPT_TEMPLATE.format(source=source, date=date, text=text)


def build_batch_prompt(articles: list[dict]) -> str:
    """
    Birden fazla haber için toplu prompt oluşturur.

    Parameters
    ----------
    articles : list[dict]
        Her dict: text, source, date
    """
    parts = []
    for i, a in enumerate(articles, 1):
        parts.append(f"[{i}] KAYNAK: {a.get('source', '?')} | "
                     f"TARİH: {a.get('date', '?')}\n{a['text']}")

    return BATCH_USER_PROMPT_TEMPLATE.format(
        count=len(articles),
        articles="\n\n".join(parts),
    )
