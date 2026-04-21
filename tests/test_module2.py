"""
Modül 2 Entegrasyon Testi
NLP pipeline'ı uçtan uca test eder: MockNLP → EventExtractor → DB
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import logging
import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-20s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_module2")


def main():
    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    np.random.seed(config.get("seed", 42))

    print("=" * 70)
    print("  MODÜL 2 — NLP OLAY ÇIKARIMI TESTİ")
    print("=" * 70)

    # ---- 1. DB bağlantısı ----
    from src.pipeline.database import DatabaseManager
    db = DatabaseManager(ROOT / config["database"]["path"])
    counts = db.table_counts()
    print(f"\n[1] DB durumu: {counts}")

    # ---- 2. MockNLP tek metin testi ----
    from src.nlp.mock_nlp import MockNLP
    mock = MockNLP()

    test_texts = [
        "TCMB politika faizini artırdı, piyasalarda tedirginlik hakim.",
        "Ekonomi stabil, büyüme rakamları olumlu.",
        "Jeopolitik gerilimler sınırda tırmandı, çatışma riski artıyor.",
        "Sıradan bir işlem günü.",
    ]

    print("\n[2] MockNLP tek metin testleri:")
    for text in test_texts:
        result = mock.analyze(text)
        print(f"  Skor={result['uncertainty_score']:.2f} | "
              f"Yön={result['impact_direction']:+d} | "
              f"Olay={result['event_type']:30s} | "
              f"Metin: {text[:50]}...")

    # ---- 3. EventExtractor (force_mock=True) ----
    from src.nlp.event_extractor import EventExtractor
    extractor = EventExtractor(db, config["nlp"], force_mock=True)
    print(f"\n[3] {extractor}")

    # ---- 4. DB'deki haberleri çek ve analiz et ----
    news_df = db.get_news()
    if news_df.empty:
        print("  UYARI: DB'de haber yok. Önce test_module1.py çalıştırın.")
        db.close()
        return

    articles = news_df.to_dict("records")
    print(f"[4] DB'den {len(articles)} haber okundu, analiz başlıyor...")

    # ---- 5. Toplu analiz + DB'ye kaydet ----
    saved = extractor.extract_and_save(articles)
    print(f"[5] {saved} skor DB'ye kaydedildi.")

    # ---- 6. Özet istatistikler ----
    results = extractor.extract_batch(articles)
    summary = extractor.summary(results)
    print(f"\n[6] Analiz Özeti:")
    print(f"  Toplam metin     : {summary['total']}")
    print(f"  Ort. belirsizlik : {summary['avg_uncertainty']:.4f}")
    print(f"  Std. belirsizlik : {summary['std_uncertainty']:.4f}")
    print(f"  Min / Max        : {summary['min_uncertainty']:.4f} / {summary['max_uncertainty']:.4f}")
    print(f"  Negatif etki (%) : {summary['negative_pct']}%")
    print(f"  Olay dağılımı    : {summary['event_distribution']}")

    # ---- 7. Merged view kontrolü ----
    merged = db.get_merged_data(config["financial"]["default_ticker"])
    print(f"\n[7] Merged view: {len(merged)} satır")
    if not merged.empty:
        print(f"  avg_uncertainty ort: {merged['avg_uncertainty'].mean():.4f}")
        print(f"  news_count ort    : {merged['news_count'].mean():.1f}")

    # ---- 8. Final DB durumu ----
    counts = db.table_counts()
    print(f"\n{'=' * 70}")
    print(f"  DB Durumu: {counts}")
    print(f"{'=' * 70}")

    db.close()

    # ---- Assertions ----
    assert saved > 0, "Hiç skor kaydedilemedi!"
    assert summary["avg_uncertainty"] > 0, "Ortalama belirsizlik 0!"
    assert len(summary["event_distribution"]) > 1, "Tek olay türü var!"
    assert not merged.empty, "Merged view boş!"

    print("\n  TÜM TESTLER BAŞARILI ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
