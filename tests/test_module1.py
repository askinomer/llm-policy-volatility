"""
Modül 1 Entegrasyon Testi
Tüm pipeline'ı uçtan uca çalıştırır ve sonuçları doğrular.
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
logger = logging.getLogger("test_module1")


def main():
    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    np.random.seed(config.get("seed", 42))

    print("=" * 70)
    print("  MODÜL 1 — VERİ BORU HATTI TESTİ")
    print("=" * 70)

    # ---- 1. DatabaseManager ----
    from src.pipeline.database import DatabaseManager

    db_path = ROOT / config["database"]["path"]
    db = DatabaseManager(db_path)
    print(f"\n[1] DB oluşturuldu: {db}")

    # ---- 2. FinancialDataFetcher ----
    from src.pipeline.fetcher import FinancialDataFetcher

    fin_fetcher = FinancialDataFetcher(db, config["financial"])
    fin_df = fin_fetcher.fetch_and_save()
    print(f"[2] Finansal veri: {len(fin_df)} satır")
    print(f"    Kolonlar: {list(fin_df.columns)}")
    print(f"    Tarih aralığı: {fin_df.index[0].date()} → {fin_df.index[-1].date()}")
    print(f"    Log Return → ort: {fin_df['log_return'].mean():.4f}, "
          f"std: {fin_df['log_return'].std():.4f}")

    # ---- 3. MockNewsFetcher ----
    from src.pipeline.fetcher import MockNewsFetcher

    news_fetcher = MockNewsFetcher(db, config["news"])
    start = fin_df.index[0].strftime("%Y-%m-%d")
    end = fin_df.index[-1].strftime("%Y-%m-%d")
    news_ids = news_fetcher.fetch_and_save(start, end)
    print(f"[3] Haberler: {len(news_ids)} kayıt DB'ye yazıldı")

    # ---- 4. DataPreprocessor ----
    from src.pipeline.preprocessor import DataPreprocessor

    preprocessor = DataPreprocessor(config["preprocessing"])

    news_df = db.get_news(start, end)
    print(f"[4] DB'den {len(news_df)} haber okundu")

    processed = preprocessor.run(
        financial_df=fin_df,
        outlier_columns=["log_return"],
    )
    print(f"[5] Preprocessed: {len(processed)} satır, {len(processed.columns)} kolon")
    print(f"    Kolonlar: {list(processed.columns)}")

    # ---- 5. Merged view testi ----
    merged = db.get_merged_data(config["financial"]["default_ticker"])
    print(f"[6] DB merged view: {len(merged)} satır")

    # ---- 6. Özet ----
    counts = db.table_counts()
    print(f"\n{'=' * 70}")
    print(f"  DB Durumu: {counts}")
    print(f"  {db}")
    print(f"{'=' * 70}")

    db.close()

    # ---- Assertions ----
    assert len(fin_df) > 0, "Finansal veri boş!"
    assert "log_return" in fin_df.columns, "log_return kolonu yok!"
    assert len(news_ids) > 0, "Haber kaydedilemedi!"
    assert "rolling_vol" in processed.columns, "Feature engineering çalışmadı!"
    assert len(merged) > 0, "Merged view boş!"

    print("\n  TÜM TESTLER BAŞARILI ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
