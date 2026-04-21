"""
Modül 3 Entegrasyon Testi
DB'den veri çeker → 6 GARCH modeli eğitir → benchmark raporu üretir.
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
logger = logging.getLogger("test_module3")


def main():
    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    np.random.seed(config.get("seed", 42))

    print("=" * 80)
    print("  MODÜL 3 — FİNANSAL MODELLEME TESTİ")
    print("=" * 80)

    # ---- 1. DB'den merged veri çek ----
    from src.pipeline.database import DatabaseManager
    db = DatabaseManager(ROOT / config["database"]["path"])
    merged = db.get_merged_data(config["financial"]["default_ticker"])
    print(f"\n[1] Merged veri: {len(merged)} satır, kolonlar: {list(merged.columns)}")

    if merged.empty:
        print("  HATA: Merged veri boş. Önce test_module1 ve test_module2 çalıştırın.")
        db.close()
        return

    y = merged["log_return"].dropna()
    x = merged[["avg_uncertainty"]].loc[y.index]
    print(f"  Getiri serisi: {len(y)} gözlem")
    print(f"  Belirsizlik skoru ort: {x['avg_uncertainty'].mean():.4f}")

    # ---- 2. GARCHEngine — tüm modelleri eğit ----
    from src.models.garch_engine import GARCHEngine
    engine = GARCHEngine(
        p=config["garch"]["p"],
        q=config["garch"]["q"],
        dist=config["garch"].get("distribution", "normal"),
    )

    print(f"\n[2] Modeller eğitiliyor...")
    results = engine.fit_all(y, x)
    print(f"  {len(results)} model eğitildi: {list(results.keys())}")

    # ---- 3. Volatilite DataFrame ----
    vol_df = engine.volatility_dataframe()
    print(f"\n[3] Volatilite DataFrame: {vol_df.shape}")
    print(f"  Kolonlar: {list(vol_df.columns)}")

    # ---- 4. Forecast (5 gün) ----
    print(f"\n[4] 5 günlük volatilite tahmini (GARCH-X):")
    forecast = engine.forecast("GARCH-X", horizon=5)
    print(forecast.to_string())

    # ---- 5. Benchmark raporu ----
    from src.models.benchmark import ModelBenchmark
    benchmark = ModelBenchmark(
        engine=engine,
        returns=y,
        rolling_window=config["preprocessing"]["rolling_window"],
    )

    benchmark.print_report()

    best = benchmark.best_model("AIC")
    print(f"\n  Kazanan model: {best}")

    # ---- 6. Exog anlamlılığı ----
    exog_df = benchmark.exog_significance()
    if not exog_df.empty:
        print(f"\n[6] Dışsal değişken detayı:")
        print(exog_df.to_string(index=False))

    db.close()

    # ---- Assertions ----
    assert len(results) == 6, f"6 model bekleniyordu, {len(results)} geldi!"
    assert vol_df.shape[1] == 6, "Volatilite DataFrame 6 kolon olmalı!"
    assert len(forecast) == 5, "Forecast 5 satır olmalı!"
    assert best in results, "Best model sonuçlarda olmalı!"

    print("\n  TÜM TESTLER BAŞARILI ✓")
    print("=" * 80)


if __name__ == "__main__":
    main()
