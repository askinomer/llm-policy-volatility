"""
Modül 4 Entegrasyon Testi
DB → GARCH eğitim → Dashboard + XAI raporları üretir.
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
logger = logging.getLogger("test_module4")


def main():
    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    np.random.seed(config.get("seed", 42))

    print("=" * 70)
    print("  MODÜL 4 — GÖRSELLEŞTİRME ve XAI TESTİ")
    print("=" * 70)

    # ---- 1. DB'den veri ----
    from src.pipeline.database import DatabaseManager
    db = DatabaseManager(ROOT / config["database"]["path"])
    merged = db.get_merged_data(config["financial"]["default_ticker"])
    print(f"\n[1] Merged veri: {len(merged)} satır")

    # ---- 2. GARCH modelleri eğit ----
    from src.models.garch_engine import GARCHEngine
    from src.models.benchmark import ModelBenchmark

    y = merged["log_return"].dropna()
    x = merged[["avg_uncertainty"]].loc[y.index]

    engine = GARCHEngine(p=1, q=1)
    engine.fit_all(y, x)
    print(f"[2] {len(engine.results)} model eğitildi")

    benchmark = ModelBenchmark(engine, y, rolling_window=20)
    bench_table = benchmark.comparison_table()

    # Volatilite kolonlarını merged'e ekle
    vol_df = engine.volatility_dataframe()
    for col in vol_df.columns:
        merged.loc[vol_df.index, col] = vol_df[col]

    # ---- 3. Plotly Dashboard ----
    from src.visualization.plotly_dashboard import InteractiveDashboard

    dashboard = InteractiveDashboard(merged, bench_table)
    html_path = dashboard.save_html(str(ROOT / "outputs/figures/dashboard.html"))
    print(f"[3] Dashboard HTML: {html_path}")

    # ---- 4. XAI Raporları ----
    from src.visualization.explainability import VolatilityExplainer

    explainer = VolatilityExplainer(
        df=merged,
        model_results=engine.results,
        output_dir=str(ROOT / "outputs/figures"),
    )

    paths = explainer.generate_all("GARCH-X")
    print(f"[4] XAI grafikleri:")
    for name, path in paths.items():
        print(f"    {name}: {path}")

    explainer.print_summary("GARCH-X")

    # ---- 5. Sensitivity testi ----
    sens = explainer.sensitivity_analysis("GARCH-X")
    print(f"\n[5] Sensitivity tablosu: {len(sens)} satır")

    events = explainer.event_volatility_map("GARCH-X", top_n=5)
    print(f"[6] Top 5 volatilite zıplaması: {len(events)} satır")

    db.close()

    # ---- Assertions ----
    assert Path(html_path).exists(), "Dashboard HTML oluşturulmadı!"
    assert all(Path(p).exists() for p in paths.values() if p), "XAI grafikleri eksik!"
    assert len(sens) > 0, "Sensitivity tablosu boş!"
    assert len(events) > 0, "Event map boş!"

    print("\n  TÜM TESTLER BAŞARILI ✓")
    print("=" * 70)


if __name__ == "__main__":
    main()
