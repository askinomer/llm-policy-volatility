"""
main.py — Tüm Pipeline'ı Tek Komutla Çalıştırır

Kullanım:
    python3 main.py                     # Tam pipeline (gerçek TCMB+FED scraper)
    python3 main.py --mock-only         # Sadece mock haber kullan
    python3 main.py --skip-fetch        # Veri çekmeden (DB'deki veriyle)
    python3 main.py --skip-nlp          # NLP adımını atla
    python3 main.py --ticker "^GSPC"    # Farklı ticker
    python3 main.py --force-mock        # Ollama varsa bile MockNLP kullan
    python3 main.py --forecast 10       # 10 günlük tahmin
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)-20s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args():
    p = argparse.ArgumentParser(description="LLM Tabanlı Politika Belirsizliği Pipeline")
    p.add_argument("--ticker", type=str, default=None, help="Ticker (varsayılan: config'deki default)")
    p.add_argument("--skip-fetch", action="store_true", help="Finansal veri + haber çekmeyi atla")
    p.add_argument("--skip-nlp", action="store_true", help="NLP olay çıkarımını atla")
    p.add_argument("--skip-viz", action="store_true", help="Görselleştirme adımını atla")
    p.add_argument("--force-mock", action="store_true", help="Ollama yerine MockNLP kullan")
    p.add_argument("--forecast", type=int, default=5, help="Forecast horizon (gün)")
    p.add_argument("--no-dashboard", action="store_true", help="Dashboard üretme")
    p.add_argument("--mock-only", action="store_true", help="Gerçek scraper yerine sadece mock haber kullan")
    p.add_argument("--test-ratio", type=float, default=0.20, help="OOS test seti oranı (varsayılan: 0.20)")
    p.add_argument("--skip-oos", action="store_true", help="Out-of-sample değerlendirmeyi atla")
    p.add_argument("--nlp-model", type=str, default=None,
                   help="GARCH-X için kullanılacak NLP modeli (ör. qwen2.5:3b, llama3, mock)")
    return p.parse_args()


def banner(text: str, width: int = 70):
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def step_header(num, text: str):
    print(f"\n{'─' * 50}")
    print(f"  ADIM {num}: {text}")
    print(f"{'─' * 50}")


def run():
    args = parse_args()
    t0 = time.time()

    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    np.random.seed(config.get("seed", 42))
    ticker = args.ticker or config["financial"]["default_ticker"]

    banner("LLM TABANLI POLİTİKA BELİRSİZLİĞİ VE FİNANSAL VOLATİLİTE")
    print(f"  Ticker  : {ticker}")
    print(f"  Forecast: {args.forecast} gün")
    print(f"  Adımlar : fetch={'ATLA' if args.skip_fetch else 'OK'} | "
          f"nlp={'ATLA' if args.skip_nlp else 'OK'} | "
          f"viz={'ATLA' if args.skip_viz else 'OK'}")

    # ── ADIM 1: Veritabanı ─────────────────────────────────────────────
    step_header(1, "VERİTABANI BAĞLANTISI")
    from src.pipeline.database import DatabaseManager

    db_path = ROOT / config["database"]["path"]
    db = DatabaseManager(db_path)
    counts = db.table_counts()
    print(f"  DB: {db_path}")
    print(f"  Mevcut kayıtlar: {counts}")

    # ── ADIM 2: Finansal Veri + Haber Çekme ────────────────────────────
    if not args.skip_fetch:
        step_header(2, "VERİ ÇEKME")
        from src.pipeline.fetcher import (
            FinancialDataFetcher,
            MockNewsFetcher,
            TCMBNewsFetcher,
            FEDNewsFetcher,
            GDELTNewsFetcher,
        )

        fin_fetcher = FinancialDataFetcher(db, config["financial"])
        fin_df = fin_fetcher.fetch_and_save(ticker=ticker)
        print(f"  Finansal: {len(fin_df)} satır "
              f"({fin_df.index[0].date()} → {fin_df.index[-1].date()})")
        print(f"  Log Return: ort={fin_df['log_return'].mean():.4f}, "
              f"std={fin_df['log_return'].std():.4f}")

        start = fin_df.index[0].strftime("%Y-%m-%d")
        end = fin_df.index[-1].strftime("%Y-%m-%d")

        total_news = 0

        if args.mock_only:
            print("\n  [MOCK] --mock-only modu, gerçek scraper atlanıyor...")
            mock_fetcher = MockNewsFetcher(db, config["news"])
            mock_ids = mock_fetcher.fetch_and_save(start, end)
            print(f"  [MOCK] {len(mock_ids)} mock haber eklendi")
            total_news = len(mock_ids)
        else:
            # --- TCMB gerçek scraper ---
            print("\n  [TCMB] Basın duyuruları çekiliyor...")
            tcmb_fetcher = TCMBNewsFetcher(db, config["news"])
            tcmb_ids = tcmb_fetcher.fetch_and_save(start, end)
            print(f"  [TCMB] {len(tcmb_ids)} haber DB'ye yazıldı")
            total_news += len(tcmb_ids)

            # --- FED gerçek scraper ---
            print("\n  [FED]  Basın duyuruları çekiliyor...")
            fed_fetcher = FEDNewsFetcher(db, config["news"])
            fed_ids = fed_fetcher.fetch_and_save(start, end)
            print(f"  [FED]  {len(fed_ids)} haber DB'ye yazıldı")
            total_news += len(fed_ids)

            # --- GDELT API ---
            print("\n  [GDELT] Türkiye ekonomi haberleri çekiliyor...")
            gdelt_fetcher = GDELTNewsFetcher(db, config["news"])
            gdelt_ids = gdelt_fetcher.fetch_and_save(start, end)
            print(f"  [GDELT] {len(gdelt_ids)} haber DB'ye yazıldı")
            total_news += len(gdelt_ids)

            # --- Mock fallback ---
            min_articles = config["news"].get("min_articles_fallback", 100)
            if total_news < min_articles:
                print(f"\n  [MOCK] Gerçek haber sayısı ({total_news}) < "
                      f"minimum ({min_articles}), mock ile tamamlanıyor...")
                mock_fetcher = MockNewsFetcher(db, config["news"])
                mock_ids = mock_fetcher.fetch_and_save(start, end)
                print(f"  [MOCK] {len(mock_ids)} mock haber eklendi")
                total_news += len(mock_ids)

        print(f"\n  Toplam haber: {total_news} kayıt DB'de")
    else:
        step_header(2, "VERİ ÇEKME [ATLANDI]")

    # ── ADIM 3: NLP Olay Çıkarımı ─────────────────────────────────────
    if not args.skip_nlp:
        step_header(3, "NLP OLAY ÇIKARIMI")
        from src.nlp.event_extractor import EventExtractor

        extractor = EventExtractor(db, config["nlp"], force_mock=args.force_mock)
        news_df = db.get_news()

        if news_df.empty:
            logger.warning("DB'de haber yok! NLP adımı atlanıyor.")
        else:
            articles = news_df.to_dict("records")
            saved = extractor.extract_and_save(articles)
            print(f"  {saved} skor DB'ye kaydedildi")

            results = extractor.extract_batch(articles)
            summary = extractor.summary(results)
            print(f"  Ort. belirsizlik : {summary['avg_uncertainty']:.4f}")
            print(f"  Olay türü sayısı : {len(summary['event_distribution'])}")
    else:
        step_header(3, "NLP OLAY ÇIKARIMI [ATLANDI]")

    # ── ADIM 4: Veri Önişleme ──────────────────────────────────────────
    step_header(4, "VERİ ÖNİŞLEME + BİRLEŞTİRME")
    from src.pipeline.preprocessor import DataPreprocessor

    merged = db.get_merged_data(ticker, nlp_model=args.nlp_model)
    if args.nlp_model:
        print(f"  NLP modeli  : {args.nlp_model} (sadece bu skorlar kullanılıyor)")
    if merged.empty:
        logger.error("Merged veri boş! Pipeline durduruluyor.")
        db.close()
        sys.exit(1)

    preprocessor = DataPreprocessor(config["preprocessing"])
    processed = preprocessor.run(
        financial_df=merged,
        outlier_columns=["log_return"],
    )
    print(f"  Merged veri  : {len(merged)} satır")
    print(f"  Processed    : {len(processed)} satır, {len(processed.columns)} kolon")
    print(f"  avg_uncertainty ort: {merged['avg_uncertainty'].mean():.4f}")

    # ── ADIM 5: GARCH Modelleme ────────────────────────────────────────
    step_header(5, "GARCH MODELLEME")
    from src.models.garch_engine import GARCHEngine
    from src.models.benchmark import ModelBenchmark

    y = merged["log_return"].dropna()
    x = merged[["avg_uncertainty"]].loc[y.index]

    engine = GARCHEngine(
        p=config["garch"]["p"],
        q=config["garch"]["q"],
        dist=config["garch"].get("distribution", "normal"),
    )
    model_results = engine.fit_all(y, x)
    print(f"  {len(model_results)} model eğitildi: {list(model_results.keys())}")

    benchmark = ModelBenchmark(
        engine=engine,
        returns=y,
        rolling_window=config["preprocessing"]["rolling_window"],
    )
    best = benchmark.best_model("AIC")
    bench_table = benchmark.comparison_table()
    print(f"\n  Model Karşılaştırma:")
    print(bench_table.to_string(index=False))
    print(f"\n  En iyi model (AIC): {best}")

    # ── ADIM 5b: Out-of-Sample Değerlendirme ────────────────────────────
    if not args.skip_oos:
        step_header("5b", "OUT-OF-SAMPLE DEĞERLENDİRME")
        from src.models.benchmark import OutOfSampleEvaluator

        oos = OutOfSampleEvaluator(
            returns=y,
            exog=x,
            test_ratio=args.test_ratio,
            garch_config=config["garch"],
        )
        print(f"  Train: {oos.train_size} gün | Test: {oos.test_size} gün")
        print(f"  Split tarihi: {oos.split_date}")
        print(f"  Yöntem: Expanding window, 1-step-ahead forecast\n")

        oos_table = oos.evaluate_all()
        oos.print_report(oos_table)
    else:
        step_header("5b", "OUT-OF-SAMPLE [ATLANDI]")

    # ── ADIM 6: Forecast ───────────────────────────────────────────────
    step_header(6, f"VOLATİLİTE TAHMİNİ ({args.forecast} GÜN)")
    for name in model_results:
        try:
            fc = engine.forecast(name, horizon=args.forecast)
            last_vol = fc["volatility"].iloc[-1]
            print(f"  {name:12s} → son gün vol: {last_vol:.6f}")
        except Exception as e:
            print(f"  {name:12s} → forecast hatası: {e}")

    # ── ADIM 7: Görselleştirme + XAI ──────────────────────────────────
    if not args.skip_viz:
        step_header(7, "GÖRSELLEŞTİRME + XAI")

        vol_df = engine.volatility_dataframe()
        for col in vol_df.columns:
            merged.loc[vol_df.index, col] = vol_df[col]

        if not args.no_dashboard:
            from src.visualization.plotly_dashboard import InteractiveDashboard

            dashboard = InteractiveDashboard(merged, bench_table)
            html_path = dashboard.save_html(
                str(ROOT / config["output"]["figures_dir"] / "dashboard.html")
            )
            print(f"  Dashboard: {html_path}")

        from src.visualization.explainability import VolatilityExplainer

        explainer = VolatilityExplainer(
            df=merged,
            model_results=engine.results,
            output_dir=str(ROOT / config["output"]["figures_dir"]),
        )
        xai_model = "GARCH-X" if "GARCH-X" in engine.results else best
        paths = explainer.generate_all(xai_model)
        print(f"  XAI grafikleri ({xai_model}):")
        for name, path in paths.items():
            print(f"    {name}: {path}")

        explainer.print_summary(xai_model)
    else:
        step_header(7, "GÖRSELLEŞTİRME [ATLANDI]")

    # ── ÖZET ───────────────────────────────────────────────────────────
    final_counts = db.table_counts()
    db.close()
    elapsed = time.time() - t0

    banner("PİPELINE TAMAMLANDI")
    print(f"  DB kayıtları : {final_counts}")
    print(f"  En iyi model : {best}")
    print(f"  Toplam süre  : {elapsed:.1f} saniye")


if __name__ == "__main__":
    run()
