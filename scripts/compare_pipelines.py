"""
compare_pipelines.py — Farklı NLP modellerinin GARCH-X sonuçlarını kıyaslar.

Aynı finansal veri üzerinde, farklı `model_used` skorları ile pipeline'ı
koşup AIC/BIC/Vol_Corr ve OOS RMSE/MAE/QLIKE metriklerini tek tabloda
side-by-side gösterir.

Kullanım:
    python scripts/compare_pipelines.py
    python scripts/compare_pipelines.py --models mock qwen2.5:3b llama3
    python scripts/compare_pipelines.py --ticker XU100.IS --test-ratio 0.20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.database import DatabaseManager
from src.models.garch_engine import GARCHEngine
from src.models.benchmark import ModelBenchmark, OutOfSampleEvaluator


def run_for_model(
    db: DatabaseManager,
    ticker: str,
    nlp_model: str,
    garch_cfg: dict,
    rolling_window: int,
    test_ratio: float,
    skip_oos: bool = False,
) -> dict:
    merged = db.get_merged_data(ticker, nlp_model=nlp_model)
    if merged.empty:
        raise RuntimeError(f"Merged veri boş: {nlp_model}")

    y = merged["log_return"].dropna()
    x = merged[["avg_uncertainty"]].loc[y.index]

    engine = GARCHEngine(
        p=garch_cfg["p"],
        q=garch_cfg["q"],
        dist=garch_cfg.get("distribution", "normal"),
    )
    engine.fit_all(y, x)

    bench = ModelBenchmark(engine=engine, returns=y, rolling_window=rolling_window)
    in_sample = bench.comparison_table().set_index("Model")

    if skip_oos:
        oos_df = pd.DataFrame()
    else:
        oos = OutOfSampleEvaluator(
            returns=y, exog=x, test_ratio=test_ratio, garch_config=garch_cfg
        )
        oos_df = oos.evaluate_all().set_index("Model")

    return {
        "nlp_model": nlp_model,
        "avg_unc": float(merged["avg_uncertainty"].mean()),
        "unc_std": float(merged["avg_uncertainty"].std()),
        "news_days": int((merged["news_count"] > 0).sum()),
        "in_sample": in_sample,
        "oos": oos_df,
    }


def build_side_by_side(results: list[dict]) -> pd.DataFrame:
    rows = []
    models = ["GARCH", "GARCH-X", "EGARCH", "EGARCH-X", "TARCH", "TARCH-X"]
    for m in models:
        row = {"Model": m}
        for r in results:
            tag = r["nlp_model"]
            try:
                row[f"AIC[{tag}]"] = r["in_sample"].loc[m, "AIC"]
                row[f"VolCorr[{tag}]"] = r["in_sample"].loc[m, "Vol_Corr"]
            except KeyError:
                row[f"AIC[{tag}]"] = np.nan
                row[f"VolCorr[{tag}]"] = np.nan
            if not r["oos"].empty:
                try:
                    row[f"OOS_RMSE[{tag}]"] = r["oos"].loc[m, "RMSE"]
                    row[f"OOS_QLIKE[{tag}]"] = r["oos"].loc[m, "QLIKE"]
                except KeyError:
                    row[f"OOS_RMSE[{tag}]"] = np.nan
                    row[f"OOS_QLIKE[{tag}]"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description="NLP model karşılaştırması (GARCH pipeline)")
    p.add_argument("--models", nargs="+", default=["mock", "qwen2.5:3b"],
                   help="Karşılaştırılacak model_used değerleri")
    p.add_argument("--ticker", type=str, default=None)
    p.add_argument("--test-ratio", type=float, default=0.20)
    p.add_argument("--skip-oos", action="store_true")
    p.add_argument("--output", type=str, default="outputs/reports/nlp_compare.csv")
    return p.parse_args()


def main():
    args = parse_args()

    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    np.random.seed(config.get("seed", 42))
    ticker = args.ticker or config["financial"]["default_ticker"]

    db = DatabaseManager(ROOT / config["database"]["path"])
    try:
        results = []
        for m in args.models:
            print(f"\n[RUN] {m} için GARCH pipeline koşuluyor...")
            r = run_for_model(
                db=db,
                ticker=ticker,
                nlp_model=m,
                garch_cfg=config["garch"],
                rolling_window=config["preprocessing"]["rolling_window"],
                test_ratio=args.test_ratio,
                skip_oos=args.skip_oos,
            )
            print(f"  avg_uncertainty: {r['avg_unc']:.4f} (std={r['unc_std']:.4f}), "
                  f"{r['news_days']} haberli gün")
            results.append(r)

        print("\n" + "=" * 90)
        print("  ÖZET: NLP MODELLERİNİN GARCH PERFORMANSINA ETKİSİ")
        print("=" * 90)

        meta = pd.DataFrame([
            {
                "NLP Model": r["nlp_model"],
                "Avg Uncertainty": f"{r['avg_unc']:.4f}",
                "Std": f"{r['unc_std']:.4f}",
                "Haberli Gün": r["news_days"],
            }
            for r in results
        ])
        print("\nNLP Skor İstatistikleri:")
        print(meta.to_string(index=False))

        side = build_side_by_side(results)
        print("\nIn-Sample & OOS karşılaştırma:")
        with pd.option_context("display.float_format", "{:.4f}".format,
                               "display.max_columns", None,
                               "display.width", 160):
            print(side.to_string(index=False))

        out_path = ROOT / args.output
        out_path.parent.mkdir(parents=True, exist_ok=True)
        side.to_csv(out_path, index=False)
        print(f"\n[SAVE] {out_path}")

        print("\nGözlem:")
        if len(results) == 2:
            a, b = results
            def _delta(model, metric, frame="in_sample"):
                try:
                    va = a[frame].loc[model, metric]
                    vb = b[frame].loc[model, metric]
                    return va - vb
                except Exception:
                    return np.nan

            print(f"  GARCH-X AIC farkı   ({a['nlp_model']} - {b['nlp_model']}): "
                  f"{_delta('GARCH-X', 'AIC'):+.3f}")
            print(f"  TARCH-X AIC farkı   ({a['nlp_model']} - {b['nlp_model']}): "
                  f"{_delta('TARCH-X', 'AIC'):+.3f}")
            print(f"  EGARCH-X AIC farkı  ({a['nlp_model']} - {b['nlp_model']}): "
                  f"{_delta('EGARCH-X', 'AIC'):+.3f}")

            if not a["oos"].empty and not b["oos"].empty:
                print(f"  GARCH-X OOS RMSE farkı : "
                      f"{_delta('GARCH-X', 'RMSE', 'oos'):+.4f}")
                print(f"  EGARCH-X OOS RMSE farkı: "
                      f"{_delta('EGARCH-X', 'RMSE', 'oos'):+.4f}")
                print(f"  TARCH-X OOS RMSE farkı : "
                      f"{_delta('TARCH-X', 'RMSE', 'oos'):+.4f}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
