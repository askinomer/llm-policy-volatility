"""
build_gold_set.py — DB'den dengeli 50 haberlik altın set (manual labeling) örnekler.

Kullanım:
    python scripts/build_gold_set.py [--n 50] [--output data/gold_set.csv]

Çıktı CSV'de doldurulacak kolonlar:
    gold_event_type, gold_uncertainty_score, gold_impact_direction

Altın set doldurulduktan sonra değerlendirme için:
    python scripts/evaluate_nlp.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml

from src.pipeline.database import DatabaseManager
from src.nlp.prompts import EVENT_TYPES


def stratified_sample(news_df: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    """Kaynak bazında dengeli örneklem alır (tcmb/fed/gdelt/yerel ~eşit)."""
    rng = np.random.default_rng(seed)
    sources = news_df["source"].unique().tolist()
    per_src = max(1, n // len(sources))
    pieces = []

    for src in sources:
        subset = news_df[news_df["source"] == src]
        k = min(per_src, len(subset))
        if k == 0:
            continue
        idx = rng.choice(subset.index.values, size=k, replace=False)
        pieces.append(subset.loc[idx])

    sampled = pd.concat(pieces).reset_index(drop=True)

    if len(sampled) < n:
        rest = news_df[~news_df["id"].isin(sampled["id"])]
        extra_k = min(n - len(sampled), len(rest))
        if extra_k > 0:
            extra_idx = rng.choice(rest.index.values, size=extra_k, replace=False)
            sampled = pd.concat([sampled, rest.loc[extra_idx]]).reset_index(drop=True)

    return sampled.head(n)


def main() -> None:
    parser = argparse.ArgumentParser(description="NLP altın set örnekleyici")
    parser.add_argument("--n", type=int, default=50, help="Örnek boyutu")
    parser.add_argument(
        "--output",
        type=str,
        default="data/gold_set.csv",
        help="Çıktı CSV yolu",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    db = DatabaseManager(ROOT / config["database"]["path"])
    news = db.get_news()
    db.close()

    if news.empty:
        print("DB'de haber yok. Önce pipeline'ı çalıştır.")
        sys.exit(1)

    print(f"Toplam haber: {len(news)}")
    print(f"Kaynak dağılımı: {news['source'].value_counts().to_dict()}")

    sampled = stratified_sample(news, args.n, args.seed)

    output = pd.DataFrame({
        "news_id": sampled["id"],
        "date": sampled["date"].dt.strftime("%Y-%m-%d") if hasattr(sampled["date"], "dt") else sampled["date"],
        "source": sampled["source"],
        "title": sampled["title"].fillna("").str.slice(0, 120),
        "content": sampled["content"].fillna("").str.slice(0, 500),
        "gold_event_type": "",
        "gold_uncertainty_score": "",
        "gold_impact_direction": "",
    })

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nAltın set şablonu yazıldı: {output_path}")
    print(f"Satır sayısı: {len(output)}")
    print(f"Kaynak dağılımı (örneklem): {output['source'].value_counts().to_dict()}")
    print(f"\nEtiketlemek için şu kolonları doldur:")
    print(f"  gold_event_type         : {EVENT_TYPES}")
    print(f"  gold_uncertainty_score  : 0.00 - 1.00 arası float")
    print(f"  gold_impact_direction   : -1 (negatif), 0 (nötr), 1 (pozitif)")
    print(f"\nSonra: python scripts/evaluate_nlp.py --gold {args.output}")


if __name__ == "__main__":
    main()
