"""
compare_nlp_scores.py — İki NLP modelinin aynı haberlerde verdiği skorları karşılaştır.

Kullanım:
    python scripts/compare_nlp_scores.py              # mock vs qwen2.5:3b
    python scripts/compare_nlp_scores.py --a mock --b llama3
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


def load_scores(db: DatabaseManager, model: str) -> pd.DataFrame:
    q = """
        SELECT news_id, event_type, uncertainty_score, impact_direction
        FROM nlp_scores WHERE model_used = ?
    """
    return pd.read_sql_query(q, db.connection, params=[model])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="mock", help="1. model (default: mock)")
    ap.add_argument("--b", default="qwen2.5:3b", help="2. model")
    args = ap.parse_args()

    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)
    db = DatabaseManager(ROOT / config["database"]["path"])

    a = load_scores(db, args.a)
    b = load_scores(db, args.b)
    print(f"\nModel A ({args.a:>12}): {len(a):>5} kayıt")
    print(f"Model B ({args.b:>12}): {len(b):>5} kayıt")

    merged = a.merge(b, on="news_id", suffixes=("_a", "_b"))
    n = len(merged)
    print(f"Ortak news_id         : {n:>5}")
    if n == 0:
        print("Karşılaştırılacak ortak haber yok.")
        db.close()
        return

    ua, ub = merged["uncertainty_score_a"].values, merged["uncertainty_score_b"].values
    diff = ub - ua
    pearson = np.corrcoef(ua, ub)[0, 1]
    spearman = pd.Series(ua).corr(pd.Series(ub), method="spearman")
    mae = np.mean(np.abs(diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    print(f"\n{'=' * 60}\n  UNCERTAINTY SCORE\n{'=' * 60}")
    print(f"  Ortalama A     : {ua.mean():.4f}   std: {ua.std():.4f}")
    print(f"  Ortalama B     : {ub.mean():.4f}   std: {ub.std():.4f}")
    print(f"  Ortalama fark  : {diff.mean():+.4f}")
    print(f"  MAE            : {mae:.4f}")
    print(f"  RMSE           : {rmse:.4f}")
    print(f"  Pearson r      : {pearson:+.4f}")
    print(f"  Spearman rho   : {spearman:+.4f}")

    ia = merged["impact_direction_a"].values
    ib = merged["impact_direction_b"].values
    agree = (ia == ib).mean()
    print(f"\n{'=' * 60}\n  IMPACT DIRECTION\n{'=' * 60}")
    print(f"  Anlaşma oranı  : {agree:.2%} ({(ia == ib).sum()}/{n})")
    print("  Dağılım A:", dict(pd.Series(ia).value_counts().sort_index()))
    print("  Dağılım B:", dict(pd.Series(ib).value_counts().sort_index()))
    cm = pd.crosstab(merged["impact_direction_a"], merged["impact_direction_b"],
                     rownames=[f"A={args.a}"], colnames=[f"B={args.b}"])
    print("  Confusion matrix:")
    print(cm.to_string())

    ea = merged["event_type_a"].values
    eb = merged["event_type_b"].values
    evt_agree = (ea == eb).mean()
    print(f"\n{'=' * 60}\n  EVENT TYPE\n{'=' * 60}")
    print(f"  Anlaşma oranı  : {evt_agree:.2%}")
    top_a = pd.Series(ea).value_counts().head(5)
    top_b = pd.Series(eb).value_counts().head(5)
    print(f"  Top 5 A        : {dict(top_a)}")
    print(f"  Top 5 B        : {dict(top_b)}")

    db.close()


if __name__ == "__main__":
    main()
