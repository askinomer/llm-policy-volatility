"""
evaluate_vs_reference.py — Bir modeli "referans altın" sayarak diğerlerini
onun üzerinden değerlendirir (elle etiketli set henüz tamamlanmadığı için
yarı-otomatik değerlendirme).

Örn: Llama3:8b (en yavaş ama en tutarlı) referans, Mock ve Qwen aday.

Metrikler:
    uncertainty_score -> Pearson, Spearman, MAE, RMSE
    impact_direction  -> Accuracy, macro-F1, confusion matrix
    event_type        -> Accuracy, macro-F1, top-5 confusion

Kullanım:
    python scripts/evaluate_vs_reference.py
    python scripts/evaluate_vs_reference.py --ref llama3 --candidates mock qwen2.5:3b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.database import DatabaseManager


def load_scores(db: DatabaseManager, model: str) -> pd.DataFrame:
    q = """
        SELECT news_id, event_type, uncertainty_score, impact_direction
        FROM nlp_scores WHERE model_used = ?
    """
    return pd.read_sql_query(q, db.connection, params=[model])


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return {"n": len(y_true)}
    pearson_r, _ = stats.pearsonr(y_true, y_pred)
    spearman_r, _ = stats.spearmanr(y_true, y_pred)
    return {
        "n": len(y_true),
        "pearson_r": round(float(pearson_r), 4),
        "spearman_rho": round(float(spearman_r), 4),
        "MAE": round(float(np.mean(np.abs(y_true - y_pred))), 4),
        "RMSE": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = pd.notna(y_true) & pd.notna(y_pred)
    y_true, y_pred = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    if len(y_true) == 0:
        return {"n": 0}
    acc = float((y_true == y_pred).mean())
    labels = sorted(set(y_true) | set(y_pred))
    f1s = []
    for lbl in labels:
        tp = int(((y_true == lbl) & (y_pred == lbl)).sum())
        fp = int(((y_true != lbl) & (y_pred == lbl)).sum())
        fn = int(((y_true == lbl) & (y_pred != lbl)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
    return {
        "n": len(y_true),
        "accuracy": round(acc, 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "n_classes": len(labels),
    }


def evaluate_pair(ref: pd.DataFrame, cand: pd.DataFrame, cand_name: str, ref_name: str) -> dict:
    merged = ref.merge(cand, on="news_id", suffixes=("_ref", "_cand"))
    if merged.empty:
        return {"candidate": cand_name, "n": 0}

    print(f"\n{'-' * 70}")
    print(f"  {cand_name}  vs  {ref_name}  (n={len(merged)} ortak haber)")
    print(f"{'-' * 70}")

    unc = regression_metrics(
        merged["uncertainty_score_ref"].values,
        merged["uncertainty_score_cand"].values,
    )
    print(f"\n  [uncertainty_score]")
    for k, v in unc.items():
        print(f"    {k:14s}: {v}")

    imp = classification_metrics(
        merged["impact_direction_ref"].values,
        merged["impact_direction_cand"].values,
    )
    print(f"\n  [impact_direction]")
    for k, v in imp.items():
        print(f"    {k:14s}: {v}")
    cm_imp = pd.crosstab(
        merged["impact_direction_ref"], merged["impact_direction_cand"],
        rownames=[f"REF={ref_name}"], colnames=[f"CAND={cand_name}"],
    )
    print("  confusion:")
    print("  " + cm_imp.to_string().replace("\n", "\n  "))

    evt = classification_metrics(
        merged["event_type_ref"].values,
        merged["event_type_cand"].values,
    )
    print(f"\n  [event_type]")
    for k, v in evt.items():
        print(f"    {k:14s}: {v}")

    return {
        "candidate": cand_name,
        "n": len(merged),
        "unc_pearson": unc.get("pearson_r"),
        "unc_MAE": unc.get("MAE"),
        "impact_acc": imp.get("accuracy"),
        "impact_f1": imp.get("macro_f1"),
        "event_acc": evt.get("accuracy"),
        "event_f1": evt.get("macro_f1"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="llama3",
                    help="Referans model (default: llama3)")
    ap.add_argument("--candidates", nargs="+", default=["mock", "qwen2.5:3b"],
                    help="Değerlendirilecek adaylar")
    ap.add_argument("--output", default="outputs/reports/eval_vs_reference.csv")
    args = ap.parse_args()

    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    db = DatabaseManager(ROOT / config["database"]["path"])
    try:
        ref_scores = load_scores(db, args.ref)
        if ref_scores.empty:
            print(f"[ERROR] Referans '{args.ref}' için DB'de skor yok.")
            sys.exit(1)

        print(f"Referans: {args.ref} ({len(ref_scores)} skor)")

        summary_rows = []
        for cand in args.candidates:
            cand_scores = load_scores(db, cand)
            if cand_scores.empty:
                print(f"[SKIP] {cand} için skor yok.")
                continue
            row = evaluate_pair(ref_scores, cand_scores, cand, args.ref)
            summary_rows.append(row)

        if summary_rows:
            print(f"\n{'=' * 70}")
            print(f"  ÖZET: Referans = {args.ref}")
            print(f"{'=' * 70}")
            df = pd.DataFrame(summary_rows)
            print(df.to_string(index=False))

            out = ROOT / args.output
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out, index=False)
            print(f"\n[SAVE] {out}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
