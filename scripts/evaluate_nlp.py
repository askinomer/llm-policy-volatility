"""
evaluate_nlp.py — Altın set vs NLP modelleri (Mock / Llama 3) karşılaştırma.

Metrikler:
    - uncertainty_score: Pearson r, Spearman rho, MAE
    - impact_direction : Accuracy, macro-F1, confusion matrix
    - event_type       : Accuracy, macro-F1, top-5 confusion

Kullanım:
    python scripts/evaluate_nlp.py [--gold data/gold_set.csv]
        [--models mock,llama3]

DB'de ilgili model_used değerleri için nlp_scores olmalı.
Mock skorlar pipeline her run'da üretilir. Llama 3 için:
    python main.py  (Ollama açıkken, --force-mock olmadan)
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
from scipy import stats

from src.pipeline.database import DatabaseManager


def load_gold(path: Path) -> pd.DataFrame:
    """Altın set CSV'sini yükler ve etiketlenmiş satırları filtreler."""
    df = pd.read_csv(path)
    required = ["news_id", "gold_event_type", "gold_uncertainty_score",
                "gold_impact_direction"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Eksik kolonlar: {missing}")

    for col in ("gold_event_type", "gold_uncertainty_score", "gold_impact_direction"):
        df[col] = df[col].replace("", np.nan)

    df["gold_uncertainty_score"] = pd.to_numeric(df["gold_uncertainty_score"], errors="coerce")
    df["gold_impact_direction"] = pd.to_numeric(df["gold_impact_direction"], errors="coerce")

    labeled = df.dropna(subset=["gold_event_type", "gold_uncertainty_score",
                                 "gold_impact_direction"]).copy()
    labeled["gold_impact_direction"] = labeled["gold_impact_direction"].astype(int)
    return labeled


def load_model_scores(db: DatabaseManager, model_used: str) -> pd.DataFrame:
    """Belirli bir model_used için DB'den skorları çeker."""
    query = """
        SELECT news_id, event_type, uncertainty_score, impact_direction
        FROM nlp_scores
        WHERE model_used = ?
    """
    df = pd.read_sql_query(query, db.connection, params=[model_used])
    return df


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Sürekli skor için Pearson, Spearman, MAE, RMSE."""
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return {"n": len(y_true)}

    pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
    spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)
    return {
        "n": len(y_true),
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": round(float(pearson_p), 4),
        "spearman_rho": round(float(spearman_r), 4),
        "spearman_p": round(float(spearman_p), 4),
        "MAE": round(float(np.mean(np.abs(y_true - y_pred))), 4),
        "RMSE": round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 4),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Kategorik etiket için accuracy ve macro-F1 (external lib yok, elle)."""
    mask = pd.notna(y_true) & pd.notna(y_pred)
    y_true, y_pred = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    if len(y_true) == 0:
        return {"n": 0}

    accuracy = float((y_true == y_pred).mean())

    labels = sorted(set(y_true) | set(y_pred))
    f1_per_class = []
    for lbl in labels:
        tp = int(((y_true == lbl) & (y_pred == lbl)).sum())
        fp = int(((y_true != lbl) & (y_pred == lbl)).sum())
        fn = int(((y_true == lbl) & (y_pred != lbl)).sum())
        if tp + fp == 0 or tp + fn == 0:
            f1_per_class.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_per_class.append(f1)

    return {
        "n": len(y_true),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(np.mean(f1_per_class)), 4),
        "n_classes": len(labels),
    }


def confusion_matrix(y_true: pd.Series, y_pred: pd.Series, top_n: int = 8) -> pd.DataFrame:
    """En sık görülen top_n etiket için confusion matrix."""
    mask = y_true.notna() & y_pred.notna()
    y_true, y_pred = y_true[mask], y_pred[mask]
    top_labels = y_true.value_counts().head(top_n).index.tolist()
    y_true_f = y_true.where(y_true.isin(top_labels), "OTHER")
    y_pred_f = y_pred.where(y_pred.isin(top_labels), "OTHER")
    cm = pd.crosstab(y_true_f, y_pred_f, rownames=["Gerçek"], colnames=["Tahmin"])
    return cm


def evaluate_model(gold: pd.DataFrame, scores: pd.DataFrame, model_name: str) -> dict:
    """Bir model için tüm metrikleri hesaplar ve raporlar."""
    merged = gold.merge(scores, on="news_id", how="inner")

    if merged.empty:
        print(f"\n[{model_name}] DB'de bu modele ait skor yok.")
        return {}

    print(f"\n{'=' * 70}")
    print(f"  MODEL: {model_name} ({len(merged)} eşleşen örnek)")
    print(f"{'=' * 70}")

    unc_metrics = regression_metrics(
        merged["gold_uncertainty_score"].values,
        merged["uncertainty_score"].values,
    )
    print(f"\n  [uncertainty_score]")
    for k, v in unc_metrics.items():
        print(f"    {k:14s}: {v}")

    imp_metrics = classification_metrics(
        merged["gold_impact_direction"].values,
        merged["impact_direction"].values,
    )
    print(f"\n  [impact_direction]")
    for k, v in imp_metrics.items():
        print(f"    {k:14s}: {v}")

    evt_metrics = classification_metrics(
        merged["gold_event_type"].values,
        merged["event_type"].values,
    )
    print(f"\n  [event_type]")
    for k, v in evt_metrics.items():
        print(f"    {k:14s}: {v}")

    print(f"\n  Event type confusion matrix (top 8):")
    cm = confusion_matrix(merged["gold_event_type"], merged["event_type"])
    if not cm.empty:
        print("  " + cm.to_string().replace("\n", "\n  "))

    return {
        "model": model_name,
        "n_samples": len(merged),
        "uncertainty": unc_metrics,
        "impact": imp_metrics,
        "event_type": evt_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="NLP model değerlendirme")
    parser.add_argument("--gold", type=str, default="data/gold_set.csv",
                        help="Etiketlenmiş altın set CSV yolu")
    parser.add_argument("--models", type=str, default="mock,llama3",
                        help="Değerlendirilecek model_used değerleri (virgülle)")
    args = parser.parse_args()

    gold_path = ROOT / args.gold
    if not gold_path.exists():
        print(f"Altın set bulunamadı: {gold_path}")
        print(f"Önce çalıştır: python scripts/build_gold_set.py")
        sys.exit(1)

    gold = load_gold(gold_path)
    if gold.empty:
        print("Altın set henüz etiketlenmemiş (gold_* kolonları boş).")
        sys.exit(1)

    print(f"Etiketli örnek sayısı: {len(gold)}")
    print(f"Event type dağılımı: {gold['gold_event_type'].value_counts().to_dict()}")

    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    db = DatabaseManager(ROOT / config["database"]["path"])

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    results = {}
    for model_name in model_names:
        scores = load_model_scores(db, model_name)
        if scores.empty:
            print(f"\n[{model_name}] DB'de skor yok, atlanıyor.")
            continue
        results[model_name] = evaluate_model(gold, scores, model_name)

    db.close()

    if len(results) >= 2:
        print(f"\n{'=' * 70}")
        print("  ÖZET KARŞILAŞTIRMA")
        print(f"{'=' * 70}")
        summary_rows = []
        for name, res in results.items():
            if not res:
                continue
            summary_rows.append({
                "Model": name,
                "N": res["n_samples"],
                "Unc_Pearson": res["uncertainty"].get("pearson_r"),
                "Unc_MAE": res["uncertainty"].get("MAE"),
                "Impact_Acc": res["impact"].get("accuracy"),
                "Impact_F1": res["impact"].get("macro_f1"),
                "Event_Acc": res["event_type"].get("accuracy"),
                "Event_F1": res["event_type"].get("macro_f1"),
            })
        print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
