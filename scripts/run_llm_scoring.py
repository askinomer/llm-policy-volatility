"""
run_llm_scoring.py — Llama 3 (Ollama) ile haberleri skorla.

Özellikler:
  - Resume: DB'de zaten bu model ile skorlu haberleri atlar (UNIQUE constraint'e güvenir).
  - Fallback bypass: LLM başarısız olursa haber atlanır, MockNLP'ye düşmez.
  - Progress bar (tqdm) + ortalama süre + ETA.
  - Batch commit: her N haberde bir DB'ye yazar (çökme toleransı).
  - Hız ayarı: max_tokens, timeout, keep_alive override edilebilir.
  - --sample ile rastgele alt küme alıp sadece onu skorla.

Kullanım:
    python scripts/run_llm_scoring.py                    # tüm eksikler
    python scripts/run_llm_scoring.py --limit 10         # ilk 10
    python scripts/run_llm_scoring.py --sample 500       # rastgele 500
    python scripts/run_llm_scoring.py --reset            # llama3 skorlarını sil
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from tqdm import tqdm

from src.pipeline.database import DatabaseManager
from src.nlp.llm_client import OllamaClient
from src.nlp.prompts import EVENT_TYPES, build_system_prompt, build_user_prompt

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)-20s] %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)


def get_pending_news(db: DatabaseManager, model_used: str, limit: int | None = None):
    query = """
        SELECT n.id, n.date, n.source, n.title, n.content
        FROM news_articles n
        LEFT JOIN nlp_scores s
            ON n.id = s.news_id AND s.model_used = ?
        WHERE s.id IS NULL
        ORDER BY n.id
    """
    params: list = [model_used]
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)
    cur = db.connection.execute(query, params)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]


def validate_and_build(raw: dict, article: dict, model_name: str) -> dict | None:
    """LLM çıktısını doğrula, DB kaydı için dict üret; bozuksa None."""
    if not raw:
        return None
    if isinstance(raw, list):
        if not raw:
            return None
        raw = raw[0]
    if not isinstance(raw, dict):
        return None
    try:
        score = float(raw.get("uncertainty_score", 0.5))
    except (TypeError, ValueError):
        return None
    try:
        direction = int(raw.get("impact_direction", 0))
    except (TypeError, ValueError):
        direction = 0
    event = raw.get("event_type", "diger")
    if event not in EVENT_TYPES:
        event = "diger"

    score = float(np.clip(score, 0.0, 1.0))
    direction = max(-1, min(1, direction))

    return {
        "news_id": article["id"],
        "date": str(article["date"])[:10],
        "event_type": event,
        "uncertainty_score": round(score, 4),
        "impact_direction": direction,
        "model_used": model_name,
        "raw_llm_response": str(raw)[:2000],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Llama 3 ile haber skorlama")
    parser.add_argument("--limit", type=int, default=None,
                        help="Sadece ilk N haberi işle")
    parser.add_argument("--sample", type=int, default=None,
                        help="Eksiklerden rastgele N haber çek (seed ile)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--reset", action="store_true",
                        help="Mevcut bu-model skorlarını sil")
    parser.add_argument("--model", type=str, default=None,
                        help="Ollama modeli (default: config.yaml nlp.model)")
    parser.add_argument("--max-tokens", type=int, default=200,
                        help="LLM max token (default 200, JSON için yeterli)")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Her haber için LLM timeout (sn)")
    parser.add_argument("--content-chars", type=int, default=800,
                        help="Prompt'a gidecek maksimum içerik karakteri")
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        config = yaml.safe_load(f)

    model_name = args.model or config["nlp"].get("model", "llama3")

    db = DatabaseManager(ROOT / config["database"]["path"])

    ollama = OllamaClient(
        base_url=config["nlp"].get("ollama_base_url", "http://localhost:11434"),
        model=model_name,
        temperature=config["nlp"].get("temperature", 0.1),
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    if not ollama.is_available():
        print(f"HATA: Ollama erişilemez ({ollama}).")
        db.close()
        sys.exit(1)

    print(f"Ollama: {ollama}")

    if args.reset:
        with db._transaction() as cur:
            cur.execute("DELETE FROM nlp_scores WHERE model_used = ?", [model_name])
            print(f"Silindi: {cur.rowcount} '{model_name}' kaydı")

    pending = get_pending_news(db, model_name, args.limit)

    if args.sample is not None and args.sample < len(pending):
        random.Random(args.seed).shuffle(pending)
        pending = pending[:args.sample]

    total_news = db.connection.execute("SELECT COUNT(*) FROM news_articles").fetchone()[0]

    print(f"\nToplam haber : {total_news}")
    print(f"İşlenecek    : {len(pending)}")
    print(f"Model        : {model_name}  (max_tokens={args.max_tokens}, timeout={args.timeout}s)")
    print(f"Batch size   : {args.batch_size}")

    if not pending:
        print("\nYapılacak iş yok.")
        db.close()
        return

    system_prompt = build_system_prompt()
    t0 = time.time()
    buffer: list[dict] = []
    saved_total = 0
    failed = 0

    pbar = tqdm(pending, desc=f"{model_name}", unit="haber")
    for article in pbar:
        text = (article.get("content") or "")[:args.content_chars]
        prompt = build_user_prompt(text, article.get("source", ""), str(article.get("date", "")))

        raw = ollama.generate_json(prompt, system_prompt)
        record = validate_and_build(raw, article, model_name)

        if record is None:
            failed += 1
        else:
            buffer.append(record)

        if len(buffer) >= args.batch_size:
            db.insert_scores(buffer)
            saved_total += len(buffer)
            buffer.clear()
            elapsed = time.time() - t0
            pbar.set_postfix(saved=saved_total, failed=failed,
                             rate=f"{saved_total / max(elapsed, 1):.2f}/s")

    if buffer:
        db.insert_scores(buffer)
        saved_total += len(buffer)

    elapsed = time.time() - t0
    db.close()

    print(f"\n{'=' * 60}")
    print(f"  LLM SKORLAMA TAMAMLANDI")
    print(f"{'=' * 60}")
    print(f"  Kaydedilen   : {saved_total}")
    print(f"  Başarısız    : {failed}")
    print(f"  Toplam süre  : {elapsed:.1f} sn ({elapsed/60:.1f} dk)")
    if saved_total > 0:
        print(f"  Ortalama     : {elapsed/saved_total:.2f} sn/haber")


if __name__ == "__main__":
    main()
