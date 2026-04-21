"""
DatabaseManager — SQLite veritabanı yönetim sınıfı.

Finansal veri, haber metinleri ve NLP skorlarını tek bir SQLite DB'de
saklar. Upsert mantığı ile tekrar eden kayıtlar güncellenir.
"""

from __future__ import annotations

import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    SQLite üzerinden tüm proje verisini yöneten CRUD sınıfı.

    Parameters
    ----------
    db_path : str | Path
        Veritabanı dosya yolu. Yoksa otomatik oluşturulur.

    Usage
    -----
    >>> db = DatabaseManager("data/thesis.db")
    >>> db.insert_financial([{"date": "2025-01-02", "ticker": "XU100.IS", ...}])
    >>> df = db.get_merged_data("XU100.IS", "2025-01-01", "2025-06-01")
    >>> db.close()
    """

    _SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._create_tables()
        logger.info("DatabaseManager başlatıldı: %s", self._db_path)

    # ------------------------------------------------------------------
    # Bağlantı yönetimi
    # ------------------------------------------------------------------

    def _connect(self) -> None:
        """SQLite bağlantısı kurar."""
        try:
            self._conn = sqlite3.connect(
                str(self._db_path),
                detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        except sqlite3.Error as exc:
            logger.error("DB bağlantı hatası: %s", exc)
            raise

    @contextmanager
    def _transaction(self):
        """Atomik yazma işlemleri için context manager."""
        cursor = self._conn.cursor()
        try:
            yield cursor
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            logger.error("Transaction geri alındı: %s", exc)
            raise

    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._connect()
        return self._conn

    def close(self) -> None:
        """Bağlantıyı güvenli şekilde kapatır."""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("DB bağlantısı kapatıldı.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ------------------------------------------------------------------
    # Şema oluşturma
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        """Tüm tabloları ve view'ı oluşturur (IF NOT EXISTS)."""
        with self._transaction() as cur:
            cur.executescript("""
                CREATE TABLE IF NOT EXISTS financial_data (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    date        DATE    NOT NULL,
                    ticker      TEXT    NOT NULL,
                    open        REAL,
                    high        REAL,
                    low         REAL,
                    close       REAL    NOT NULL,
                    volume      INTEGER,
                    log_return  REAL,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(date, ticker)
                );

                CREATE INDEX IF NOT EXISTS idx_fin_date_ticker
                    ON financial_data(date, ticker);

                CREATE TABLE IF NOT EXISTS news_articles (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    date        DATE    NOT NULL,
                    source      TEXT    NOT NULL,
                    title       TEXT,
                    content     TEXT    NOT NULL,
                    url         TEXT,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_news_date
                    ON news_articles(date);

                CREATE TABLE IF NOT EXISTS nlp_scores (
                    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                    news_id             INTEGER NOT NULL,
                    date                DATE    NOT NULL,
                    event_type          TEXT,
                    uncertainty_score   REAL    NOT NULL CHECK(
                                            uncertainty_score >= 0.0
                                            AND uncertainty_score <= 1.0
                                        ),
                    impact_direction    INTEGER CHECK(
                                            impact_direction IN (-1, 0, 1)
                                        ),
                    model_used          TEXT    NOT NULL DEFAULT 'mock',
                    raw_llm_response    TEXT,
                    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(news_id, model_used),
                    FOREIGN KEY (news_id) REFERENCES news_articles(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_nlp_date
                    ON nlp_scores(date);
                CREATE INDEX IF NOT EXISTS idx_nlp_news_model
                    ON nlp_scores(news_id, model_used);

                CREATE VIEW IF NOT EXISTS daily_merged AS
                SELECT
                    f.date,
                    f.ticker,
                    f.close,
                    f.log_return,
                    COALESCE(AVG(n.uncertainty_score), 0.5) AS avg_uncertainty,
                    COUNT(n.id) AS news_count,
                    GROUP_CONCAT(DISTINCT n.event_type) AS events
                FROM financial_data f
                LEFT JOIN nlp_scores n ON f.date = n.date
                GROUP BY f.date, f.ticker;
            """)
        logger.info("Veritabanı şeması oluşturuldu / doğrulandı.")

    # ------------------------------------------------------------------
    # INSERT işlemleri
    # ------------------------------------------------------------------

    def insert_financial(self, records: list[dict]) -> int:
        """
        Finansal verileri toplu ekler (upsert: varsa günceller).

        Parameters
        ----------
        records : list[dict]
            Her dict: date, ticker, open, high, low, close, volume, log_return

        Returns
        -------
        int
            Eklenen/güncellenen kayıt sayısı.
        """
        if not records:
            return 0

        sql = """
            INSERT INTO financial_data
                (date, ticker, open, high, low, close, volume, log_return)
            VALUES
                (:date, :ticker, :open, :high, :low, :close, :volume, :log_return)
            ON CONFLICT(date, ticker) DO UPDATE SET
                open       = excluded.open,
                high       = excluded.high,
                low        = excluded.low,
                close      = excluded.close,
                volume     = excluded.volume,
                log_return = excluded.log_return
        """
        with self._transaction() as cur:
            cur.executemany(sql, records)
            count = cur.rowcount
        logger.info("financial_data: %d kayıt yazıldı.", len(records))
        return len(records)

    def insert_news(self, records: list[dict]) -> list[int]:
        """
        Haber kayıtlarını toplu ekler.

        Returns
        -------
        list[int]
            Eklenen kayıtların id listesi.
        """
        if not records:
            return []

        sql = """
            INSERT INTO news_articles (date, source, title, content, url)
            VALUES (:date, :source, :title, :content, :url)
        """
        ids = []
        with self._transaction() as cur:
            for rec in records:
                cur.execute(sql, rec)
                ids.append(cur.lastrowid)
        logger.info("news_articles: %d kayıt yazıldı.", len(ids))
        return ids

    def insert_scores(self, records: list[dict]) -> int:
        """
        NLP skorlarını toplu ekler.

        Parameters
        ----------
        records : list[dict]
            Her dict: news_id, date, event_type, uncertainty_score,
                      impact_direction, model_used, raw_llm_response
        """
        if not records:
            return 0

        sql = """
            INSERT INTO nlp_scores
                (news_id, date, event_type, uncertainty_score,
                 impact_direction, model_used, raw_llm_response)
            VALUES
                (:news_id, :date, :event_type, :uncertainty_score,
                 :impact_direction, :model_used, :raw_llm_response)
            ON CONFLICT(news_id, model_used) DO UPDATE SET
                date              = excluded.date,
                event_type        = excluded.event_type,
                uncertainty_score = excluded.uncertainty_score,
                impact_direction  = excluded.impact_direction,
                raw_llm_response  = excluded.raw_llm_response
        """
        with self._transaction() as cur:
            cur.executemany(sql, records)
        logger.info("nlp_scores: %d kayıt yazıldı.", len(records))
        return len(records)

    # ------------------------------------------------------------------
    # SELECT işlemleri
    # ------------------------------------------------------------------

    def get_financial_data(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Finansal verileri DataFrame olarak döner."""
        query = "SELECT * FROM financial_data WHERE ticker = ?"
        params: list = [ticker]

        if start:
            query += " AND date >= ?"
            params.append(start)
        if end:
            query += " AND date <= ?"
            params.append(end)

        query += " ORDER BY date"
        df = pd.read_sql_query(query, self.connection, params=params, parse_dates=["date"])
        if not df.empty:
            df.set_index("date", inplace=True)
        return df

    def get_news(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        source: Optional[str] = None,
    ) -> pd.DataFrame:
        """Haber verilerini DataFrame olarak döner."""
        query = "SELECT * FROM news_articles WHERE 1=1"
        params: list = []

        if start:
            query += " AND date >= ?"
            params.append(start)
        if end:
            query += " AND date <= ?"
            params.append(end)
        if source:
            query += " AND source = ?"
            params.append(source)

        query += " ORDER BY date"
        return pd.read_sql_query(query, self.connection, params=params, parse_dates=["date"])

    def get_merged_data(
        self,
        ticker: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        daily_merged view'ından birleştirilmiş veriyi döner.
        GARCH modeline doğrudan verilebilecek formatta.
        """
        query = "SELECT * FROM daily_merged WHERE ticker = ?"
        params: list = [ticker]

        if start:
            query += " AND date >= ?"
            params.append(start)
        if end:
            query += " AND date <= ?"
            params.append(end)

        query += " ORDER BY date"
        df = pd.read_sql_query(query, self.connection, params=params, parse_dates=["date"])
        if not df.empty:
            df.set_index("date", inplace=True)
        return df

    # ------------------------------------------------------------------
    # Yardımcılar
    # ------------------------------------------------------------------

    def table_counts(self) -> dict[str, int]:
        """Her tablodaki kayıt sayısını döner (debug/log amaçlı)."""
        counts = {}
        for table in ("financial_data", "news_articles", "nlp_scores"):
            row = self.connection.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()
            counts[table] = row[0]
        return counts

    def __repr__(self) -> str:
        counts = self.table_counts()
        return (
            f"DatabaseManager(path='{self._db_path}', "
            f"financial={counts.get('financial_data', 0)}, "
            f"news={counts.get('news_articles', 0)}, "
            f"scores={counts.get('nlp_scores', 0)})"
        )
