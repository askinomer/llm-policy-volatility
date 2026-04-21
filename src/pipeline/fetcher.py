"""
Veri Çekme Sınıfları — Finansal veri ve haber akışı.

FinancialDataFetcher : yfinance üzerinden fiyat verisi çeker.
NewsDataFetcher      : Abstract base; alt sınıflar (TCMB, FED, GDELT) implemente eder.
MockNewsFetcher      : Demo/test için sahte haber üretir.
TCMBNewsFetcher      : TCMB basın duyuruları ve PPK kararlarını çeker.
FEDNewsFetcher       : Fed basın duyuruları ve FOMC açıklamalarını çeker.
GDELTNewsFetcher     : GDELT DOC 2.0 API ile Türkiye ekonomi haberlerini çeker.
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from .database import DatabaseManager

logger = logging.getLogger(__name__)


class FinancialDataFetcher:
    """
    yfinance API ile finansal fiyat verisi çeken ve DB'ye kaydeden sınıf.

    Parameters
    ----------
    db : DatabaseManager
        Veritabanı bağlantısı.
    config : dict
        financial bölümü (tickers, period, interval).
    """

    def __init__(self, db: DatabaseManager, config: dict) -> None:
        self._db = db
        self._config = config

    def fetch(
        self,
        ticker: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Belirtilen ticker için fiyat verisi çeker.

        Parameters
        ----------
        ticker : str, optional
            Hisse/endeks kodu. Yoksa config'deki default_ticker kullanılır.
        start / end : str, optional
            "YYYY-MM-DD" formatında. Verilmezse period kullanılır.
        period : str, optional
            yfinance period parametresi ("6mo", "1y" vb.).

        Returns
        -------
        pd.DataFrame
            Kolonlar: Open, High, Low, Close, Volume, Log_Return
        """
        ticker = ticker or self._config.get("default_ticker", "XU100.IS")
        period = period or self._config.get("period", "1y")

        try:
            if start and end:
                raw = yf.download(ticker, start=start, end=end,
                                  interval=self._config.get("interval", "1d"),
                                  progress=False)
            else:
                raw = yf.download(ticker, period=period,
                                  interval=self._config.get("interval", "1d"),
                                  progress=False)
        except Exception as exc:
            logger.error("yfinance veri çekme hatası [%s]: %s", ticker, exc)
            raise

        if raw.empty:
            logger.warning("Boş veri döndü: %s", ticker)
            return pd.DataFrame()

        df = self._normalize(raw, ticker)
        df = self._compute_log_returns(df)

        logger.info("%s: %d günlük veri çekildi.", ticker, len(df))
        return df

    def fetch_and_save(self, **kwargs) -> pd.DataFrame:
        """Veriyi çeker ve DB'ye kaydeder."""
        df = self.fetch(**kwargs)
        if df.empty:
            return df

        export = df.reset_index()
        export["date"] = export["date"].dt.strftime("%Y-%m-%d")
        self._db.insert_financial(export.to_dict("records"))
        return df

    @staticmethod
    def _normalize(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """yfinance çıktısını düz sütun formatına dönüştürür."""
        df = pd.DataFrame(index=raw.index)
        df.index.name = "date"

        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col in raw.columns:
                vals = raw[col]
                if hasattr(vals, "values"):
                    df[col.lower()] = vals.values.flatten()

        df["ticker"] = ticker
        df.dropna(subset=["close"], inplace=True)
        return df

    @staticmethod
    def _compute_log_returns(df: pd.DataFrame) -> pd.DataFrame:
        """Logaritmik getiri: ln(P_t / P_{t-1}) * 100"""
        df["log_return"] = np.log(df["close"] / df["close"].shift(1)) * 100
        df.dropna(subset=["log_return"], inplace=True)
        return df


# ======================================================================
# Haber Çekme — Abstract Base + Mock Implementation
# ======================================================================

class NewsDataFetcher(ABC):
    """
    Haber verisi çeken abstract base sınıf.
    TCMB, FED, GDELT gibi her kaynak bu sınıfı miras alır.

    Parameters
    ----------
    db : DatabaseManager
    config : dict
        news bölümü.
    """

    def __init__(self, db: DatabaseManager, config: dict) -> None:
        self._db = db
        self._config = config

    @abstractmethod
    def fetch(
        self,
        start: str,
        end: str,
    ) -> list[dict]:
        """
        Belirtilen tarih aralığındaki haberleri çeker.

        Her dict şu anahtarları içermelidir:
            date, source, title, content, url
        """
        ...

    def fetch_and_save(self, start: str, end: str) -> list[int]:
        """Haberleri çeker ve DB'ye kaydeder, id listesi döner."""
        articles = self.fetch(start, end)
        if not articles:
            return []
        return self._db.insert_news(articles)


class MockNewsFetcher(NewsDataFetcher):
    """
    Demo/test amaçlı sahte haber üretici.
    İleride gerçek TCMB/FED scraper'ları ile değiştirilecek.
    """

    _MOCK_POOL: dict[str, list[str]] = {
        "tcmb": [
            "TCMB politika faizini 500 baz puan artırdı, piyasalarda tedirginlik hakim.",
            "Merkez Bankası enflasyon hedefini korudu, faiz sabit bırakıldı, piyasa stabil.",
            "TCMB döviz rezervleri eridi, kur baskısı artıyor, gerilim tırmanıyor.",
            "Para politikası kurulu toplantısında sürpriz olmadı, beklentiler yolunda.",
            "TCMB başkanı sıkılaştırma mesajı verdi, belirsizlik devam ediyor.",
            "Enflasyon raporu beklentilerin altında geldi, piyasalar olumlu karşıladı.",
            "Merkez Bankası swap ihalesi düzenledi, likidite yönetimi devam ediyor.",
        ],
        "fed": [
            "Fed faiz oranlarını 25 baz puan artırdı, küresel piyasalarda tedirginlik.",
            "Powell: Ekonomi güçlü, yumuşak iniş mümkün. Piyasalar stabil seyrediyor.",
            "ABD istihdam verileri güçlü geldi, faiz indirimi beklentisi zayıfladı.",
            "Fed bilanço küçültmeye devam, gelişen piyasalarda sermaye çıkışı riski.",
            "FOMC tutanakları şahin ton içeriyordu, küresel gerilim arttı.",
            "Fed yetkilileri enflasyonun kontrol altına alındığını belirtti, rahatlatıcı mesaj.",
            "Dolar endeksi güçleniyor, gelişen piyasa paraları baskı altında.",
        ],
        "yerel": [
            "Cari açık verileri beklentinin üzerinde geldi, TL üzerinde baskı artıyor.",
            "Sanayi üretimi yükseldi, büyüme rakamları olumlu, ekonomi yolunda.",
            "Jeopolitik gerilimler sınırda tırmandı, savunma harcamaları tartışılıyor.",
            "Kredi derecelendirme kuruluşu Türkiye notunu teyit etti, görünüm stabil.",
            "Seçim belirsizliği piyasaları olumsuz etkiliyor, yatırımcı temkinli.",
            "İhracat rekor kırdı, dış ticaret dengesi iyileşiyor.",
            "Bütçe açığı hedefin üzerine çıktı, mali disiplin sorgulanıyor.",
        ],
    }

    _SOURCE_WEIGHTS = {"tcmb": 0.4, "fed": 0.3, "yerel": 0.3}

    def fetch(self, start: str, end: str) -> list[dict]:
        """
        start-end arasındaki iş günleri için rastgele mock haber üretir.
        Her iş günü için 1-3 haber döner.
        """
        dates = pd.bdate_range(start=start, end=end)
        sources = list(self._SOURCE_WEIGHTS.keys())
        probs = list(self._SOURCE_WEIGHTS.values())

        articles = []
        for dt in dates:
            n_articles = np.random.randint(1, 4)
            for _ in range(n_articles):
                source = np.random.choice(sources, p=probs)
                content = np.random.choice(self._MOCK_POOL[source])
                articles.append({
                    "date": dt.strftime("%Y-%m-%d"),
                    "source": source,
                    "title": content[:60] + "...",
                    "content": content,
                    "url": None,
                })

        logger.info("MockNewsFetcher: %d haber üretildi (%s → %s).",
                     len(articles), start, end)
        return articles


# ======================================================================
# TCMB Basın Duyuruları Scraper
# ======================================================================

class TCMBNewsFetcher(NewsDataFetcher):
    """TCMB basın duyuruları ve PPK kararlarını web'den çeker."""

    BASE_URL = "https://www.tcmb.gov.tr"
    PRESS_URL = (
        "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/"
        "Main+Menu/Duyurular/Basin+Duyurulari/"
    )
    PPK_URL = (
        "https://www.tcmb.gov.tr/wps/wcm/connect/TR/TCMB+TR/"
        "Main+Menu/Duyurular/Para+Politikasi+Kurulu/"
    )
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
    }
    REQUEST_TIMEOUT = 15

    def fetch(self, start: str, end: str) -> list[dict]:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        articles: list[dict] = []

        for url, label in [
            (self.PRESS_URL, "basin_duyurusu"),
            (self.PPK_URL, "ppk_karari"),
        ]:
            try:
                articles.extend(self._scrape_listing(url, label, start_dt, end_dt))
            except Exception as exc:
                logger.warning("TCMB %s çekme hatası: %s", label, exc)

        articles.sort(key=lambda a: a["date"])
        logger.info("TCMBNewsFetcher: %d haber çekildi (%s → %s).", len(articles), start, end)
        return articles

    def _scrape_listing(self, url, label, start_dt, end_dt) -> list[dict]:
        resp = self._get(url)
        if resp is None:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        items: list[dict] = []

        for tag in soup.select("a[href]"):
            text = tag.get_text(strip=True)
            if len(text) < 20:
                continue

            date_match = re.search(r"(\d{1,2})[./](\d{1,2})[./](\d{4})", text)
            if not date_match:
                parent = tag.find_parent(["tr", "li", "div"])
                if parent:
                    date_match = re.search(r"(\d{1,2})[./](\d{1,2})[./](\d{4})", parent.get_text())
            if not date_match:
                continue

            try:
                art_date = datetime(int(date_match.group(3)), int(date_match.group(2)), int(date_match.group(1)))
            except ValueError:
                continue
            if art_date < start_dt or art_date > end_dt:
                continue

            href = tag.get("href", "")
            full_url = href if href.startswith("http") else self.BASE_URL + href
            content = self._fetch_article_content(full_url) or text[:120]

            items.append({
                "date": art_date.strftime("%Y-%m-%d"),
                "source": "tcmb",
                "title": f"[{label}] {text[:80]}",
                "content": content[:2000],
                "url": full_url,
            })
        return items

    def _fetch_article_content(self, url: str) -> str:
        resp = self._get(url)
        if resp is None:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for sel in ["div.tcmb-content", "div.content-area", "article", "div#mainContent", "div.wpthemeContentArea"]:
            block = soup.select_one(sel)
            if block:
                return block.get_text(separator=" ", strip=True)[:2000]
        body = soup.find("body")
        return body.get_text(separator=" ", strip=True)[:2000] if body else ""

    def _get(self, url: str, retries: int = 2) -> Optional[requests.Response]:
        for attempt in range(retries + 1):
            try:
                resp = requests.get(url, headers=self.HEADERS, timeout=self.REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                logger.debug("TCMB GET hatası (%d/%d): %s", attempt + 1, retries + 1, exc)
                if attempt < retries:
                    time.sleep(2)
        return None


# ======================================================================
# FED Basın Duyuruları Scraper
# ======================================================================

class FEDNewsFetcher(NewsDataFetcher):
    """Federal Reserve basın duyuruları ve FOMC açıklamalarını çeker."""

    RSS_FEEDS = [
        "https://www.federalreserve.gov/feeds/press_monetary.xml",
        "https://www.federalreserve.gov/feeds/press_all.xml",
    ]
    FOMC_CALENDAR = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/rss+xml, application/xml, text/xml, */*",
    }
    REQUEST_TIMEOUT = 15

    def fetch(self, start: str, end: str) -> list[dict]:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        articles: list[dict] = []
        seen: set[str] = set()

        for feed_url in self.RSS_FEEDS:
            try:
                articles.extend(self._parse_rss(feed_url, start_dt, end_dt, seen))
            except Exception as exc:
                logger.warning("FED RSS hatası [%s]: %s", feed_url, exc)

        try:
            articles.extend(self._scrape_fomc(start_dt, end_dt, seen))
        except Exception as exc:
            logger.warning("FOMC scraping hatası: %s", exc)

        articles.sort(key=lambda a: a["date"])
        logger.info("FEDNewsFetcher: %d haber çekildi (%s → %s).", len(articles), start, end)
        return articles

    def _parse_rss(self, feed_url, start_dt, end_dt, seen) -> list[dict]:
        resp = self._get(feed_url)
        if resp is None:
            return []
        root = ET.fromstring(resp.content)
        ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
        items: list[dict] = []

        for item in root.iter(f"{ns}item"):
            title_el = item.find(f"{ns}title")
            link_el = item.find(f"{ns}link")
            pub_el = item.find(f"{ns}pubDate")
            desc_el = item.find(f"{ns}description")
            if title_el is None or pub_el is None:
                continue

            title = title_el.text or ""
            link = (link_el.text or "").strip() if link_el is not None else ""
            if link in seen:
                continue

            art_date = self._parse_rss_date(pub_el.text or "")
            if art_date is None or art_date < start_dt or art_date > end_dt:
                continue

            content = (desc_el.text or "").strip() if desc_el is not None else ""
            if link:
                page = self._fetch_article_content(link)
                if page:
                    content = page
            if not content:
                content = title

            seen.add(link)
            items.append({
                "date": art_date.strftime("%Y-%m-%d"),
                "source": "fed",
                "title": title[:120].strip(),
                "content": content[:2000],
                "url": link or None,
            })
        return items

    def _scrape_fomc(self, start_dt, end_dt, seen) -> list[dict]:
        resp = self._get(self.FOMC_CALENDAR)
        if resp is None:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        items: list[dict] = []

        for link in soup.select("a[href*='pressreleases/monetary']"):
            href = link.get("href", "")
            if not href:
                continue
            full_url = href if href.startswith("http") else "https://www.federalreserve.gov" + href
            if full_url in seen:
                continue

            dm = re.search(r"(\d{8})", href)
            if not dm:
                continue
            try:
                art_date = datetime.strptime(dm.group(1), "%Y%m%d")
            except ValueError:
                continue
            if art_date < start_dt or art_date > end_dt:
                continue

            title = link.get_text(strip=True) or "FOMC Statement"
            content = self._fetch_article_content(full_url) or title
            seen.add(full_url)
            items.append({
                "date": art_date.strftime("%Y-%m-%d"),
                "source": "fed",
                "title": f"[FOMC] {title[:100]}",
                "content": content[:2000],
                "url": full_url,
            })
        return items

    def _fetch_article_content(self, url: str) -> str:
        resp = self._get(url)
        if resp is None:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        for sel in ["div#article", "div.col-xs-12.col-sm-8.col-md-8", "div#content", "article"]:
            block = soup.select_one(sel)
            if block:
                return block.get_text(separator=" ", strip=True)[:2000]
        body = soup.find("body")
        return body.get_text(separator=" ", strip=True)[:2000] if body else ""

    @staticmethod
    def _parse_rss_date(date_str: str) -> Optional[datetime]:
        for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S %Z",
                     "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str.strip(), fmt).replace(tzinfo=None)
            except ValueError:
                continue
        return None

    def _get(self, url: str, retries: int = 2) -> Optional[requests.Response]:
        for attempt in range(retries + 1):
            try:
                resp = requests.get(url, headers=self.HEADERS, timeout=self.REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                logger.debug("FED GET hatası (%d/%d): %s", attempt + 1, retries + 1, exc)
                if attempt < retries:
                    time.sleep(2)
        return None


# ======================================================================
# GDELT DOC 2.0 API Haber Scraper
# ======================================================================

class GDELTNewsFetcher(NewsDataFetcher):
    """
    GDELT DOC 2.0 API üzerinden Türkiye ekonomi/politika haberlerini çeker.

    API: https://api.gdeltproject.org/api/v2/doc/doc
    Ücretsiz, API key gerektirmez, JSON formatında sonuç döner.
    250 haber/sorgu limiti var; tarih aralığını parçalayarak aşılır.
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

    QUERY_TERMS = [
        '("Turkey" OR "Türkiye" OR "TCMB" OR "Turkish lira")'
        ' AND ("economy" OR "interest rate" OR "inflation" OR "monetary policy"'
        ' OR "central bank" OR "GDP" OR "trade" OR "fiscal")',
    ]

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    REQUEST_TIMEOUT = 20
    MAX_PER_QUERY = 250
    CHUNK_DAYS = 30

    def fetch(self, start: str, end: str) -> list[dict]:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        max_per_day = self._config.get("max_articles_per_day", 50)

        all_articles: list[dict] = []
        seen_urls: set[str] = set()

        chunk_start = start_dt
        while chunk_start <= end_dt:
            chunk_end = min(chunk_start + timedelta(days=self.CHUNK_DAYS - 1), end_dt)

            for query in self.QUERY_TERMS:
                try:
                    batch = self._query_api(query, chunk_start, chunk_end, seen_urls)
                    all_articles.extend(batch)
                except Exception as exc:
                    logger.warning(
                        "GDELT sorgu hatası (%s → %s): %s",
                        chunk_start.strftime("%Y-%m-%d"),
                        chunk_end.strftime("%Y-%m-%d"),
                        exc,
                    )
            chunk_start = chunk_end + timedelta(days=1)

        date_counts: dict[str, int] = {}
        filtered: list[dict] = []
        for art in sorted(all_articles, key=lambda a: a["date"]):
            dc = date_counts.get(art["date"], 0)
            if dc < max_per_day:
                filtered.append(art)
                date_counts[art["date"]] = dc + 1

        logger.info(
            "GDELTNewsFetcher: %d haber çekildi (%s → %s).",
            len(filtered), start, end,
        )
        return filtered

    def _query_api(
        self,
        query: str,
        start_dt: datetime,
        end_dt: datetime,
        seen: set[str],
    ) -> list[dict]:
        params = {
            "query": query,
            "mode": "ArtList",
            "maxrecords": str(self.MAX_PER_QUERY),
            "format": "json",
            "startdatetime": start_dt.strftime("%Y%m%d%H%M%S"),
            "enddatetime": (end_dt + timedelta(days=1)).strftime("%Y%m%d%H%M%S"),
            "sort": "DateDesc",
        }

        url = f"{self.BASE_URL}?{urlencode(params)}"
        resp = self._get(url)
        if resp is None:
            return []

        try:
            data = resp.json()
        except Exception:
            logger.warning("GDELT JSON parse hatası")
            return []

        raw_articles = data.get("articles", [])
        if not raw_articles:
            return []

        items: list[dict] = []
        for art in raw_articles:
            art_url = art.get("url", "")
            if art_url in seen:
                continue

            seendate = art.get("seendate", "")
            art_date = self._parse_gdelt_date(seendate)
            if art_date is None:
                continue

            title = art.get("title", "").strip()
            domain = art.get("domain", "")
            language = art.get("language", "")

            content = title
            if len(title) < 40:
                content = f"{title}. {art.get('socialimage', '')}"

            seen.add(art_url)
            items.append({
                "date": art_date.strftime("%Y-%m-%d"),
                "source": "gdelt",
                "title": f"[{domain}] {title[:100]}",
                "content": content[:2000],
                "url": art_url,
            })

        return items

    @staticmethod
    def _parse_gdelt_date(date_str: str) -> Optional[datetime]:
        for fmt in ("%Y%m%dT%H%M%SZ", "%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return datetime.strptime(date_str.strip(), fmt)
            except ValueError:
                continue
        return None

    def _get(self, url: str, retries: int = 2) -> Optional[requests.Response]:
        for attempt in range(retries + 1):
            try:
                resp = requests.get(url, headers=self.HEADERS, timeout=self.REQUEST_TIMEOUT)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                logger.debug("GDELT GET hatası (%d/%d): %s", attempt + 1, retries + 1, exc)
                if attempt < retries:
                    time.sleep(3)
        return None
