"""
DataPreprocessor — Veri ön işleme sınıfı.

Zaman damgası eşleme, aykırı değer temizleme, eksik veri doldurma
ve özellik mühendisliği (feature engineering) işlemlerini yapar.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Finansal ve haber verilerini modellemeye hazır hale getiren sınıf.

    Parameters
    ----------
    config : dict
        preprocessing bölümü (outlier_method, threshold, fill_strategy, rolling_window).
    """

    _VALID_OUTLIER_METHODS = ("iqr", "zscore")
    _VALID_FILL_STRATEGIES = ("ffill", "bfill", "interpolate", "mean")

    def __init__(self, config: dict) -> None:
        self._outlier_method = config.get("outlier_method", "iqr")
        self._outlier_threshold = config.get("outlier_threshold", 3.0)
        self._fill_strategy = config.get("fill_strategy", "ffill")
        self._rolling_window = config.get("rolling_window", 20)

        if self._outlier_method not in self._VALID_OUTLIER_METHODS:
            raise ValueError(
                f"Geçersiz outlier metodu: {self._outlier_method}. "
                f"Geçerli: {self._VALID_OUTLIER_METHODS}"
            )
        if self._fill_strategy not in self._VALID_FILL_STRATEGIES:
            raise ValueError(
                f"Geçersiz fill stratejisi: {self._fill_strategy}. "
                f"Geçerli: {self._VALID_FILL_STRATEGIES}"
            )

    # ------------------------------------------------------------------
    # Zaman damgası eşleme
    # ------------------------------------------------------------------

    def align_timestamps(
        self,
        financial_df: pd.DataFrame,
        news_df: pd.DataFrame,
        agg_func: str = "mean",
    ) -> pd.DataFrame:
        """
        Finansal veri ile haber verisini tarih bazlı birleştirir.
        Hafta sonu/tatil haberleri bir sonraki iş gününe kaydırılır.

        Parameters
        ----------
        financial_df : pd.DataFrame
            Index = date, kolonlar: close, log_return, ...
        news_df : pd.DataFrame
            Kolonlar: date, uncertainty_score, impact_direction, event_type, ...
        agg_func : str
            Aynı güne düşen birden fazla haber skoru için toplama fonksiyonu.

        Returns
        -------
        pd.DataFrame
            Birleştirilmiş, iş günlerine hizalanmış DataFrame.
        """
        if financial_df.empty:
            logger.warning("Finansal veri boş, align atlanıyor.")
            return financial_df

        trading_days = financial_df.index

        if not news_df.empty:
            if "date" in news_df.columns:
                news_df = news_df.set_index("date")

            news_df.index = pd.to_datetime(news_df.index)
            news_df.index = news_df.index.map(
                lambda d: self._next_trading_day(d, trading_days)
            )

            numeric_cols = news_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                daily_scores = news_df[numeric_cols].groupby(level=0).agg(agg_func)
                merged = financial_df.join(daily_scores, how="left")
            else:
                merged = financial_df.copy()
        else:
            merged = financial_df.copy()

        if "uncertainty_score" in merged.columns:
            merged["uncertainty_score"].fillna(0.5, inplace=True)

        logger.info("Timestamp eşleme tamamlandı: %d satır.", len(merged))
        return merged

    @staticmethod
    def _next_trading_day(date, trading_days: pd.DatetimeIndex):
        """Verilen tarihi en yakın ileri iş gününe kaydırır."""
        future = trading_days[trading_days >= date]
        if len(future) > 0:
            return future[0]
        past = trading_days[trading_days <= date]
        return past[-1] if len(past) > 0 else date

    # ------------------------------------------------------------------
    # Aykırı değer tespiti ve temizleme
    # ------------------------------------------------------------------

    def remove_outliers(
        self,
        df: pd.DataFrame,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Belirtilen kolonlardaki aykırı değerleri NaN'e çevirir.

        Parameters
        ----------
        df : pd.DataFrame
        columns : list[str], optional
            İşlenecek kolonlar. Yoksa tüm sayısal kolonlar.

        Returns
        -------
        pd.DataFrame
            Aykırı değerleri temizlenmiş DataFrame.
        """
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        original_count = df.notna().sum().sum()

        for col in columns:
            if col not in df.columns:
                continue
            mask = self._outlier_mask(df[col])
            n_outliers = mask.sum()
            if n_outliers > 0:
                df.loc[mask, col] = np.nan
                logger.info("  %s: %d aykırı değer NaN yapıldı.", col, n_outliers)

        cleaned_count = df.notna().sum().sum()
        total_removed = original_count - cleaned_count
        logger.info("Toplam %d aykırı değer temizlendi.", total_removed)
        return df

    def _outlier_mask(self, series: pd.Series) -> pd.Series:
        """Seçilen metoda göre outlier maskesi döner."""
        if self._outlier_method == "iqr":
            return self._iqr_mask(series)
        return self._zscore_mask(series)

    def _iqr_mask(self, series: pd.Series) -> pd.Series:
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        factor = self._outlier_threshold
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        return (series < lower) | (series > upper)

    def _zscore_mask(self, series: pd.Series) -> pd.Series:
        mean = series.mean()
        std = series.std()
        if std == 0:
            return pd.Series(False, index=series.index)
        z = (series - mean).abs() / std
        return z > self._outlier_threshold

    # ------------------------------------------------------------------
    # Eksik veri doldurma
    # ------------------------------------------------------------------

    def fill_missing(
        self,
        df: pd.DataFrame,
        columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Eksik verileri seçilen strateji ile doldurur.

        Parameters
        ----------
        df : pd.DataFrame
        columns : list[str], optional
            Doldurulacak kolonlar. Yoksa tüm sayısal kolonlar.
        """
        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        missing_before = df[columns].isna().sum().sum()

        for col in columns:
            if col not in df.columns:
                continue
            if self._fill_strategy == "ffill":
                df[col] = df[col].ffill()
            elif self._fill_strategy == "bfill":
                df[col] = df[col].bfill()
            elif self._fill_strategy == "interpolate":
                df[col] = df[col].interpolate(method="time")
            elif self._fill_strategy == "mean":
                df[col] = df[col].fillna(df[col].mean())

        # ffill/bfill sonrası hala NaN varsa ters yönle doldur
        df[columns] = df[columns].bfill().ffill()

        missing_after = df[columns].isna().sum().sum()
        logger.info("Eksik veri: %d → %d (dolduruldu: %d)",
                     missing_before, missing_after, missing_before - missing_after)
        return df

    # ------------------------------------------------------------------
    # Özellik mühendisliği (Feature Engineering)
    # ------------------------------------------------------------------

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Modelleme için ek özellikler türetir.

        Eklenen kolonlar:
            - abs_return       : Getirinin mutlak değeri
            - return_sq        : Getiri karesi (volatilite proxy)
            - rolling_vol      : Gerçekleşen volatilite (rolling std)
            - rolling_mean     : Hareketli ortalama getiri
            - uncertainty_sma  : Belirsizlik skorunun hareketli ortalaması
            - high_uncertainty : Belirsizlik > 0.7 ise 1 (dummy)
        """
        df = df.copy()
        ret_col = "log_return" if "log_return" in df.columns else "Return"

        if ret_col not in df.columns:
            logger.warning("Getiri kolonu bulunamadı, feature eklenmedi.")
            return df

        w = self._rolling_window

        df["abs_return"] = df[ret_col].abs()
        df["return_sq"] = df[ret_col] ** 2
        df["rolling_vol"] = df[ret_col].rolling(window=w, min_periods=1).std()
        df["rolling_mean"] = df[ret_col].rolling(window=w, min_periods=1).mean()

        unc_col = None
        for candidate in ("uncertainty_score", "avg_uncertainty", "Uncertainty_Score"):
            if candidate in df.columns:
                unc_col = candidate
                break

        if unc_col:
            df["uncertainty_sma"] = df[unc_col].rolling(window=w, min_periods=1).mean()
            df["high_uncertainty"] = (df[unc_col] > 0.7).astype(int)

        logger.info("Feature engineering tamamlandı: %d yeni kolon.", 
                     sum(1 for c in ("abs_return", "return_sq", "rolling_vol",
                         "rolling_mean", "uncertainty_sma", "high_uncertainty") 
                         if c in df.columns))
        return df

    # ------------------------------------------------------------------
    # Tam pipeline
    # ------------------------------------------------------------------

    def run(
        self,
        financial_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame] = None,
        outlier_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Tüm ön işleme adımlarını sırayla çalıştırır.

        1. Timestamp eşleme
        2. Aykırı değer temizleme
        3. Eksik veri doldurma
        4. Feature engineering
        """
        logger.info("Preprocessing pipeline başlatılıyor...")

        if news_df is not None and not news_df.empty:
            df = self.align_timestamps(financial_df, news_df)
        else:
            df = financial_df.copy()

        df = self.remove_outliers(df, columns=outlier_columns)
        df = self.fill_missing(df)
        df = self.add_features(df)

        df.dropna(inplace=True)
        logger.info("Pipeline tamamlandı: %d satır, %d kolon.", len(df), len(df.columns))
        return df
