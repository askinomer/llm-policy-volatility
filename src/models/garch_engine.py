"""
GARCHEngine — Finansal volatilite modelleme motoru.

GARCH(1,1), EGARCH(1,1), TARCH/GJR-GARCH(1,1) ve bunların
dışsal değişkenli (-X) versiyonlarını tek bir arayüzden yönetir.

Matematiksel model (GARCH-X):
    σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} + γ·U_t
    U_t: LLM'den gelen belirsizlik skoru (exogenous variable)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Tek bir GARCH model sonucunu tutan veri sınıfı."""
    name: str
    vol_model: str
    has_exog: bool
    aic: float
    bic: float
    log_likelihood: float
    num_params: int
    params: dict
    pvalues: dict
    conditional_volatility: pd.Series
    std_residuals: pd.Series
    _arch_result: ARCHModelResult = field(repr=False)

    @property
    def summary(self) -> str:
        return str(self._arch_result.summary())


class GARCHEngine:
    """
    Çoklu GARCH model ailesi motoru.

    Desteklenen modeller:
        - GARCH(p,q)
        - EGARCH(p,q)     — Asimetrik (leverage effect)
        - TARCH/GJR(p,o,q) — Threshold asimetrik
        - Yukarıdakilerin -X versiyonları (dışsal değişkenli)

    Parameters
    ----------
    p : int
        ARCH terimi derecesi (varsayılan 1).
    q : int
        GARCH terimi derecesi (varsayılan 1).
    o : int
        Asimetrik terim derecesi, TARCH için (varsayılan 1).
    dist : str
        Hata dağılımı: "normal", "t", "skewt", "ged".
    rescale : bool
        arch kütüphanesi ölçekleme (büyük sayılarda True önerilir).
    """

    _SUPPORTED_VOL_MODELS = ("Garch", "EGARCH", "GARCH")
    _SUPPORTED_DISTS = ("normal", "t", "skewt", "ged")

    def __init__(
        self,
        p: int = 1,
        q: int = 1,
        o: int = 1,
        dist: str = "normal",
        rescale: bool = False,
    ) -> None:
        self._p = p
        self._q = q
        self._o = o
        self._dist = dist if dist in self._SUPPORTED_DISTS else "normal"
        self._rescale = rescale
        self._results: dict[str, ModelResult] = {}

    # ------------------------------------------------------------------
    # Model kurma ve eğitme
    # ------------------------------------------------------------------

    def fit_garch(
        self,
        y: pd.Series,
        x: Optional[pd.DataFrame] = None,
        name: Optional[str] = None,
    ) -> ModelResult:
        """
        Standart GARCH(p,q) veya GARCH-X modeli eğitir.

        Parameters
        ----------
        y : pd.Series
            Getiri serisi (log return veya yüzdesel).
        x : pd.DataFrame, optional
            Dışsal değişkenler. Verilirse GARCH-X olur.
        name : str, optional
            Model adı. Yoksa otomatik oluşturulur.
        """
        has_exog = x is not None
        mean_model = "ARX" if has_exog else "AR"
        model_name = name or (f"GARCH-X({self._p},{self._q})" if has_exog
                              else f"GARCH({self._p},{self._q})")

        kwargs = dict(
            y=y, mean=mean_model, vol="Garch",
            p=self._p, q=self._q,
            dist=self._dist, rescale=self._rescale,
        )
        if has_exog:
            kwargs["x"] = x

        return self._fit_and_store(model_name, "Garch", has_exog, **kwargs)

    def fit_egarch(
        self,
        y: pd.Series,
        x: Optional[pd.DataFrame] = None,
        name: Optional[str] = None,
    ) -> ModelResult:
        """
        EGARCH(p,q) veya EGARCH-X modeli eğitir.
        Asimetrik etkiyi (kaldıraç etkisi / leverage effect) yakalar.

        EGARCH denklemi:
            ln(σ²_t) = ω + α·|z_{t-1}| + γ·z_{t-1} + β·ln(σ²_{t-1})
        """
        has_exog = x is not None
        mean_model = "ARX" if has_exog else "AR"
        model_name = name or (f"EGARCH-X({self._p},{self._q})" if has_exog
                              else f"EGARCH({self._p},{self._q})")

        kwargs = dict(
            y=y, mean=mean_model, vol="EGARCH",
            p=self._p, q=self._q,
            o=self._o,
            dist=self._dist, rescale=self._rescale,
        )
        if has_exog:
            kwargs["x"] = x

        return self._fit_and_store(model_name, "EGARCH", has_exog, **kwargs)

    def fit_tarch(
        self,
        y: pd.Series,
        x: Optional[pd.DataFrame] = None,
        name: Optional[str] = None,
    ) -> ModelResult:
        """
        TARCH / GJR-GARCH(p,o,q) veya TARCH-X modeli eğitir.
        Negatif şoklara farklı tepki veren asimetrik model.

        TARCH denklemi:
            σ²_t = ω + α·ε²_{t-1} + γ·ε²_{t-1}·I(ε<0) + β·σ²_{t-1}
        """
        has_exog = x is not None
        mean_model = "ARX" if has_exog else "AR"
        model_name = name or (f"TARCH-X({self._p},{self._o},{self._q})" if has_exog
                              else f"TARCH({self._p},{self._o},{self._q})")

        kwargs = dict(
            y=y, mean=mean_model, vol="Garch",
            p=self._p, q=self._q,
            o=self._o,
            dist=self._dist, rescale=self._rescale,
        )
        if has_exog:
            kwargs["x"] = x

        return self._fit_and_store(model_name, "TARCH", has_exog, **kwargs)

    def fit_all(
        self,
        y: pd.Series,
        x: Optional[pd.DataFrame] = None,
    ) -> dict[str, ModelResult]:
        """
        Tüm model varyantlarını sırayla eğitir:
        GARCH, EGARCH, TARCH (hem baseline hem -X versiyonları).

        Returns
        -------
        dict[str, ModelResult]
            Model adı → sonuç eşlemesi.
        """
        self._results.clear()

        logger.info("Tüm modeller eğitiliyor...")

        self.fit_garch(y, name="GARCH")
        self.fit_egarch(y, name="EGARCH")
        self.fit_tarch(y, name="TARCH")

        if x is not None:
            self.fit_garch(y, x, name="GARCH-X")
            self.fit_egarch(y, x, name="EGARCH-X")
            self.fit_tarch(y, x, name="TARCH-X")

        logger.info("%d model eğitildi.", len(self._results))
        return self._results

    # ------------------------------------------------------------------
    # Tahmin (Forecasting)
    # ------------------------------------------------------------------

    def forecast(
        self,
        model_name: str,
        horizon: int = 5,
        x_forecast: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Eğitilmiş modelden ileriye dönük volatilite tahmini yapar.

        Parameters
        ----------
        model_name : str
            Daha önce eğitilmiş model adı.
        horizon : int
            Tahmin ufku (gün sayısı).
        x_forecast : pd.DataFrame, optional
            Exogenous değişkenlerin gelecek değerleri.
            -X modelleri için zorunlu. Yoksa son değer tekrarlanır.

        Returns
        -------
        pd.DataFrame
            Kolonlar: variance, volatility
        """
        if model_name not in self._results:
            raise KeyError(f"Model bulunamadı: {model_name}. "
                           f"Mevcut: {list(self._results.keys())}")

        model_result = self._results[model_name]
        res = model_result._arch_result

        try:
            kwargs = {"horizon": horizon}

            if model_result.has_exog:
                if x_forecast is not None:
                    kwargs["x"] = x_forecast
                else:
                    raw_x = res.model.x
                    if isinstance(raw_x, np.ndarray):
                        last_row = raw_x[-1]
                    else:
                        last_row = raw_x.iloc[-1].values
                    last_row = np.atleast_1d(last_row)

                    x_names = getattr(res.model, "_x_names", None)
                    if x_names is None and hasattr(raw_x, "columns"):
                        x_names = list(raw_x.columns)
                    if x_names is None:
                        x_names = [f"x{i}" for i in range(len(last_row))]

                    x_dict = {
                        name: np.full(horizon, val)
                        for name, val in zip(x_names, last_row)
                    }
                    kwargs["x"] = x_dict

            fcast = res.forecast(**kwargs)
            return pd.DataFrame({
                "variance": fcast.variance.iloc[-1].values,
                "volatility": np.sqrt(fcast.variance.iloc[-1].values),
            }, index=[f"t+{i+1}" for i in range(horizon)])
        except Exception as exc:
            logger.error("Forecast hatası [%s]: %s", model_name, exc)
            raise

    # ------------------------------------------------------------------
    # Sonuçları DataFrame'e dönüştürme
    # ------------------------------------------------------------------

    def volatility_dataframe(self) -> pd.DataFrame:
        """Tüm modellerin conditional volatility'lerini tek DataFrame'de döner."""
        if not self._results:
            return pd.DataFrame()

        dfs = {}
        for name, result in self._results.items():
            dfs[f"Vol_{name}"] = result.conditional_volatility

        return pd.DataFrame(dfs)

    def residuals_dataframe(self) -> pd.DataFrame:
        """Tüm modellerin standardize residual'larını tek DataFrame'de döner."""
        if not self._results:
            return pd.DataFrame()

        dfs = {}
        for name, result in self._results.items():
            dfs[f"StdResid_{name}"] = result.std_residuals

        return pd.DataFrame(dfs)

    # ------------------------------------------------------------------
    # Dahili yardımcılar
    # ------------------------------------------------------------------

    def _fit_and_store(
        self,
        name: str,
        vol_type: str,
        has_exog: bool,
        **kwargs,
    ) -> ModelResult:
        """Model oluştur, eğit, sonucu sakla."""
        try:
            am = arch_model(**kwargs)
            res = am.fit(disp="off")

            params = dict(res.params)
            pvalues = dict(res.pvalues)

            result = ModelResult(
                name=name,
                vol_model=vol_type,
                has_exog=has_exog,
                aic=res.aic,
                bic=res.bic,
                log_likelihood=res.loglikelihood,
                num_params=res.num_params,
                params=params,
                pvalues=pvalues,
                conditional_volatility=res.conditional_volatility,
                std_residuals=res.resid / res.conditional_volatility,
                _arch_result=res,
            )

            self._results[name] = result
            logger.info("  %-12s → AIC=%.2f  BIC=%.2f  LogL=%.2f",
                         name, res.aic, res.bic, res.loglikelihood)
            return result

        except Exception as exc:
            logger.error("Model eğitim hatası [%s]: %s", name, exc)
            raise

    # ------------------------------------------------------------------
    # Erişim
    # ------------------------------------------------------------------

    @property
    def results(self) -> dict[str, ModelResult]:
        return self._results

    def get(self, name: str) -> Optional[ModelResult]:
        return self._results.get(name)

    def __repr__(self) -> str:
        models = list(self._results.keys()) if self._results else ["(henüz yok)"]
        return f"GARCHEngine(p={self._p}, q={self._q}, models={models})"
