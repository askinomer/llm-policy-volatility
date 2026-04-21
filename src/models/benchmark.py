"""
ModelBenchmark — GARCH model ailesi karşılaştırma ve diagnostik sınıfı.
OutOfSampleEvaluator — Train/test split ile OOS performans değerlendirmesi.

AIC/BIC/LogLikelihood sıralaması, Ljung-Box testi,
realized volatility korelasyonu ve model seçimi yapar.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from arch import arch_model
from scipy import stats

from .garch_engine import GARCHEngine, ModelResult

logger = logging.getLogger(__name__)


class ModelBenchmark:
    """
    Eğitilmiş GARCH modellerini karşılaştırır ve en iyi modeli seçer.

    Parameters
    ----------
    engine : GARCHEngine
        Eğitilmiş modelleri içeren motor nesnesi.
    returns : pd.Series
        Orijinal getiri serisi (realized vol hesabı için).
    rolling_window : int
        Realized volatility pencere boyutu.
    """

    def __init__(
        self,
        engine: GARCHEngine,
        returns: pd.Series,
        rolling_window: int = 20,
    ) -> None:
        self._engine = engine
        self._returns = returns
        self._rolling_window = rolling_window
        self._realized_vol = returns.rolling(window=rolling_window).std().dropna()

    # ------------------------------------------------------------------
    # Karşılaştırma tablosu
    # ------------------------------------------------------------------

    def comparison_table(self) -> pd.DataFrame:
        """
        Tüm modellerin AIC, BIC, LogLik, parametre sayısı ve
        realized vol korelasyonunu içeren sıralı tablo döner.
        """
        rows = []
        for name, result in self._engine.results.items():
            corr = self._vol_correlation(result)
            mae = self._vol_mae(result)

            rows.append({
                "Model": name,
                "AIC": round(result.aic, 2),
                "BIC": round(result.bic, 2),
                "Log-Lik": round(result.log_likelihood, 2),
                "Params": result.num_params,
                "Exog": "Var" if result.has_exog else "Yok",
                "Vol_Corr": round(corr, 4),
                "Vol_MAE": round(mae, 4),
            })

        df = pd.DataFrame(rows)
        df.sort_values("AIC", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.index = df.index + 1
        df.index.name = "Sıra"
        return df

    def best_model(self, criterion: str = "AIC") -> str:
        """Belirtilen kritere göre en iyi model adını döner."""
        table = self.comparison_table()
        col_map = {"AIC": "AIC", "BIC": "BIC", "LogLik": "Log-Lik"}
        col = col_map.get(criterion, "AIC")

        if criterion == "LogLik":
            best_idx = table[col].idxmax()
        else:
            best_idx = table[col].idxmin()

        return table.loc[best_idx, "Model"]

    # ------------------------------------------------------------------
    # Realized volatility karşılaştırması
    # ------------------------------------------------------------------

    def _vol_correlation(self, result: ModelResult) -> float:
        """Model volatilitesi ile realized vol arasındaki korelasyon."""
        try:
            common = self._realized_vol.index.intersection(
                result.conditional_volatility.index
            )
            if len(common) < 10:
                return 0.0
            return float(self._realized_vol.loc[common].corr(
                result.conditional_volatility.loc[common]
            ))
        except Exception:
            return 0.0

    def _vol_mae(self, result: ModelResult) -> float:
        """Model volatilitesi ile realized vol arasındaki MAE."""
        try:
            common = self._realized_vol.index.intersection(
                result.conditional_volatility.index
            )
            if len(common) < 10:
                return float("inf")
            diff = (self._realized_vol.loc[common] -
                    result.conditional_volatility.loc[common]).abs()
            return float(diff.mean())
        except Exception:
            return float("inf")

    # ------------------------------------------------------------------
    # Diagnostik testler
    # ------------------------------------------------------------------

    def ljung_box_test(self, model_name: str, lags: int = 10) -> dict:
        """
        Standardize residual'lar üzerinde Ljung-Box otokorelasyon testi.
        H0: Residual'larda otokorelasyon yoktur.

        Returns
        -------
        dict
            statistic, p_value, lags, is_adequate (p > 0.05 ise True)
        """
        result = self._engine.get(model_name)
        if result is None:
            raise KeyError(f"Model bulunamadı: {model_name}")

        resid = result.std_residuals.dropna()
        n = len(resid)

        acf_vals = []
        for k in range(1, lags + 1):
            acf_vals.append(float(resid.autocorr(lag=k)))

        q_stat = n * (n + 2) * sum(
            (r ** 2) / (n - k) for k, r in enumerate(acf_vals, 1)
        )
        p_value = 1 - stats.chi2.cdf(q_stat, df=lags)

        return {
            "model": model_name,
            "statistic": round(q_stat, 4),
            "p_value": round(p_value, 4),
            "lags": lags,
            "is_adequate": p_value > 0.05,
        }

    def jarque_bera_test(self, model_name: str) -> dict:
        """
        Standardize residual'ların normallik testi.
        H0: Residual'lar normal dağılımlıdır.
        """
        result = self._engine.get(model_name)
        if result is None:
            raise KeyError(f"Model bulunamadı: {model_name}")

        resid = result.std_residuals.dropna().values
        jb_stat, jb_pval = stats.jarque_bera(resid)

        return {
            "model": model_name,
            "statistic": round(float(jb_stat), 4),
            "p_value": round(float(jb_pval), 4),
            "skewness": round(float(stats.skew(resid)), 4),
            "kurtosis": round(float(stats.kurtosis(resid)), 4),
            "is_normal": jb_pval > 0.05,
        }

    def exog_significance(self) -> pd.DataFrame:
        """
        Dışsal değişkenli (-X) modellerdeki exog katsayısının
        p-value ve anlamlılık durumunu döner.
        """
        rows = []
        for name, result in self._engine.results.items():
            if not result.has_exog:
                continue

            for param_name, pval in result.pvalues.items():
                if param_name in ("Const", "omega", "alpha[1]", "beta[1]",
                                  "gamma[1]", "eta[1]"):
                    continue
                rows.append({
                    "Model": name,
                    "Parametre": param_name,
                    "Katsayı": round(result.params.get(param_name, 0), 6),
                    "P-Value": round(pval, 4),
                    "Anlamlı (α=0.05)": "Evet" if pval < 0.05 else "Hayır",
                    "Anlamlı (α=0.10)": "Evet" if pval < 0.10 else "Hayır",
                })

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def full_diagnostics(self) -> dict:
        """Tüm modeller için diagnostik raporu."""
        diag = {}
        for name in self._engine.results:
            diag[name] = {
                "ljung_box": self.ljung_box_test(name),
                "jarque_bera": self.jarque_bera_test(name),
            }
        return diag

    # ------------------------------------------------------------------
    # Raporlama
    # ------------------------------------------------------------------

    def print_report(self) -> None:
        """Konsola detaylı karşılaştırma raporu yazdırır."""
        table = self.comparison_table()
        best_aic = self.best_model("AIC")
        best_bic = self.best_model("BIC")

        print("\n" + "=" * 80)
        print("  MODEL KARŞILAŞTIRMA RAPORU")
        print("=" * 80)
        print(table.to_string())

        print(f"\n  En iyi model (AIC): {best_aic}")
        print(f"  En iyi model (BIC): {best_bic}")

        exog_df = self.exog_significance()
        if not exog_df.empty:
            print("\n  DIŞSAL DEĞİŞKEN ANLAMLILIĞI:")
            print("  " + exog_df.to_string(index=False).replace("\n", "\n  "))

        print("\n  DİAGNOSTİK TESTLER:")
        for name in self._engine.results:
            lb = self.ljung_box_test(name)
            jb = self.jarque_bera_test(name)
            lb_ok = "✓" if lb["is_adequate"] else "✗"
            jb_ok = "✓" if jb["is_normal"] else "✗"
            print(f"  {name:15s} | Ljung-Box p={lb['p_value']:.3f} {lb_ok} | "
                  f"Jarque-Bera p={jb['p_value']:.3f} {jb_ok}")

        print("=" * 80)


# ======================================================================
# Out-of-Sample Evaluator
# ======================================================================

class OutOfSampleEvaluator:
    """
    Train/test split ile GARCH modellerinin out-of-sample performansını ölçer.

    Expanding window 1-step-ahead forecast yaparak test döneminde
    tahmin edilen volatiliteyi realized volatilite ile karşılaştırır.

    Metrikler: RMSE, MAE, QLIKE (quasi-likelihood), Hit Rate

    Parameters
    ----------
    returns : pd.Series
        Tam getiri serisi.
    exog : pd.DataFrame, optional
        Tam exog serisi (avg_uncertainty vb.).
    test_ratio : float
        Test setinin oranı (varsayılan 0.2 = son %20).
    garch_config : dict
        p, q, distribution parametreleri.
    """

    def __init__(
        self,
        returns: pd.Series,
        exog: Optional[pd.DataFrame] = None,
        test_ratio: float = 0.20,
        garch_config: Optional[dict] = None,
    ) -> None:
        self._returns = returns.dropna()
        self._exog = exog
        self._test_ratio = test_ratio
        self._cfg = garch_config or {"p": 1, "q": 1, "distribution": "normal"}

        n = len(self._returns)
        self._split_idx = int(n * (1 - test_ratio))
        self._train_y = self._returns.iloc[: self._split_idx]
        self._test_y = self._returns.iloc[self._split_idx:]

        if exog is not None:
            self._train_x = exog.loc[self._train_y.index]
            self._test_x = exog.loc[self._test_y.index]
        else:
            self._train_x = None
            self._test_x = None

        self._realized_vol = self._test_y.pow(2)
        self._oos_results: dict[str, dict] = {}

        logger.info(
            "OOS split: train=%d, test=%d (ratio=%.0f%%)",
            len(self._train_y), len(self._test_y), test_ratio * 100,
        )

    @property
    def train_size(self) -> int:
        return len(self._train_y)

    @property
    def test_size(self) -> int:
        return len(self._test_y)

    @property
    def split_date(self) -> str:
        return str(self._test_y.index[0].date())

    def evaluate_all(self) -> pd.DataFrame:
        """
        6 model varyantı için expanding window
        1-step-ahead forecast ile OOS metriklerini hesaplar.
        """
        p = self._cfg.get("p", 1)
        q = self._cfg.get("q", 1)
        dist = self._cfg.get("distribution", "normal")

        model_specs = [
            ("GARCH", "Garch", "AR", False),
            ("EGARCH", "EGARCH", "AR", False),
            ("TARCH", "Garch", "AR", False),
            ("GARCH-X", "Garch", "ARX", True),
            ("EGARCH-X", "EGARCH", "ARX", True),
            ("TARCH-X", "Garch", "ARX", True),
        ]

        rows = []
        for name, vol, mean, use_exog in model_specs:
            if use_exog and self._exog is None:
                continue
            try:
                metrics = self._rolling_forecast(name, vol, mean, use_exog, p, q, dist)
                self._oos_results[name] = metrics
                rows.append({"Model": name, **metrics})
                logger.info(
                    "  OOS %-12s → RMSE=%.4f  MAE=%.4f  QLIKE=%.4f",
                    name, metrics["RMSE"], metrics["MAE"], metrics["QLIKE"],
                )
            except Exception as exc:
                logger.warning("OOS %s hatası: %s", name, exc)

        df = pd.DataFrame(rows)
        if not df.empty:
            df.sort_values("RMSE", inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def _rolling_forecast(
        self,
        name: str,
        vol: str,
        mean: str,
        use_exog: bool,
        p: int,
        q: int,
        dist: str,
    ) -> dict:
        """Expanding window 1-step-ahead forecast."""
        is_tarch = "TARCH" in name
        forecast_var = []
        actual_var = []

        test_indices = self._test_y.index
        full_y = self._returns

        for i, test_date in enumerate(test_indices):
            train_end_pos = self._split_idx + i
            y_train = full_y.iloc[:train_end_pos]

            kwargs = dict(
                y=y_train, mean=mean, vol=vol,
                p=p, q=q, dist=dist, rescale=False,
            )
            if is_tarch:
                kwargs["o"] = 1
            if use_exog:
                kwargs["x"] = self._exog.loc[y_train.index]

            try:
                am = arch_model(**kwargs)
                res = am.fit(disp="off", show_warning=False)

                fc_kwargs = {"horizon": 1}
                if use_exog:
                    x_row = self._exog.loc[[test_date]]
                    fc_kwargs["x"] = {
                        col: x_row[col].values for col in x_row.columns
                    }

                if vol == "EGARCH":
                    fc_kwargs["method"] = "simulation"
                    fc_kwargs["simulations"] = 1000
                fcast = res.forecast(**fc_kwargs)

                pred_var = float(fcast.variance.iloc[-1, 0])
                forecast_var.append(pred_var)
                actual_var.append(float(self._realized_vol.loc[test_date]))

            except Exception:
                forecast_var.append(np.nan)
                actual_var.append(float(self._realized_vol.loc[test_date]))

        pred = np.array(forecast_var)
        actual = np.array(actual_var)
        mask = ~np.isnan(pred) & ~np.isnan(actual) & (pred > 0)
        pred, actual = pred[mask], actual[mask]

        if len(pred) < 5:
            return {"RMSE": np.nan, "MAE": np.nan, "QLIKE": np.nan,
                    "HitRate": np.nan, "N_valid": int(mask.sum())}

        rmse = float(np.sqrt(np.mean((pred - actual) ** 2)))
        mae = float(np.mean(np.abs(pred - actual)))
        qlike = float(np.mean(np.log(pred) + actual / pred))
        high_vol_threshold = np.median(actual)
        hits = np.sum((pred > high_vol_threshold) == (actual > high_vol_threshold))
        hit_rate = float(hits / len(pred))

        return {
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "QLIKE": round(qlike, 4),
            "HitRate": round(hit_rate, 4),
            "N_valid": int(mask.sum()),
        }

    def print_report(self, oos_table: pd.DataFrame) -> None:
        """OOS sonuç raporunu konsola yazdırır."""
        print(f"\n{'=' * 70}")
        print("  OUT-OF-SAMPLE PERFORMANS RAPORU")
        print(f"{'=' * 70}")
        print(f"  Train: {self.train_size} gün | "
              f"Test: {self.test_size} gün | "
              f"Split: {self.split_date}")
        print(f"  Yöntem: Expanding window, 1-step-ahead forecast\n")

        if oos_table.empty:
            print("  Sonuç yok.")
        else:
            print(oos_table.to_string(index=False))

            best_rmse = oos_table.iloc[0]["Model"]
            best_qlike = oos_table.loc[oos_table["QLIKE"].idxmin(), "Model"]
            print(f"\n  En iyi model (RMSE) : {best_rmse}")
            print(f"  En iyi model (QLIKE): {best_qlike}")

        print(f"{'=' * 70}")
