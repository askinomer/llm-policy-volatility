"""
Açıklanabilirlik (XAI) Modülü — SHAP ve Feature Importance analizi.

GARCH modellerinin parametre duyarlılığını ve belirsizlik skorunun
volatilite üzerindeki etkisini açıklar.
"""

from __future__ import annotations

import logging
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)


class VolatilityExplainer:
    """
    GARCH model sonuçlarını açıklayan ve görselleştiren sınıf.

    SHAP kütüphanesi tree/linear modeller için tasarlandığından,
    GARCH ailesi için doğrudan uygulanamaz. Bunun yerine:
    1. Parametre bazlı katkı ayrıştırması (variance decomposition)
    2. Perturbation-based sensitivity analizi
    3. Event-volatility eşleştirmesi

    Parameters
    ----------
    df : pd.DataFrame
        Tüm verileri içeren merged DataFrame.
    model_results : dict
        GARCHEngine.results çıktısı.
    output_dir : str
        Grafiklerin kaydedileceği dizin.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        model_results: dict,
        output_dir: str = "outputs/figures",
    ) -> None:
        self._df = df
        self._results = model_results
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Parametre Katkı Ayrıştırması
    # ------------------------------------------------------------------

    def variance_decomposition(self, model_name: str = "GARCH-X") -> pd.DataFrame:
        """
        GARCH denklemindeki her terimin koşullu varyansa katkısını hesaplar.

        σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1} + γ·U_t

        Her t zamanı için ω, α·ε², β·σ², γ·U ayrı ayrı hesaplanır.
        """
        result = self._results.get(model_name)
        if result is None:
            logger.warning("Model bulunamadı: %s", model_name)
            return pd.DataFrame()

        params = result.params
        arch_res = result._arch_result

        omega = params.get("omega", 0)
        alpha = params.get("alpha[1]", 0)
        beta = params.get("beta[1]", 0)

        exog_param_name = None
        gamma = 0
        for key in params:
            if key not in ("Const", "omega", "alpha[1]", "beta[1]",
                           "gamma[1]", "eta[1]"):
                exog_param_name = key
                gamma = params[key]
                break

        cond_vol = result.conditional_volatility
        resid = arch_res.resid

        idx = cond_vol.index
        n = len(idx)

        decomp = pd.DataFrame(index=idx)
        decomp["omega_contrib"] = omega
        decomp["arch_contrib"] = 0.0
        decomp["garch_contrib"] = 0.0
        decomp["exog_contrib"] = 0.0
        decomp["cond_variance"] = cond_vol ** 2

        for i in range(1, n):
            t = idx[i]
            t_prev = idx[i - 1]
            decomp.loc[t, "arch_contrib"] = alpha * (resid.loc[t_prev] ** 2)
            decomp.loc[t, "garch_contrib"] = beta * (cond_vol.loc[t_prev] ** 2)

        unc_col = None
        for c in ("avg_uncertainty", "uncertainty_score", "Uncertainty_Score"):
            if c in self._df.columns:
                unc_col = c
                break

        if unc_col and gamma != 0:
            common_idx = idx.intersection(self._df.index)
            decomp.loc[common_idx, "exog_contrib"] = (
                gamma * self._df.loc[common_idx, unc_col]
            )

        decomp["total_explained"] = (
            decomp["omega_contrib"] + decomp["arch_contrib"] +
            decomp["garch_contrib"] + decomp["exog_contrib"]
        )

        logger.info("Variance decomposition tamamlandı: %s", model_name)
        return decomp

    # ------------------------------------------------------------------
    # 2. Sensitivity Analizi (Perturbation-based)
    # ------------------------------------------------------------------

    def sensitivity_analysis(
        self,
        model_name: str = "GARCH-X",
        perturbation_range: tuple = (0.0, 0.25, 0.5, 0.75, 1.0),
    ) -> pd.DataFrame:
        """
        Belirsizlik skorunun farklı seviyelerinde ortalama volatiliteyi ölçer.
        """
        result = self._results.get(model_name)
        if result is None:
            return pd.DataFrame()

        params = result.params
        gamma = 0
        for key in params:
            if key not in ("Const", "omega", "alpha[1]", "beta[1]",
                           "gamma[1]", "eta[1]"):
                gamma = params[key]
                break

        omega = params.get("omega", 0)
        alpha = params.get("alpha[1]", 0)
        beta = params.get("beta[1]", 0)

        unconditional_var = omega / (1 - alpha - beta) if (alpha + beta) < 1 else omega

        rows = []
        for u in perturbation_range:
            adj_var = unconditional_var + gamma * u
            rows.append({
                "uncertainty_level": u,
                "expected_variance": round(adj_var, 6),
                "expected_volatility": round(np.sqrt(max(adj_var, 0)), 6),
                "gamma_contribution": round(gamma * u, 6),
                "pct_change_vs_base": round(
                    (gamma * u / unconditional_var * 100) if unconditional_var > 0 else 0, 2
                ),
            })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 3. Event-Volatility Eşleştirmesi
    # ------------------------------------------------------------------

    def event_volatility_map(
        self,
        model_name: str = "GARCH-X",
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        En yüksek volatilite zıplaması olan günleri tespit eder ve
        o günlerdeki haber/olay bilgisini eşleştirir.
        """
        result = self._results.get(model_name)
        if result is None:
            return pd.DataFrame()

        vol = result.conditional_volatility
        vol_change = vol.diff().abs()
        top_days = vol_change.nlargest(top_n)

        rows = []
        for date, change in top_days.items():
            row = {"date": date, "vol_jump": round(change, 4)}

            if date in self._df.index:
                for col in ("avg_uncertainty", "uncertainty_score"):
                    if col in self._df.columns:
                        row["uncertainty"] = round(self._df.loc[date, col], 4)
                        break
                for col in ("events", "event_type"):
                    if col in self._df.columns:
                        row["event"] = self._df.loc[date, col]
                        break
                if "log_return" in self._df.columns:
                    row["return"] = round(self._df.loc[date, "log_return"], 4)

            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # 4. Görselleştirmeler
    # ------------------------------------------------------------------

    def plot_decomposition(self, model_name: str = "GARCH-X") -> str:
        """Variance decomposition stacked area chart."""
        decomp = self.variance_decomposition(model_name)
        if decomp.empty:
            return ""

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.stackplot(
            decomp.index,
            decomp["omega_contrib"],
            decomp["arch_contrib"],
            decomp["garch_contrib"],
            decomp["exog_contrib"],
            labels=["ω (sabit)", "α·ε² (ARCH)", "β·σ² (GARCH)", "γ·U (Belirsizlik)"],
            colors=["#90a4ae", "#1a73e8", "#7b1fa2", "#e53935"],
            alpha=0.7,
        )

        ax.plot(decomp.index, decomp["cond_variance"],
                color="black", linewidth=1.5, linestyle="--", label="Gerçek σ²")

        ax.set_title(f"Varyans Ayrıştırması — {model_name}", fontweight="bold", fontsize=13)
        ax.set_ylabel("Koşullu Varyans (σ²)")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

        path = str(self._output_dir / f"decomposition_{model_name.replace('-','_')}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Decomposition grafiği kaydedildi: %s", path)
        return path

    def plot_sensitivity(self, model_name: str = "GARCH-X") -> str:
        """Sensitivity analizi bar chart."""
        sens = self.sensitivity_analysis(model_name)
        if sens.empty:
            return ""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        colors = plt.cm.RdYlGn_r(sens["uncertainty_level"])
        ax1.bar(sens["uncertainty_level"].astype(str), sens["expected_volatility"],
                color=colors, edgecolor="white", linewidth=0.5)
        ax1.set_title("Belirsizlik Seviyesine Göre Beklenen Volatilite", fontweight="bold")
        ax1.set_xlabel("Belirsizlik Skoru")
        ax1.set_ylabel("Beklenen Volatilite")
        ax1.grid(True, alpha=0.3, axis="y")

        ax2.bar(sens["uncertainty_level"].astype(str), sens["pct_change_vs_base"],
                color=colors, edgecolor="white", linewidth=0.5)
        ax2.set_title("Bazal Varyansa Göre % Değişim", fontweight="bold")
        ax2.set_xlabel("Belirsizlik Skoru")
        ax2.set_ylabel("% Değişim")
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.axhline(y=0, color="black", linewidth=0.5)

        fig.suptitle(f"Sensitivity Analizi — {model_name}", fontsize=14, fontweight="bold")
        fig.tight_layout()

        path = str(self._output_dir / f"sensitivity_{model_name.replace('-','_')}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Sensitivity grafiği kaydedildi: %s", path)
        return path

    def plot_event_timeline(self, model_name: str = "GARCH-X", top_n: int = 10) -> str:
        """Volatilite zıplamaları + olay etiketleri timeline."""
        result = self._results.get(model_name)
        if result is None:
            return ""

        vol = result.conditional_volatility
        events = self.event_volatility_map(model_name, top_n)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(vol.index, vol, color="#7b1fa2", linewidth=1.5, label=f"{model_name} Volatilite")

        for _, row in events.iterrows():
            date = row["date"]
            if date in vol.index:
                ax.annotate(
                    row.get("event", "?")[:25] if pd.notna(row.get("event")) else "?",
                    xy=(date, vol.loc[date]),
                    xytext=(0, 20), textcoords="offset points",
                    fontsize=7, ha="center", rotation=45,
                    arrowprops=dict(arrowstyle="->", color="#e53935", lw=0.8),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#fff3e0", alpha=0.8),
                )

        ax.set_title(f"Volatilite Zıplamaları ve Olaylar — {model_name}",
                      fontweight="bold", fontsize=13)
        ax.set_ylabel("Volatilite")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %y"))

        path = str(self._output_dir / f"event_timeline_{model_name.replace('-','_')}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Event timeline kaydedildi: %s", path)
        return path

    # ------------------------------------------------------------------
    # Tam rapor
    # ------------------------------------------------------------------

    def generate_all(self, model_name: str = "GARCH-X") -> dict[str, str]:
        """Tüm XAI görselleştirmelerini üretir ve dosya yollarını döner."""
        paths = {}
        paths["decomposition"] = self.plot_decomposition(model_name)
        paths["sensitivity"] = self.plot_sensitivity(model_name)
        paths["event_timeline"] = self.plot_event_timeline(model_name)

        logger.info("Tüm XAI grafikleri üretildi: %d dosya.", len(paths))
        return paths

    def print_summary(self, model_name: str = "GARCH-X") -> None:
        """Konsola XAI özet raporu yazdırır."""
        print("\n" + "=" * 70)
        print(f"  XAI RAPORU — {model_name}")
        print("=" * 70)

        sens = self.sensitivity_analysis(model_name)
        if not sens.empty:
            print("\n  Sensitivity Analizi:")
            print("  " + sens.to_string(index=False).replace("\n", "\n  "))

        events = self.event_volatility_map(model_name, top_n=5)
        if not events.empty:
            print(f"\n  En Büyük {len(events)} Volatilite Zıplaması:")
            print("  " + events.to_string(index=False).replace("\n", "\n  "))

        print("=" * 70)
