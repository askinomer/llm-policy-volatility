"""
InteractiveDashboard — Plotly tabanlı etkileşimli görselleştirme.

6 panelli dashboard: Fiyat, Getiri, Belirsizlik, Volatilite karşılaştırma,
Scatter korelasyon, Model benchmark tablosu.
"""

from __future__ import annotations

import logging
from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


class InteractiveDashboard:
    """
    Plotly ile etkileşimli finansal dashboard oluşturan sınıf.

    Parameters
    ----------
    df : pd.DataFrame
        Merged veri (close, log_return, avg_uncertainty, Vol_* kolonları).
    benchmark_table : pd.DataFrame, optional
        Model karşılaştırma tablosu.
    title : str
        Dashboard başlığı.
    """

    _COLORS = {
        "price": "#1a73e8",
        "return_pos": "#43a047",
        "return_neg": "#e53935",
        "uncertainty": "#ff9800",
        "uncertainty_sma": "#d84315",
        "garch": "#7b1fa2",
        "egarch": "#0288d1",
        "tarch": "#2e7d32",
        "garch_x": "#e53935",
        "egarch_x": "#f06292",
        "tarch_x": "#ff7043",
        "realized": "#90a4ae",
    }

    def __init__(
        self,
        df: pd.DataFrame,
        benchmark_table: Optional[pd.DataFrame] = None,
        title: str = "Politik Belirsizlik ve Finansal Volatilite Dashboard",
    ) -> None:
        self._df = df
        self._bench = benchmark_table
        self._title = title

    # ------------------------------------------------------------------
    # Ana dashboard
    # ------------------------------------------------------------------

    def build(self) -> go.Figure:
        """6 panelli tam dashboard figürü oluşturur."""
        has_bench = self._bench is not None and not self._bench.empty

        specs = [
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy", "colspan": 2}, None],
            [{"type": "xy"}, {"type": "table"}] if has_bench else [{"type": "xy"}, {"type": "xy"}],
        ]
        subtitles = [
            "Kapanış Fiyatı", "Günlük Getiriler (%)",
            "NLP Belirsizlik Skoru",
            "Volatilite Karşılaştırması",
            "Belirsizlik vs Volatilite",
            "Model Benchmark" if has_bench else "Getiri Dağılımı",
        ]

        fig = make_subplots(
            rows=4, cols=2, specs=specs,
            subplot_titles=subtitles,
            vertical_spacing=0.06,
            horizontal_spacing=0.08,
        )

        self._add_price_panel(fig, row=1, col=1)
        self._add_returns_panel(fig, row=1, col=2)
        self._add_uncertainty_panel(fig, row=2, col=1)
        self._add_volatility_panel(fig, row=3, col=1)
        self._add_scatter_panel(fig, row=4, col=1)

        if has_bench:
            self._add_benchmark_table(fig, row=4, col=2)
        else:
            self._add_return_dist_panel(fig, row=4, col=2)

        fig.update_layout(
            height=1400,
            width=1200,
            title_text=self._title,
            title_font_size=16,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.02, x=0.5, xanchor="center"),
            template="plotly_white",
            hovermode="x unified",
        )

        logger.info("Dashboard oluşturuldu: 6 panel.")
        return fig

    # ------------------------------------------------------------------
    # Panel fonksiyonları
    # ------------------------------------------------------------------

    def _add_price_panel(self, fig: go.Figure, row: int, col: int) -> None:
        df = self._df
        if "close" not in df.columns:
            return
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df["close"],
                mode="lines", name="Kapanış",
                line=dict(color=self._COLORS["price"], width=1.5),
                showlegend=False,
            ),
            row=row, col=col,
        )
        fig.update_yaxes(title_text="Fiyat (TL)", row=row, col=col)

    def _add_returns_panel(self, fig: go.Figure, row: int, col: int) -> None:
        df = self._df
        ret_col = "log_return" if "log_return" in df.columns else "Return"
        if ret_col not in df.columns:
            return

        colors = [self._COLORS["return_neg"] if v < 0
                  else self._COLORS["return_pos"] for v in df[ret_col]]
        fig.add_trace(
            go.Bar(
                x=df.index, y=df[ret_col],
                marker_color=colors, name="Getiri",
                showlegend=False, opacity=0.7,
            ),
            row=row, col=col,
        )
        fig.update_yaxes(title_text="Getiri (%)", row=row, col=col)

    def _add_uncertainty_panel(self, fig: go.Figure, row: int, col: int) -> None:
        df = self._df
        unc_col = None
        for c in ("avg_uncertainty", "uncertainty_score", "Uncertainty_Score"):
            if c in df.columns:
                unc_col = c
                break
        if unc_col is None:
            return

        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[unc_col],
                mode="markers", name="Belirsizlik Skoru",
                marker=dict(
                    color=df[unc_col], colorscale="RdYlGn_r",
                    size=6, opacity=0.6,
                    colorbar=dict(title="Skor", x=1.02, len=0.25, y=0.62),
                    cmin=0, cmax=1,
                ),
            ),
            row=row, col=col,
        )

        sma_col = None
        for c in ("uncertainty_sma", "Uncertainty_SMA"):
            if c in df.columns:
                sma_col = c
                break
        if sma_col is None and unc_col in df.columns:
            sma = df[unc_col].rolling(window=20, min_periods=1).mean()
        else:
            sma = df.get(sma_col)

        if sma is not None:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=sma,
                    mode="lines", name="Belirsizlik SMA(20)",
                    line=dict(color=self._COLORS["uncertainty_sma"], width=2),
                ),
                row=row, col=col,
            )

        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                       opacity=0.4, row=row, col=col)
        fig.update_yaxes(title_text="Skor (0-1)", range=[-0.05, 1.05],
                          row=row, col=col)

    def _add_volatility_panel(self, fig: go.Figure, row: int, col: int) -> None:
        df = self._df
        vol_map = {
            "Vol_GARCH": ("GARCH", self._COLORS["garch"]),
            "Vol_EGARCH": ("EGARCH", self._COLORS["egarch"]),
            "Vol_TARCH": ("TARCH", self._COLORS["tarch"]),
            "Vol_GARCH-X": ("GARCH-X", self._COLORS["garch_x"]),
            "Vol_EGARCH-X": ("EGARCH-X", self._COLORS["egarch_x"]),
            "Vol_TARCH-X": ("TARCH-X", self._COLORS["tarch_x"]),
        }

        for col_name, (label, color) in vol_map.items():
            if col_name in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[col_name],
                        mode="lines", name=label,
                        line=dict(color=color, width=1.5 if "-X" not in label else 2.5),
                        opacity=0.7 if "-X" not in label else 1.0,
                    ),
                    row=row, col=col,
                )

        for c in ("Realized_Vol", "realized_vol", "rolling_vol"):
            if c in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[c],
                        mode="lines", name="Realized Vol",
                        line=dict(color=self._COLORS["realized"], width=1, dash="dash"),
                    ),
                    row=row, col=col,
                )
                break

        fig.update_yaxes(title_text="Volatilite", row=row, col=col)

    def _add_scatter_panel(self, fig: go.Figure, row: int, col: int) -> None:
        df = self._df
        unc_col = None
        for c in ("avg_uncertainty", "uncertainty_score", "Uncertainty_Score"):
            if c in df.columns:
                unc_col = c
                break

        vol_col = None
        for c in ("Vol_GARCH-X", "Vol_GARCH_X", "Vol_GARCH"):
            if c in df.columns:
                vol_col = c
                break

        if unc_col is None or vol_col is None:
            return

        corr = df[unc_col].corr(df[vol_col])

        fig.add_trace(
            go.Scatter(
                x=df[unc_col], y=df[vol_col],
                mode="markers", name=f"r = {corr:.3f}",
                marker=dict(color=self._COLORS["garch_x"], size=5,
                            opacity=0.5, line=dict(width=0.3, color="white")),
                showlegend=True,
            ),
            row=row, col=col,
        )

        z = np.polyfit(df[unc_col].values, df[vol_col].values, 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[unc_col].min(), df[unc_col].max(), 50)
        fig.add_trace(
            go.Scatter(
                x=x_line, y=p(x_line),
                mode="lines", name="Trend",
                line=dict(color="black", width=1.5, dash="dash"),
                showlegend=False,
            ),
            row=row, col=col,
        )

        fig.update_xaxes(title_text="Belirsizlik Skoru", row=row, col=col)
        fig.update_yaxes(title_text="Volatilite", row=row, col=col)

    def _add_benchmark_table(self, fig: go.Figure, row: int, col: int) -> None:
        if self._bench is None:
            return

        header_vals = list(self._bench.columns)
        cell_vals = [self._bench[c].tolist() for c in self._bench.columns]

        fig.add_trace(
            go.Table(
                header=dict(
                    values=header_vals,
                    fill_color="#1a73e8",
                    font=dict(color="white", size=11),
                    align="center",
                ),
                cells=dict(
                    values=cell_vals,
                    fill_color=[["#f8f9fa", "white"] * (len(self._bench) // 2 + 1)],
                    font=dict(size=10),
                    align="center",
                ),
            ),
            row=row, col=col,
        )

    def _add_return_dist_panel(self, fig: go.Figure, row: int, col: int) -> None:
        df = self._df
        ret_col = "log_return" if "log_return" in df.columns else "Return"
        if ret_col not in df.columns:
            return

        fig.add_trace(
            go.Histogram(
                x=df[ret_col], nbinsx=40, name="Getiri Dağılımı",
                marker_color=self._COLORS["price"], opacity=0.7,
                showlegend=False,
            ),
            row=row, col=col,
        )
        fig.update_xaxes(title_text="Getiri (%)", row=row, col=col)
        fig.update_yaxes(title_text="Frekans", row=row, col=col)

    # ------------------------------------------------------------------
    # Kaydetme / Gösterme
    # ------------------------------------------------------------------

    def save_html(self, path: str = "outputs/figures/dashboard.html") -> str:
        """Dashboard'u interaktif HTML olarak kaydeder."""
        fig = self.build()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(path, include_plotlyjs="cdn")
        logger.info("Dashboard kaydedildi: %s", path)
        return path

    def save_png(self, path: str = "outputs/figures/dashboard.png", **kwargs) -> str:
        """Dashboard'u statik PNG olarak kaydeder (kaleido gerekir)."""
        fig = self.build()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.write_image(path, width=1200, height=1400, scale=2, **kwargs)
            logger.info("PNG kaydedildi: %s", path)
        except Exception as exc:
            logger.warning("PNG kaydetme başarısız (kaleido yüklü mü?): %s", exc)
            html_path = path.replace(".png", ".html")
            fig.write_html(html_path, include_plotlyjs="cdn")
            logger.info("Fallback HTML kaydedildi: %s", html_path)
            return html_path
        return path

    def show(self) -> None:
        """Dashboard'u tarayıcıda açar."""
        fig = self.build()
        fig.show()
