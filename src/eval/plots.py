"""Plots comparativos del backtest — F7-T6.

Cuatro figuras que alimentan los Capítulos 5–6 de la memoria: curvas de equity,
curvas de drawdown, distribución de retornos diarios y barras de métricas. Las
funciones son puras — reciben los DataFrames del backtest (F7-T1/T4) y devuelven
un ``matplotlib.figure.Figure`` sin escribir a disco; el script 06 las persiste.

El backend de matplotlib lo fija el llamador (``Agg`` en script y tests).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

_COLOR_A = "#1f77b4"  # Agente A (real)
_COLOR_B = "#d62728"  # Agente B (real + sintético)
_BAND_ALPHA = 0.20


def _seed_band(
    backtests: list[pd.DataFrame], column: str
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Alinea ``column`` entre seeds y devuelve (fechas, mediana, q25, q75)."""
    aligned = pd.concat([bt[column] for bt in backtests], axis=1)
    return (
        aligned.index,
        aligned.median(axis=1).to_numpy(),
        aligned.quantile(0.25, axis=1).to_numpy(),
        aligned.quantile(0.75, axis=1).to_numpy(),
    )


def _plot_band(ax, backtests, column, color, label) -> None:
    """Dibuja la mediana entre seeds con una banda IQR sombreada."""
    dates, median, q25, q75 = _seed_band(backtests, column)
    ax.plot(dates, median, color=color, lw=1.8, label=f"{label} (mediana)")
    ax.fill_between(dates, q25, q75, color=color, alpha=_BAND_ALPHA, label=f"{label} IQR")


def plot_equity_curves(
    backtests_a: list[pd.DataFrame],
    backtests_b: list[pd.DataFrame],
    baselines: dict[str, pd.DataFrame] | None = None,
) -> Figure:
    """Curvas de riqueza A vs B (mediana + banda IQR entre seeds) + baselines."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    _plot_band(ax, backtests_a, "wealth", _COLOR_A, "Agente A")
    _plot_band(ax, backtests_b, "wealth", _COLOR_B, "Agente B")
    for name, bt in (baselines or {}).items():
        ax.plot(bt.index, bt["wealth"], ls="--", lw=1.0, alpha=0.8, label=name)
    ax.axhline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_title("Curva de equity sobre test out-of-sample (2023–2025)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Riqueza (V₀ = 1)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_drawdown_curves(
    backtests_a: list[pd.DataFrame],
    backtests_b: list[pd.DataFrame],
    baselines: dict[str, pd.DataFrame] | None = None,
) -> Figure:
    """Curvas de drawdown A vs B (mediana + banda IQR entre seeds) + baselines."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    _plot_band(ax, backtests_a, "drawdown", _COLOR_A, "Agente A")
    _plot_band(ax, backtests_b, "drawdown", _COLOR_B, "Agente B")
    for name, bt in (baselines or {}).items():
        ax.plot(bt.index, bt["drawdown"], ls="--", lw=1.0, alpha=0.8, label=name)
    ax.axhline(0.0, color="grey", lw=0.8, ls=":")
    ax.set_title("Drawdown sobre test out-of-sample (2023–2025)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_return_distributions(
    backtests_a: list[pd.DataFrame],
    backtests_b: list[pd.DataFrame],
) -> Figure:
    """KDE de los retornos diarios agrupados de todos los seeds, A vs B."""
    fig, ax = plt.subplots(figsize=(8, 5))
    pooled = {}
    for label, backtests, color in (
        ("Agente A", backtests_a, _COLOR_A),
        ("Agente B", backtests_b, _COLOR_B),
    ):
        returns = np.concatenate(
            [bt["portfolio_return"].to_numpy() for bt in backtests]
        )
        pooled[label] = returns
        grid = np.linspace(returns.min(), returns.max(), 256)
        density = gaussian_kde(returns)(grid)
        ax.plot(grid, density, color=color, lw=1.8, label=label)
        ax.fill_between(grid, density, color=color, alpha=_BAND_ALPHA)
        ax.axvline(returns.mean(), color=color, ls=":", lw=1.0)
    ax.set_title("Distribución de retornos diarios (KDE, todos los seeds)")
    ax.set_xlabel("Retorno diario neto")
    ax.set_ylabel("Densidad")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


_DEFAULT_BAR_METRICS = [
    "annualized_return", "sharpe_ratio", "sortino_ratio",
    "calmar_ratio", "max_drawdown", "cvar_95",
]


def plot_metrics_bars(
    metrics_a: pd.DataFrame,
    metrics_b: pd.DataFrame,
    metrics: list[str] | None = None,
) -> Figure:
    """Barras comparativas A vs B: media entre seeds con error bars (± std)."""
    metrics = metrics or _DEFAULT_BAR_METRICS
    x = np.arange(len(metrics))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        x - width / 2, metrics_a[metrics].mean(), width,
        yerr=metrics_a[metrics].std(), capsize=3, color=_COLOR_A, label="Agente A",
    )
    ax.bar(
        x + width / 2, metrics_b[metrics].mean(), width,
        yerr=metrics_b[metrics].std(), capsize=3, color=_COLOR_B, label="Agente B",
    )
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_title("Métricas de cartera sobre test — media ± std entre seeds")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig
