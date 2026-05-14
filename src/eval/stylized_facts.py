"""Stylized facts en series temporales de retornos financieros.

Módulo puro (sin side effects en import) diseñado para ser reutilizado en:
- F2-T2: caracterización de los datos reales (baseline).
- F4-T11: gate de calidad de los sintéticos generados por TimeGAN
  (`compare_real_vs_synthetic` lo invoca sobre los sintéticos).
- F2-T7: test anti-regresión que asegura fat tails y volatility clustering.

Las funciones aceptan un `DataFrame` de log-returns (columnas = activos) y
devuelven tablas/figuras. La separación notebook ↔ módulo permite que el
notebook quede como orquestador fino.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acf

logger = logging.getLogger(__name__)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calcula log-returns por columna: ``r_t = log(P_t / P_{t-1})``.

    La primera fila resulta NaN y se elimina. Asume índice ordenado y precios > 0.
    """
    if (prices <= 0).any().any():
        raise ValueError("log_returns: precios <=0 detectados")
    returns = np.log(prices / prices.shift(1))
    return returns.dropna(how="all")


def _acf_at_lag(series: pd.Series, lag: int) -> float:
    """ACF de una serie a un lag concreto (descarta NaN)."""
    clean = series.dropna().to_numpy()
    if len(clean) < lag + 2:
        return float("nan")
    values = acf(clean, nlags=lag, fft=True)
    return float(values[lag])


def acf_series(series: pd.Series, max_lag: int = 30) -> np.ndarray:
    """ACF completa de la serie desde lag 0 hasta ``max_lag`` inclusive."""
    clean = series.dropna().to_numpy()
    return acf(clean, nlags=max_lag, fft=True)


def leverage_effect(returns: pd.Series, horizon: int = 5) -> float:
    """Leverage effect: media de ``corr(r_t, |r_{t+k}|)`` para ``k=1..horizon``.

    Signo esperado: negativo (caídas hoy ⇒ mayor vol mañana).
    """
    r = returns.dropna()
    abs_r = r.abs()
    coefs: list[float] = []
    for k in range(1, horizon + 1):
        aligned = pd.concat([r, abs_r.shift(-k)], axis=1).dropna()
        if len(aligned) < 30:
            continue
        coefs.append(float(aligned.corr().iloc[0, 1]))
    return float(np.mean(coefs)) if coefs else float("nan")


def compute_stylized_facts(
    returns: pd.DataFrame,
    *,
    max_lag: int = 30,
    leverage_horizon: int = 5,
) -> pd.DataFrame:
    """Tabla de stylized facts por activo (columnas del DataFrame de entrada).

    Columnas devueltas:
    - ``kurtosis``: kurtosis de Fisher + 3 (i.e. normal = 3, fat tails > 3).
    - ``skew``: asimetría.
    - ``shapiro_pvalue``: p-valor del test de normalidad de Shapiro-Wilk
      (truncado a 5000 muestras por estabilidad numérica del test).
    - ``acf_ret_lag1``: ACF de los retornos a lag 1 (esperado ≈ 0).
    - ``acf_abs_ret_lag1``: ACF de ``|retornos|`` a lag 1 (esperado > 0,
      volatility clustering).
    - ``leverage_effect_k{leverage_horizon}``: media de ``corr(r_t, |r_{t+k}|)``
      para ``k=1..leverage_horizon`` (esperado negativo).

    El ``max_lag`` se usa al pre-calcular ACFs internamente; el resultado solo
    expone lag 1 por activo (lag 1 es el indicador robusto). ACFs completas se
    obtienen vía :func:`acf_series` y se grafican vía :func:`plot_acf_panel`.
    """
    rows: dict[str, dict[str, float]] = {}
    for col in returns.columns:
        series = returns[col].dropna()
        sample = series.to_numpy()
        sample_for_shapiro = sample[:5000] if len(sample) > 5000 else sample
        shapiro_p = (
            float(stats.shapiro(sample_for_shapiro).pvalue)
            if len(sample_for_shapiro) >= 3
            else float("nan")
        )
        rows[col] = {
            "kurtosis": float(stats.kurtosis(sample, fisher=False)),
            "skew": float(stats.skew(sample)),
            "shapiro_pvalue": shapiro_p,
            "acf_ret_lag1": _acf_at_lag(series, lag=1),
            "acf_abs_ret_lag1": _acf_at_lag(series.abs(), lag=1),
            f"leverage_effect_k{leverage_horizon}": leverage_effect(
                series, horizon=leverage_horizon
            ),
        }
    facts = pd.DataFrame.from_dict(rows, orient="index")
    facts.index.name = "asset"
    return facts


def plot_acf_panel(
    returns: pd.DataFrame,
    save_dir: Path,
    *,
    max_lag: int = 30,
    dpi: int = 110,
) -> list[Path]:
    """Para cada columna, guarda figura con 2 subplots: ACF(r) y ACF(|r|).

    Devuelve la lista de paths generados. El import de matplotlib es lazy para
    permitir que el módulo se importe en entornos sin backend gráfico (p.ej.
    el test F2-T7 puede importar `compute_stylized_facts` sin matplotlib).
    """
    import matplotlib.pyplot as plt  # noqa: PLC0415 (lazy import a propósito)

    save_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    lags = np.arange(max_lag + 1)
    for col in returns.columns:
        series = returns[col].dropna()
        acf_r = acf_series(series, max_lag=max_lag)
        acf_abs = acf_series(series.abs(), max_lag=max_lag)

        fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), dpi=dpi)
        axes[0].bar(lags, acf_r, width=0.7, color="#3b6ea5")
        axes[0].axhline(0, color="black", lw=0.6)
        axes[0].set_title(f"ACF de retornos — {col}")
        axes[0].set_xlabel("lag")
        axes[0].set_ylabel("acf")

        axes[1].bar(lags, acf_abs, width=0.7, color="#c0504d")
        axes[1].axhline(0, color="black", lw=0.6)
        axes[1].set_title(f"ACF de |retornos| — {col} (volatility clustering)")
        axes[1].set_xlabel("lag")

        fig.tight_layout()
        path = save_dir / f"acf_{col}.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(path)
        logger.debug("ACF figura guardada en %s", path)
    return paths


def compare_real_vs_synthetic(
    real_returns: pd.DataFrame,
    synthetic_returns: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Comparativa de stylized facts entre real y sintético, columna por columna.

    Diseñado para F4-T11. Asume que ambos DataFrames comparten conjunto de
    columnas (mismos activos). Devuelve tabla multi-index ``(asset, source)``
    para fácil pivoteo.
    """
    if set(real_returns.columns) != set(synthetic_returns.columns):
        raise ValueError(
            "compare_real_vs_synthetic: las columnas no coinciden entre real y sintético"
        )
    real = compute_stylized_facts(real_returns, **kwargs).assign(source="real")
    synth = compute_stylized_facts(synthetic_returns, **kwargs).assign(source="synthetic")
    merged = pd.concat([real, synth], axis=0)
    merged = merged.reset_index().set_index(["asset", "source"]).sort_index()
    return merged
