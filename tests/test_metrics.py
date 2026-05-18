"""Tests de la batería de métricas de cartera — F7-T8.

Validan las funciones puras de ``src/eval/metrics.py`` contra valores conocidos
(ADR §6.6 punto 7). Siguen las convenciones de ``tests/test_env.py``: docstrings
en español, ``pytest.approx`` y asserts con mensaje.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.eval.metrics import (
    annualized_return,
    annualized_turnover,
    annualized_vol,
    calmar_ratio,
    compute_all_metrics,
    cvar_95,
    max_drawdown,
    recovery_time,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)

TRADING_DAYS = 252


# --- Sharpe ratio (función existente de F6.7) --------------------------------

def test_sharpe_retornos_casi_constantes_es_enorme() -> None:
    """F7-T8(a): retornos positivos con σ=ε (≫ umbral degenerado) ⇒ Sharpe enorme.

    ``sharpe_ratio`` de F6.7 devuelve 0.0 solo si σ ≤ 1e-12; con σ≈1e-6 (el
    "σ=ε" del ADR) la señal pasa la guarda y el Sharpe tiende a infinito.
    """
    rng = np.random.default_rng(0)
    returns = np.full(200, 0.01) + rng.normal(0.0, 1e-6, 200)
    assert sharpe_ratio(returns) > 1e3, "Sharpe de serie casi-constante debe ser enorme"


def test_sharpe_de_normal_simulada_coincide_con_formula() -> None:
    """F7-T8(b): para r ~ N(μ,σ), Sharpe ≈ μ/σ·√252."""
    rng = np.random.default_rng(42)
    mu, sigma = 5e-4, 1e-2
    returns = rng.normal(mu, sigma, 50_000)
    esperado = mu / sigma * math.sqrt(TRADING_DAYS)
    assert sharpe_ratio(returns) == pytest.approx(esperado, rel=0.05)


# --- Maximum drawdown (función existente de F6.7) ----------------------------

def test_mdd_curva_monotona_creciente_es_cero() -> None:
    """F7-T8(c): una curva de riqueza no decreciente tiene MDD = 0."""
    assert max_drawdown(np.array([1.0, 1.1, 1.2, 1.5, 2.0])) == pytest.approx(0.0)


def test_mdd_caso_conocido() -> None:
    """F7-T8(d): MDD de [100, 110, 80, 90] = (80-110)/110 = -27.27 %."""
    assert max_drawdown(np.array([100.0, 110.0, 80.0, 90.0])) == pytest.approx(
        -30.0 / 110.0, rel=1e-6
    )


# --- Annualized return -------------------------------------------------------

def test_annualized_return_duplicar_en_un_anio_es_100pct() -> None:
    """252 días que duplican la riqueza ⇒ retorno anualizado ≈ 100 %."""
    r_diario = 2.0 ** (1.0 / TRADING_DAYS) - 1.0
    returns = np.full(TRADING_DAYS, r_diario)
    assert annualized_return(returns) == pytest.approx(1.0, rel=1e-6)


def test_annualized_return_retornos_cero_es_cero() -> None:
    """Una serie de retornos nulos tiene retorno anualizado 0."""
    assert annualized_return(np.zeros(100)) == pytest.approx(0.0)


# --- Annualized volatility ---------------------------------------------------

def test_annualized_vol_escala_por_raiz_252() -> None:
    """La vol anualizada es la desviación típica muestral × √252."""
    rng = np.random.default_rng(1)
    returns = rng.normal(0.0, 0.02, 1000)
    esperado = returns.std(ddof=1) * math.sqrt(TRADING_DAYS)
    assert annualized_vol(returns) == pytest.approx(esperado, rel=1e-9)


# --- Sortino ratio -----------------------------------------------------------

def test_sortino_solo_penaliza_la_desviacion_negativa() -> None:
    """Sortino usa la desviación de los retornos negativos como denominador."""
    returns = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    negativos = returns[returns < 0.0]
    downside = math.sqrt(np.mean(negativos**2))
    esperado = returns.mean() / downside * math.sqrt(TRADING_DAYS)
    assert sortino_ratio(returns) == pytest.approx(esperado, rel=1e-9)


def test_sortino_sin_retornos_negativos_es_cero() -> None:
    """Sin downside la guarda devuelve 0.0 (coherente con sharpe_ratio)."""
    assert sortino_ratio(np.array([0.01, 0.02, 0.03])) == pytest.approx(0.0)


# --- Calmar ratio ------------------------------------------------------------

def test_calmar_es_retorno_anualizado_sobre_mdd() -> None:
    """Calmar = retorno anualizado / |MDD|, con la curva de equity de los retornos."""
    returns = np.array([0.05, -0.10, 0.03, 0.04, -0.02, 0.06])
    equity = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    esperado = annualized_return(returns) / abs(max_drawdown(equity))
    assert calmar_ratio(returns) == pytest.approx(esperado, rel=1e-9)


# --- CVaR 95 % ---------------------------------------------------------------

def test_cvar_95_es_la_media_del_5pct_peor() -> None:
    """CVaR-95 = media del 5 % de retornos más bajos."""
    returns = np.arange(100, dtype=np.float64) / 100.0  # 0.00 .. 0.99
    # 5 % de 100 = 5 peores: 0.00, 0.01, 0.02, 0.03, 0.04 ⇒ media 0.02.
    assert cvar_95(returns) == pytest.approx(0.02, rel=1e-9)


# --- Annualized turnover -----------------------------------------------------

def test_annualized_turnover_escala_la_media_diaria() -> None:
    """El turnover anualizado es la media del turnover diario × 252."""
    turnover = np.array([0.1, 0.2, 0.3, 0.0])
    assert annualized_turnover(turnover) == pytest.approx(0.15 * TRADING_DAYS)


# --- Win rate ----------------------------------------------------------------

def test_win_rate_es_fraccion_de_dias_positivos() -> None:
    """Win rate = proporción de días con retorno estrictamente positivo."""
    returns = np.array([0.01, -0.02, 0.03, 0.00, 0.05])  # 3 de 5 > 0
    assert win_rate(returns) == pytest.approx(0.6)


# --- Recovery time -----------------------------------------------------------

def test_recovery_time_es_la_duracion_del_drawdown_mas_largo() -> None:
    """Recovery time = nº de pasos del drawdown más largo hasta recuperar el pico."""
    # Pico 1.5 en idx 1; vuelve a 1.5 en idx 4 ⇒ duración 4-1 = 3.
    equity = np.array([1.0, 1.5, 1.0, 1.2, 1.5, 2.0])
    assert recovery_time(equity) == 3


def test_recovery_time_curva_creciente_es_cero() -> None:
    """Una curva sin drawdowns tiene recovery time 0."""
    assert recovery_time(np.array([1.0, 1.1, 1.2, 1.3])) == 0


# --- Agregador ---------------------------------------------------------------

def test_compute_all_metrics_devuelve_las_nueve_metricas() -> None:
    """``compute_all_metrics`` agrega todas las métricas en un dict."""
    rng = np.random.default_rng(7)
    returns = rng.normal(2e-4, 1e-2, 300)
    equity = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    turnover = rng.uniform(0.0, 0.3, 300)
    out = compute_all_metrics(returns, equity, turnover)
    esperadas = {
        "annualized_return", "annualized_vol", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "calmar_ratio", "cvar_95", "annualized_turnover",
        "win_rate", "recovery_time",
    }
    assert esperadas.issubset(out.keys()), f"faltan métricas: {esperadas - out.keys()}"
    assert out["sharpe_ratio"] == pytest.approx(sharpe_ratio(returns))
