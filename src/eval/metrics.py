"""Métricas de desempeño de cartera — Fase 6 (semilla de F7-T2).

Versión mínima necesaria para el ``ValidationEvalCallback`` (F6-T6): Sharpe ratio
y máximo drawdown sobre la curva de un episodio de validación. **F7-T2 extenderá
este módulo** a la batería completa de 9 métricas (Sortino, Calmar, CVaR, etc.).
"""
from __future__ import annotations

import numpy as np

TRADING_DAYS_PER_YEAR = 252

# Por debajo de este umbral la "varianza" es ruido de coma flotante sobre una
# serie constante, no dispersión real (los retornos diarios reales son ~1e-2).
_MIN_STD = 1e-12


def sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    rf: float = 0.0,
) -> float:
    """Sharpe ratio anualizado de una serie de retornos periódicos.

    ``sharpe = (mean(returns) - rf) / std(returns) * sqrt(periods_per_year)``
    (compass §4.4), con desviación típica muestral (``ddof=1``).

    Caso degenerado: con menos de 2 retornos o varianza nula devuelve ``0.0`` —
    sin dispersión no hay señal ajustada por riesgo que comparar entre checkpoints.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    std = r.std(ddof=1)
    if std <= _MIN_STD:
        return 0.0
    return float((r.mean() - rf) / std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """Máximo drawdown de una curva de riqueza, como fracción **negativa**.

    ``MDD = min_t (V_t - max_{s<=t} V_s) / max_{s<=t} V_s`` (compass §4.4). El
    resultado es ≤ 0: ``-0.5`` significa una caída peak-to-trough del 50 %. Una
    curva no decreciente devuelve ``0.0``.
    """
    v = np.asarray(equity_curve, dtype=np.float64)
    if v.size < 2:
        return 0.0
    running_max = np.maximum.accumulate(v)
    drawdown = (v - running_max) / running_max
    return float(drawdown.min())
