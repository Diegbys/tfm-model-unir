"""Costes de transacción del entorno de cartera — F5-T4.

Funciones puras: el *turnover* mide cuánto cambia la cartera tras el rebalanceo
y el coste lo multiplica por la comisión. Es config-driven (10 bps por defecto,
ablation con 5/20 bps cambiando ``transaction_cost_pct``).

Ver ``Plan_Implementacion_Fase2_TFE.md`` §Fase 5 (F5-T4) y reporte §3.3.
"""
from __future__ import annotations

import numpy as np


def compute_turnover(w_new: np.ndarray, w_drifted: np.ndarray) -> float:
    """Turnover entre la cartera objetivo y la cartera tras el drift.

    ``turnover = 0.5 * sum(|w_new - w_drifted|)`` (turnover bilateral / 2).
    Vale 0 si no hay rebalanceo y 1 si se rota toda la cartera de un activo
    a otro distinto.

    Parameters
    ----------
    w_new:
        Pesos objetivo tras la acción del agente (proyectados al simplex).
    w_drifted:
        Pesos de la cartera anterior tras aplicar el drift del retorno del día.
    """
    w_new = np.asarray(w_new, dtype=np.float64)
    w_drifted = np.asarray(w_drifted, dtype=np.float64)
    if w_new.shape != w_drifted.shape:
        raise ValueError(
            f"compute_turnover: shapes distintos {w_new.shape} vs {w_drifted.shape}"
        )
    return float(0.5 * np.sum(np.abs(w_new - w_drifted)))


def transaction_cost(
    turnover: float,
    transaction_cost_pct: float,
    slippage_pct: float = 0.0,
) -> float:
    """Coste de transacción ``turnover * (transaction_cost_pct + slippage_pct)``.

    Siempre ``>= 0``; con ``turnover=0`` el coste es 0. Con la config default
    (``transaction_cost_pct=0.001``, 10 bps) un turnover de 1 cuesta 0.001.
    """
    if turnover < 0:
        raise ValueError(f"transaction_cost: turnover negativo ({turnover})")
    if transaction_cost_pct < 0 or slippage_pct < 0:
        raise ValueError("transaction_cost: porcentajes negativos no permitidos")
    return float(turnover * (transaction_cost_pct + slippage_pct))
