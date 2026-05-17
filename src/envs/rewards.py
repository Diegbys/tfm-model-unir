"""Función de recompensa del entorno de cartera — F5-T5.

    reward_t = log(1 + portfolio_return - cost) = log(V_t / V_{t-1})

El coste entra **dentro** del argumento del logaritmo (no restado por fuera).
Así ``V_t = V_{t-1} * exp(reward_t)`` y se cumple **exacto** el invariante de
consistencia de F5-T9 (``wealth_T / wealth_0 == exp(sum(rewards))``).

Nota de discrepancia con el ADR: F5-T5 escribe ``reward = log(V_t/V_{t-1}) -
cost_t`` con ``V_t`` ya neto de coste, lo que contaría el coste dos veces. Se
adopta la formulación de arriba — económicamente correcta (log-retorno realizado
neto de costes) y la única que satisface el test de consistencia F5-T9. Ver
``Plan_Implementacion_Fase2_TFE.md`` §Fase 5 y el plan de implementación.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Epsilon de seguridad para el argumento del logaritmo: evita log(<=0) -> -inf/nan
# en el caso (irreal a frecuencia diaria) de ruina total de la cartera.
_MIN_WEALTH_FACTOR = 1e-8


def log_return_minus_costs(portfolio_return: float, cost: float) -> float:
    """Log-retorno realizado de la cartera, neto de costes de transacción.

    ``reward = log(1 + portfolio_return - cost)``, donde
    ``portfolio_return = w_t · r_t`` (retorno simple de la cartera el día t) y
    ``cost`` es el coste de transacción del paso.

    Si ``1 + portfolio_return - cost <= 0`` (ruina total, irreal a frecuencia
    diaria) el argumento se clipa a un epsilon positivo y se registra un WARNING,
    evitando un ``log`` no finito.
    """
    wealth_factor = 1.0 + portfolio_return - cost
    if wealth_factor <= _MIN_WEALTH_FACTOR:
        logger.warning(
            "log_return_minus_costs: factor de riqueza no positivo (%.6f); "
            "clip a %.0e. portfolio_return=%.6f cost=%.6f",
            wealth_factor, _MIN_WEALTH_FACTOR, portfolio_return, cost,
        )
        wealth_factor = _MIN_WEALTH_FACTOR
    return float(np.log(wealth_factor))
