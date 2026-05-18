"""Métricas de desempeño de cartera — F7-T2 (batería completa).

Funciones puras sobre una serie de retornos diarios (o curva de riqueza / serie
de turnover). ``sharpe_ratio`` y ``max_drawdown`` nacieron en F6.7 como semilla
para el ``ValidationEvalCallback``; F7-T2 extiende el módulo a las 9 métricas del
ADR §4.4 (más ``recovery_time``) y añade ``compute_all_metrics`` como agregador.

Definiciones (compass §4.4). Convención de casos degenerados: cuando no hay
dispersión sobre la que medir riesgo (varianza nula, sin downside, MDD nulo) las
ratios devuelven ``0.0`` — coherente con ``sharpe_ratio`` de F6.7, evita ``inf``
al promediar entre seeds.
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


def annualized_return(
    returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Retorno anualizado **geométrico** de una serie de retornos diarios.

    ``ann = prod(1 + r) ** (periods_per_year / n) - 1``. Con una serie vacía
    devuelve ``0.0``.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    total_growth = float(np.prod(1.0 + r))
    return total_growth ** (periods_per_year / r.size) - 1.0


def annualized_vol(
    returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Volatilidad anualizada: desviación típica muestral × √periods_per_year."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    rf: float = 0.0,
) -> float:
    """Sortino ratio anualizado: como Sharpe pero penaliza solo el downside.

    ``sortino = mean(r - rf) / downside_dev * sqrt(periods_per_year)`` donde
    ``downside_dev = sqrt(mean(neg**2))`` sobre los retornos en exceso negativos.
    Sin retornos negativos (o con menos de 2 puntos) devuelve ``0.0``.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return 0.0
    excess = r - rf
    negatives = excess[excess < 0.0]
    if negatives.size == 0:
        return 0.0
    downside_dev = np.sqrt(np.mean(negatives**2))
    if downside_dev <= _MIN_STD:
        return 0.0
    return float(excess.mean() / downside_dev * np.sqrt(periods_per_year))


def calmar_ratio(
    returns: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Calmar ratio: retorno anualizado / |máximo drawdown|.

    La curva de riqueza se reconstruye desde ``V_0 = 1`` acumulando los retornos.
    Si la curva no tiene drawdown (MDD ≈ 0) devuelve ``0.0``.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    equity = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    mdd = max_drawdown(equity)
    if abs(mdd) <= _MIN_STD:
        return 0.0
    return float(annualized_return(r, periods_per_year) / abs(mdd))


def cvar_95(returns: np.ndarray, alpha: float = 0.95) -> float:
    """CVaR / Expected Shortfall: media del ``(1-alpha)`` peor de los retornos.

    Con ``alpha=0.95`` promedia el 5 % de retornos diarios más bajos (riesgo de
    cola). Devuelve un valor típicamente negativo.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    # round (no ceil): evita que el error de coma flotante de (1-alpha) empuje
    # el tamaño de la cola un elemento de más (p. ej. (1-0.95)*100 → 5.0000004).
    k = max(1, int(round((1.0 - alpha) * r.size)))
    worst = np.sort(r)[:k]
    return float(worst.mean())


def annualized_turnover(
    turnover: np.ndarray, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Turnover anualizado: media del turnover diario × periods_per_year."""
    t = np.asarray(turnover, dtype=np.float64)
    if t.size == 0:
        return 0.0
    return float(t.mean() * periods_per_year)


def win_rate(returns: np.ndarray) -> float:
    """Fracción de días con retorno estrictamente positivo."""
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    return float(np.mean(r > 0.0))


def recovery_time(equity_curve: np.ndarray) -> int:
    """Duración (en días) del drawdown más largo: del pico previo hasta recuperarlo.

    Recorre los tramos en los que la riqueza está estrictamente por debajo de su
    máximo previo. La duración de un tramo es su longitud + 1 (el día en que
    recupera el pico). Un tramo final sin recuperación se cuenta censurado (sin
    el +1). Una curva sin drawdowns devuelve ``0``.
    """
    v = np.asarray(equity_curve, dtype=np.float64)
    if v.size < 2:
        return 0
    running_max = np.maximum.accumulate(v)
    underwater = v < running_max
    max_rec = 0
    run = 0
    for u in underwater:
        if u:
            run += 1
        else:
            if run > 0:
                max_rec = max(max_rec, run + 1)  # +1: día de recuperación del pico
            run = 0
    if run > 0:  # tramo final censurado: el test acaba bajo el agua
        max_rec = max(max_rec, run)
    return int(max_rec)


def compute_all_metrics(
    returns: np.ndarray,
    equity_curve: np.ndarray,
    turnover: np.ndarray,
) -> dict[str, float]:
    """Agrega la batería completa de métricas de un backtest en un dict.

    Parameters
    ----------
    returns:
        Serie de retornos diarios netos de la cartera.
    equity_curve:
        Curva de riqueza ``V_t`` (incluye ``V_0``); usada para MDD y recovery.
    turnover:
        Serie de turnover diario; usada para el turnover anualizado.
    """
    return {
        "annualized_return": annualized_return(returns),
        "annualized_vol": annualized_vol(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "max_drawdown": max_drawdown(equity_curve),
        "calmar_ratio": calmar_ratio(returns),
        "cvar_95": cvar_95(returns),
        "annualized_turnover": annualized_turnover(turnover),
        "win_rate": win_rate(returns),
        "recovery_time": float(recovery_time(equity_curve)),
    }
