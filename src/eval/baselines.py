"""Baselines pasivos para el backtest — F7-T3.

Tres estrategias de referencia que se comparan contra los agentes PPO para
contextualizar los resultados y reconocer el *survivorship bias* del universo
(compass §7.2): equiponderado de los 6 activos, buy-and-hold del S&P 500 y una
cartera 60/40 acciones/bonos.

Los tres se calculan con el **mismo motor de pesos fijos** y las **mismas
primitivas de coste** que el ``PortfolioEnv`` (drift de pesos → turnover → coste
10 bps): así los costes de los baselines son idénticos a los del agente
(compass §7.4, pitfall #4 — "olvidar costes en baselines"). Cada función
devuelve un DataFrame con el mismo esquema que ``run_backtest`` (F7-T1).

Proxy de bono del 60/40
-----------------------
El Treasury 10Y no es un activo cotizado del universo, así que su retorno diario
se **aproxima** desde el yield ``^TNX``: ``r_bono ≈ carry - D_mod · Δy``, con
``carry = yield/252`` y duración modificada ``D_mod ≈ 8`` (típica de un bono a
10 años). Es una aproximación deliberada — el 60/40 es un baseline de contexto,
no el objeto de estudio. Se documenta como limitación en el Cap. 6.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.download import EQUITY_TICKERS_YF, PROJECT_ROOT, _safe_filename
from src.envs.portfolio_env import PortfolioEnv
from src.envs.rewards import log_return_minus_costs
from src.envs.transaction_costs import compute_turnover, transaction_cost

logger = logging.getLogger(__name__)

DEFAULT_COST_PCT = 0.001  # 10 bps — idéntico al del agente (ADR §3.3)
BOND_MODIFIED_DURATION = 8.0  # duración modificada aprox. del Treasury 10Y
ALIGNED_PATH = PROJECT_ROOT / "data" / "processed" / "aligned.parquet"

_ADJ_COLS = [f"{_safe_filename(t)}_AdjClose" for t in EQUITY_TICKERS_YF]
N_ASSETS = len(EQUITY_TICKERS_YF)  # 6


def _equity_returns(features_df: pd.DataFrame, idx: pd.DatetimeIndex) -> np.ndarray:
    """Retornos simples diarios de los 6 activos para las fechas de ``idx``.

    El ``pct_change`` se calcula sobre el dataframe completo y luego se
    reindexan las fechas del split, de modo que el retorno del primer día de
    ``idx`` es causal (frente al día hábil anterior).
    """
    returns = features_df[_ADJ_COLS].pct_change(fill_method=None)
    return returns.reindex(idx).to_numpy(dtype=np.float64)  # (N, 6)


def _bond_proxy_returns(idx: pd.DatetimeIndex) -> np.ndarray:
    """Retorno diario aproximado del Treasury 10Y desde el yield ``^TNX``.

    ``r ≈ yield_{t-1}/252 - D_mod · Δyield``. Lee ``TNX_Close`` (yield en %) de
    ``aligned.parquet`` y lo pasa a decimal. Ver el docstring del módulo.
    """
    aligned = pd.read_parquet(ALIGNED_PATH)
    yield_dec = aligned["TNX_Close"] / 100.0  # ^TNX cotiza el yield en %
    delta_y = yield_dec.diff()
    carry = yield_dec.shift(1) / 252.0
    r_bond = carry - BOND_MODIFIED_DURATION * delta_y
    return r_bond.reindex(idx).to_numpy(dtype=np.float64)


def _run_fixed_weight_backtest(
    target_weights: np.ndarray,
    asset_returns: np.ndarray,
    dates: pd.DatetimeIndex,
    cost_pct: float,
    *,
    rebalance: bool,
) -> pd.DataFrame:
    """Motor de backtest de una estrategia de pesos fijos.

    Réplica de la mecánica de un paso del ``PortfolioEnv``: drift de los pesos
    previos con el retorno del día, rebalanceo (o no) al objetivo, coste por
    turnover y actualización de la riqueza ``V_t = V_{t-1}·exp(reward)``.

    Parameters
    ----------
    target_weights:
        Pesos objetivo ``(K,)`` — ``K`` activos (el último es cash o bono).
    asset_returns:
        Retornos diarios ``(N, K)``; la última columna es 0 (cash) o el retorno
        del bono (60/40).
    rebalance:
        ``True`` → cada día se vuelve al objetivo (equiponderado, 60/40).
        ``False`` → buy-and-hold: solo el día 1 se compra; después se deja
        derivar la cartera. El día 1 siempre rebalancea (entrada desde cash).
    """
    target = np.asarray(target_weights, dtype=np.float64)
    n_days = asset_returns.shape[0]
    w_prev = np.zeros(target.size, dtype=np.float64)
    w_prev[-1] = 1.0  # cartera inicial 100 % en el último slot (cash)

    rows: list[dict] = []
    wealth = prev_wealth = 1.0
    for t in range(n_days):
        r_t = asset_returns[t]
        w_drifted = PortfolioEnv._drift_weights(w_prev, r_t)
        w_t = target.copy() if (rebalance or t == 0) else w_drifted
        turnover = compute_turnover(w_t, w_drifted)
        cost = transaction_cost(turnover, cost_pct)
        gross_return = float(np.dot(w_t, r_t))
        reward = log_return_minus_costs(gross_return, cost)
        wealth *= float(np.exp(reward))
        rows.append(
            {
                "date": dates[t],
                "action_weights": w_t.copy(),
                "portfolio_return": wealth / prev_wealth - 1.0,
                "cost": float(cost),
                "turnover": float(turnover),
                "wealth": wealth,
            }
        )
        prev_wealth = wealth
        w_prev = w_t

    df = pd.DataFrame(rows).set_index("date")
    running_max = df["wealth"].cummax()
    df["drawdown"] = (df["wealth"] - running_max) / running_max
    return df


def equal_weight_6assets(
    features_df: pd.DataFrame,
    idx: pd.DatetimeIndex,
    cost_pct: float = DEFAULT_COST_PCT,
) -> pd.DataFrame:
    """Baseline equiponderado: rebalanceo diario a 1/6 en cada uno de los 6 activos."""
    equity = _equity_returns(features_df, idx)
    asset_returns = np.hstack([equity, np.zeros((equity.shape[0], 1))])  # + cash
    target = np.array([1.0 / N_ASSETS] * N_ASSETS + [0.0])
    return _run_fixed_weight_backtest(
        target, asset_returns, idx, cost_pct, rebalance=True
    )


def buy_and_hold_spy(
    features_df: pd.DataFrame,
    idx: pd.DatetimeIndex,
    cost_pct: float = DEFAULT_COST_PCT,
) -> pd.DataFrame:
    """Baseline buy-and-hold del S&P 500: 100 % en el índice, sin rebalanceo."""
    equity = _equity_returns(features_df, idx)
    asset_returns = np.hstack([equity, np.zeros((equity.shape[0], 1))])  # + cash
    target = np.zeros(N_ASSETS + 1)
    target[0] = 1.0  # GSPC = S&P 500 es el primer activo del universo
    return _run_fixed_weight_backtest(
        target, asset_returns, idx, cost_pct, rebalance=False
    )


def portfolio_60_40(
    features_df: pd.DataFrame,
    idx: pd.DatetimeIndex,
    cost_pct: float = DEFAULT_COST_PCT,
) -> pd.DataFrame:
    """Baseline 60/40: 60 % equiponderado de los 6 activos + 40 % proxy de bono 10Y."""
    equity = _equity_returns(features_df, idx)
    bond = _bond_proxy_returns(idx).reshape(-1, 1)
    asset_returns = np.hstack([equity, bond])  # última columna = bono
    target = np.array([0.6 / N_ASSETS] * N_ASSETS + [0.4])
    return _run_fixed_weight_backtest(
        target, asset_returns, idx, cost_pct, rebalance=True
    )


BASELINES = {
    "equal_weight": equal_weight_6assets,
    "buy_and_hold_spy": buy_and_hold_spy,
    "portfolio_60_40": portfolio_60_40,
}
