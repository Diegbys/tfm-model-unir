"""Backtest causal de un agente PPO sobre un split real — F7-T1.

``run_backtest`` recorre un :class:`~src.envs.portfolio_env.PortfolioEnv` en modo
recorrido-completo (val o test) con la política determinista de un modelo SB3 y
devuelve la serie diaria del backtest: pesos, retorno, coste, riqueza y drawdown.

El backtest **no usa ``VecNormalize``**: los agentes se entrenaron con
``norm_obs=False`` (las observaciones nunca se normalizaron), así que
``model.predict`` sobre la observación cruda del entorno es correcto. Es el mismo
patrón que ``ValidationEvalCallback`` (F6-T6), el mecanismo ya validado de
selección del mejor checkpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

logger = logging.getLogger(__name__)

_CHECKPOINTS = {"best": "best_model.zip", "final": "model.zip"}


def load_run_model(run_dir: str | Path, checkpoint: str = "best") -> PPO:
    """Carga el modelo PPO de una corrida de la Fase 6.

    Parameters
    ----------
    run_dir:
        Carpeta de la corrida, p. ej. ``outputs/ppo_runs/agent_a/seed42``.
    checkpoint:
        ``"best"`` → ``best_model.zip`` (mejor Sharpe en val, criterio F6-T6,
        primario para la comparativa); ``"final"`` → ``model.zip`` (modelo del
        final del entrenamiento, tabla secundaria — Hallazgo 3 de F6).
    """
    if checkpoint not in _CHECKPOINTS:
        raise ValueError(
            f"checkpoint debe ser uno de {list(_CHECKPOINTS)}, no {checkpoint!r}"
        )
    model_path = Path(run_dir) / _CHECKPOINTS[checkpoint]
    if not model_path.exists():
        raise FileNotFoundError(f"no existe el modelo {model_path}")
    return PPO.load(model_path)


def run_backtest(model, env, deterministic: bool = True) -> pd.DataFrame:
    """Recorre ``env`` con la política de ``model`` y devuelve la serie del backtest.

    Parameters
    ----------
    model:
        Modelo SB3 entrenado; se usa ``predict(deterministic=deterministic)``.
    env:
        ``PortfolioEnv`` en modo recorrido-completo sobre un split real (val/test).
    deterministic:
        Si ``True`` (recomendado en evaluación, F7) usa la media de la política
        gaussiana en vez de muestrear.

    Returns
    -------
    pd.DataFrame
        Indexado por ``date``, una fila por día operable. Columnas:
        ``action_weights`` (np.ndarray de pesos aplicados), ``portfolio_return``
        (retorno diario neto ``V_t/V_{t-1}-1``), ``cost``, ``turnover``,
        ``wealth`` (``V_t``) y ``drawdown`` (≤ 0).
    """
    obs, _ = env.reset()
    rows: list[dict] = []
    prev_wealth = 1.0  # V_0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _reward, terminated, truncated, info = env.step(action)
        wealth = float(info["V_t"])
        rows.append(
            {
                "date": info["date"],
                "action_weights": info["weights"],
                "portfolio_return": wealth / prev_wealth - 1.0,
                "cost": float(info["cost"]),
                "turnover": float(info["turnover"]),
                "wealth": wealth,
            }
        )
        prev_wealth = wealth

    df = pd.DataFrame(rows).set_index("date")
    running_max = df["wealth"].cummax()
    df["drawdown"] = (df["wealth"] - running_max) / running_max
    logger.info(
        "run_backtest: %d días, wealth final=%.4f, MDD=%.4f",
        len(df),
        df["wealth"].iloc[-1],
        df["drawdown"].min(),
    )
    return df


def equity_curve(backtest_df: pd.DataFrame) -> np.ndarray:
    """Curva de riqueza con ``V_0=1`` antepuesto, para MDD / recovery time."""
    return np.concatenate([[1.0], backtest_df["wealth"].to_numpy(dtype=np.float64)])
