"""Callback de evaluación periódica sobre validación — F6-T6.

Durante el entrenamiento PPO, ``ValidationEvalCallback`` evalúa cada ``eval_freq``
steps la política actual sobre el split de **validación real** (2022, sin
sintéticos) y registra Sharpe, MDD y retorno en MLflow. El Sharpe en val —no el
``ep_rew_mean`` del rollout, que en el Agente B mezcla episodios sintéticos— es
el criterio de selección del mejor checkpoint (ADR §Fase 6, F6-T6).
"""
from __future__ import annotations

import logging
from pathlib import Path

import mlflow
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.eval.metrics import max_drawdown, sharpe_ratio

logger = logging.getLogger(__name__)


def evaluate_on_env(model, eval_env) -> dict[str, float]:
    """Recorre un episodio completo de ``eval_env`` con la política de ``model``.

    Parameters
    ----------
    model:
        Modelo SB3 (en entrenamiento o final); se usa ``predict`` determinista.
    eval_env:
        ``PortfolioEnv`` en modo recorrido-completo sobre un split real (val/test).

    Returns
    -------
    dict
        ``val_sharpe`` (Sharpe anualizado de los retornos diarios de la cartera),
        ``val_mdd`` (máximo drawdown, ≤ 0) y ``val_return`` (retorno total del
        episodio, ``V_T / V_0 - 1``).
    """
    obs, _ = eval_env.reset()
    equity = [1.0]  # V_0
    rewards: list[float] = []
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        rewards.append(reward)
        equity.append(info["V_t"])
    daily_returns = np.expm1(np.asarray(rewards, dtype=np.float64))  # exp(reward) - 1
    equity_arr = np.asarray(equity, dtype=np.float64)
    return {
        "val_sharpe": sharpe_ratio(daily_returns),
        "val_mdd": max_drawdown(equity_arr),
        "val_return": float(equity_arr[-1] / equity_arr[0] - 1.0),
    }


class ValidationEvalCallback(BaseCallback):
    """Evalúa sobre validación cada ``eval_freq`` steps y guarda el mejor modelo.

    "Mejor" se define por ``val_sharpe`` (F6-T6): el checkpoint con mayor Sharpe
    en val se persiste en ``best_model_path`` y las métricas se loguean en MLflow
    (si hay un run activo) y en el logger de SB3.
    """

    def __init__(
        self,
        eval_env,
        eval_freq: int,
        best_model_path,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.best_model_path = Path(best_model_path)
        self.best_sharpe = -np.inf
        self.evaluations: list[dict] = []

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            metrics = evaluate_on_env(self.model, self.eval_env)
            for key, value in metrics.items():
                self.logger.record(f"val/{key}", value)
                if mlflow.active_run() is not None:
                    mlflow.log_metric(key, value, step=self.num_timesteps)
            if metrics["val_sharpe"] > self.best_sharpe:
                self.best_sharpe = metrics["val_sharpe"]
                self.model.save(self.best_model_path)
                if self.verbose:
                    logger.info(
                        "nuevo mejor val_sharpe=%.4f @ %d steps → %s",
                        self.best_sharpe, self.num_timesteps, self.best_model_path,
                    )
            self.evaluations.append({"timesteps": self.num_timesteps, **metrics})
        return True
