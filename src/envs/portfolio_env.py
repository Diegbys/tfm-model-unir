"""Entorno de trading de cartera con la API Gymnasium — F5-T1/T2/T3.

``PortfolioEnv`` convierte la gestión de una cartera de 6 activos USA + cash en
un MDP que PPO (Fase 6) puede optimizar. Es **idéntico para el Agente A y el B**;
lo único que cambia es el ``dataset`` que consume.

Convención temporal (pitfall #1 del ADR §Fase 5)
-------------------------------------------------
El agente decide los pesos del día ``t`` observando **solo** información hasta el
cierre de ``t-1``; el retorno ``r_t`` se realiza después. No hay look-ahead: la
observación se construye con la ventana de features que termina en ``t-1``.

Observación, acción y recompensa
--------------------------------
- **Observación**: vector 1D ``(window * n_features + n_assets+1,)`` = la ventana
  de 30 días de las 139 features escaladas (RobustScaler) aplanada, concatenada
  con los pesos actuales de la cartera (6 activos + cash).
- **Acción**: vector continuo en ``[-1, 1]^(n_assets+1)``; ``softmax`` lo proyecta
  al simplex de pesos (el último componente es cash).
- **Recompensa**: ``log(1 + w_t·r_t - cost)`` = log-retorno realizado neto de costes.

Mecánica de un paso (F5-T3/T4/T5)
---------------------------------
1. ``w_t = softmax(action)`` — proyección al simplex (incluye cash).
2. ``w_drifted = drift(w_{t-1}, r_t)`` — los pesos previos derivan con el retorno.
3. ``turnover = 0.5 * sum(|w_t - w_drifted|)`` y ``cost = turnover * pct``.
4. ``reward_t = log(1 + w_t·r_t - cost)`` y ``V_t = V_{t-1} * exp(reward_t)``.

El coste entra dentro del log (ver :mod:`src.envs.rewards`) para que se cumpla
exacto el invariante de consistencia ``V_T / V_0 == exp(sum(rewards))`` (F5-T9).

Episodio: recorre el split una vez en orden cronológico; ``terminated=True`` el
último día. ``truncated`` solo si se configura ``max_episode_steps``. La cartera
inicial es 100% cash (``V_0 = 1.0``).
"""
from __future__ import annotations

import logging

import gymnasium as gym
import numpy as np
import pandas as pd

from src.data.download import EQUITY_TICKERS_YF, _safe_filename
from src.data.scalers import load_ppo_scaler, transform_with_scaler
from src.data.state_builder import build_state
from src.envs.rewards import log_return_minus_costs
from src.envs.transaction_costs import compute_turnover, transaction_cost

logger = logging.getLogger(__name__)


class PortfolioEnv(gym.Env):
    """Entorno Gymnasium de gestión de cartera (ver docstring del módulo)."""

    metadata = {"render_modes": []}

    def __init__(self, dataset: pd.DataFrame, splits_idx: pd.DatetimeIndex, config) -> None:
        """Construye el entorno.

        Parameters
        ----------
        dataset:
            DataFrame con el esquema de ``features.parquet`` (139 features +
            6 ``{ticker}_AdjClose``), índice ``DatetimeIndex`` creciente. El env
            escala las features internamente con el ``RobustScaler`` ajustado en
            train y deriva los retornos de los activos desde las columnas
            ``AdjClose``. Agente A y B solo intercambian este DataFrame.
        splits_idx:
            Fechas operables del split (train / val / test).
        config:
            Mapping (dict o ``DictConfig``) con las claves de
            ``configs/env/portfolio_default.yaml``.
        """
        super().__init__()

        # --- config (F5-T6) ---
        self.window = int(config["window"])
        self.n_assets = int(config["n_assets"])
        self.n_weights = self.n_assets + 1  # + cash
        self.transaction_cost_pct = float(config["transaction_cost_pct"])
        self.slippage_pct = float(config.get("slippage_pct", 0.0))
        self.reward_type = str(config["reward_type"])
        max_steps = config.get("max_episode_steps", None)
        self._max_episode_steps = None if max_steps is None else int(max_steps)

        # --- validación del dataset ---
        if not isinstance(dataset.index, pd.DatetimeIndex):
            raise TypeError("PortfolioEnv: dataset.index debe ser DatetimeIndex")
        if not dataset.index.is_monotonic_increasing:
            raise ValueError("PortfolioEnv: dataset.index no es creciente")
        self._dates = dataset.index

        # --- features escaladas para la observación (sin leakage: scaler de train) ---
        scaler, ppo_cols = load_ppo_scaler()
        self._scaled = transform_with_scaler(dataset, scaler, ppo_cols)
        self._n_features = len(ppo_cols)

        # --- retornos simples de los 6 activos para la recompensa ---
        prefixes = [_safe_filename(t) for t in EQUITY_TICKERS_YF]
        adj_cols = [f"{p}_AdjClose" for p in prefixes]
        missing = [c for c in adj_cols if c not in dataset.columns]
        if missing:
            raise ValueError(f"PortfolioEnv: faltan columnas AdjClose {missing}")
        returns = dataset[adj_cols].pct_change(fill_method=None)
        self._returns = returns.to_numpy(dtype=np.float64)  # (N, n_assets)

        # --- días operables: posiciones del split con ventana completa previa ---
        pos_by_date = {d: i for i, d in enumerate(self._dates)}
        tradeable = [
            pos_by_date[d]
            for d in splits_idx
            if d in pos_by_date and pos_by_date[d] >= self.window
        ]
        if not tradeable:
            raise ValueError(
                "PortfolioEnv: ningún día operable — el split no tiene "
                f"{self.window} días de historia previa en el dataset"
            )
        self._tradeable = tradeable
        self.n_tradeable_days = len(tradeable)

        # --- espacios Gymnasium (F5-T1) ---
        obs_dim = self.window * self._n_features + self.n_weights
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_weights,), dtype=np.float32
        )

        # --- estado del episodio (se inicializa en reset) ---
        self.render_mode = None
        self._ptr = 0
        self._V = 1.0
        self._cum_cost = 0.0
        self._w = np.zeros(self.n_weights, dtype=np.float64)

    # --- proyecciones puras (F5-T2 / F5-T3) ----------------------------------

    @staticmethod
    def _action_to_weights(action: np.ndarray) -> np.ndarray:
        """F5-T2: proyecta la acción al simplex de pesos vía softmax.

        ``weights = softmax(action)`` garantiza pesos en ``[0,1]`` que suman 1.
        Se resta el máximo antes de exponenciar para estabilidad numérica. El
        último componente es el peso de "cash".
        """
        a = np.asarray(action, dtype=np.float64)
        a = a - np.max(a)
        exp = np.exp(a)
        return exp / exp.sum()

    @staticmethod
    def _drift_weights(w_prev: np.ndarray, r: np.ndarray) -> np.ndarray:
        """F5-T3: deriva los pesos previos con el retorno del día.

        ``w_drifted = (w_prev ⊙ (1 + r)) / (w_prev · (1 + r))``. Con retornos
        cero el drift es la identidad; con retornos no nulos el resultado sigue
        sumando 1.
        """
        w_prev = np.asarray(w_prev, dtype=np.float64)
        r = np.asarray(r, dtype=np.float64)
        num = w_prev * (1.0 + r)
        total = num.sum()
        if total <= 0.0:
            raise ValueError(
                f"_drift_weights: suma no positiva tras el drift ({total})"
            )
        return num / total

    # --- API Gymnasium -------------------------------------------------------

    def _build_obs(self, pos: int) -> np.ndarray:
        """Observación de la decisión del día ``pos``: ventana hasta ``pos-1`` + pesos."""
        window_arr = build_state(self._scaled, self._dates[pos - 1], self.window)
        return np.concatenate([window_arr.reshape(-1), self._w]).astype(np.float32)

    def _info(self, pos: int, turnover: float, cost: float) -> dict:
        return {
            "date": self._dates[pos],
            "V_t": self._V,
            "weights": self._w.copy(),
            "turnover": turnover,
            "cost": cost,
            "cum_cost": self._cum_cost,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reinicia el episodio al primer día operable; cartera inicial 100% cash."""
        super().reset(seed=seed)
        self._ptr = 0
        self._V = 1.0
        self._cum_cost = 0.0
        self._w = np.zeros(self.n_weights, dtype=np.float64)
        self._w[-1] = 1.0  # 100% cash
        obs = self._build_obs(self._tradeable[0])
        return obs, self._info(self._tradeable[0], turnover=0.0, cost=0.0)

    def step(self, action: np.ndarray):
        """Avanza un día: rebalancea, aplica costes y devuelve la recompensa."""
        pos = self._tradeable[self._ptr]

        w_t = self._action_to_weights(action)
        r_t = np.append(self._returns[pos], 0.0)  # retorno de cash = 0
        w_drifted = self._drift_weights(self._w, r_t)

        turnover = compute_turnover(w_t, w_drifted)
        cost = transaction_cost(turnover, self.transaction_cost_pct, self.slippage_pct)
        portfolio_return = float(np.dot(w_t, r_t))
        reward = log_return_minus_costs(portfolio_return, cost)

        self._V *= float(np.exp(reward))
        self._cum_cost += cost
        self._w = w_t

        self._ptr += 1
        terminated = bool(self._ptr >= self.n_tradeable_days)
        truncated = bool(
            self._max_episode_steps is not None
            and self._ptr >= self._max_episode_steps
        )

        if terminated or truncated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._build_obs(self._tradeable[self._ptr])

        return obs, float(reward), terminated, truncated, self._info(pos, turnover, cost)

    def render(self):  # noqa: D102 — entorno sin render visual
        return None

    def close(self) -> None:  # noqa: D102
        pass
