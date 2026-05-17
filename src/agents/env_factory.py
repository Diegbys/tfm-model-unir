"""Factory de entornos para el entrenamiento PPO — F6-T3.

Dos constructores, uno por modo del :class:`~src.envs.portfolio_env.PortfolioEnv`:

- ``make_train_env``: entorno **episódico** (envuelto para SB3) que el agente usa
  para aprender. ``DummyVecEnv`` de 1 entorno —suficiente con datos diarios— y,
  opcionalmente, ``VecNormalize(norm_reward=True, norm_obs=False)``: los
  log-retornos diarios son ~1e-2 y normalizar la recompensa estabiliza PPO; las
  observaciones NO se normalizan porque ya vienen escaladas (RobustScaler).
- ``make_eval_env``: entorno **recorrido-completo** crudo sobre un split real
  (val/test). El ``ValidationEvalCallback`` lo recorre con ``model.predict``.

Decisión del plan F6: NO se aplica ``VecFrameStack``. El ``PortfolioEnv`` ya
incorpora una ventana de 30 días en cada observación (compass §3.6); apilar
encima sería ×10 redundante.
"""
from __future__ import annotations

import logging

import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize

from src.data.mixed_dataset import MixedDataset
from src.envs.portfolio_env import PortfolioEnv

logger = logging.getLogger(__name__)


def make_train_env(
    mixed_dataset: MixedDataset,
    env_config,
    seed: int,
    *,
    use_vecnormalize: bool = True,
    gamma: float = 0.99,
) -> VecEnv:
    """Construye el entorno vectorizado de entrenamiento PPO.

    Parameters
    ----------
    mixed_dataset:
        Muestreador de episodios real/sintético; su ``sample_episode`` alimenta
        el modo episódico del entorno.
    env_config:
        Config del entorno (``configs/env/portfolio_default.yaml``).
    seed:
        Semilla del VecEnv (acción-space sampling y primer reset deterministas).
    use_vecnormalize:
        Si ``True``, envuelve en ``VecNormalize(norm_reward=True, norm_obs=False)``.
    gamma:
        Factor de descuento; ``VecNormalize`` lo usa para normalizar la recompensa
        por la desviación del retorno descontado (debe coincidir con el de PPO).
    """
    env = PortfolioEnv(config=env_config, episode_sampler=mixed_dataset.sample_episode)
    venv: VecEnv = DummyVecEnv([lambda: env])
    if use_vecnormalize:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=gamma)
    venv.seed(seed)
    logger.info(
        "make_train_env: DummyVecEnv(1)%s, seed=%d",
        " + VecNormalize(norm_reward)" if use_vecnormalize else "",
        seed,
    )
    return venv


def make_eval_env(
    features_df: pd.DataFrame,
    eval_idx: pd.DatetimeIndex,
    env_config,
) -> PortfolioEnv:
    """Construye el entorno de evaluación: PortfolioEnv crudo en recorrido-completo.

    Recorre el split real ``eval_idx`` (val o test) una vez en orden cronológico.
    No se vectoriza ni se normaliza: el ``ValidationEvalCallback`` lo recorre con
    ``model.predict`` y calcula las métricas sobre las recompensas crudas. Es
    idéntico para el Agente A y el B (solo cambian ``features_df`` / ``eval_idx``).
    """
    return PortfolioEnv(features_df, eval_idx, env_config)
