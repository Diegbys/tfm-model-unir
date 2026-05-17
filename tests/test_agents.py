"""Tests de la infraestructura de agentes PPO — Fase 6.

Cubren la fijación de seeds global (F6-T7), el mezclador de datos real/sintético
(F6-T4) y la factory de entornos (F6-T3). Siguen las convenciones de
``tests/test_env.py``: docstrings en español y asserts con mensaje; los tests que
dependen de artefactos de fases previas hacen ``pytest.skip`` si faltan.
"""
from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.download import PROJECT_ROOT

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
SYNTHETIC_PATH = PROJECT_ROOT / "data" / "synthetic" / "run_42" / "synthetic_dataset.parquet"


# --- F6-T7: fijación de seeds global ----------------------------------------


def test_set_global_seed_reproducible() -> None:
    """F6-T7: dos llamadas con el mismo seed producen idénticas secuencias aleatorias."""
    from src.utils.seeding import set_global_seed

    set_global_seed(42)
    primera = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    set_global_seed(42)
    segunda = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert primera == segunda, f"seeds no reproducibles: {primera} != {segunda}"


def test_set_global_seed_distintos_difieren() -> None:
    """Seeds distintos producen secuencias distintas (no es un no-op)."""
    from src.utils.seeding import set_global_seed

    set_global_seed(0)
    a = float(np.random.rand())
    set_global_seed(1)
    b = float(np.random.rand())
    assert a != b, "dos seeds distintos dieron el mismo valor — set_global_seed no siembra"


# --- Métricas mínimas para el ValidationEvalCallback (F7-T2 extenderá) -------


def test_sharpe_ratio_caso_conocido() -> None:
    """Sharpe = mean/std * sqrt(periods) con rf=0 (compass §4.4)."""
    from src.eval.metrics import sharpe_ratio

    r = np.array([0.01, -0.005, 0.02, 0.0, 0.015])
    esperado = r.mean() / r.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(r) == pytest.approx(esperado)


def test_sharpe_ratio_varianza_cero_es_cero() -> None:
    """Retornos constantes (σ=0) ⇒ Sharpe 0.0 (caso degenerado, sin señal)."""
    from src.eval.metrics import sharpe_ratio

    assert sharpe_ratio(np.full(10, 0.01)) == 0.0


def test_max_drawdown_caida_conocida() -> None:
    """MDD de una curva 100→120→60→90: peor caída de 120 a 60 ⇒ -0.5."""
    from src.eval.metrics import max_drawdown

    equity = np.array([100.0, 120.0, 60.0, 90.0])
    assert max_drawdown(equity) == pytest.approx(-0.5)


def test_max_drawdown_curva_creciente_es_cero() -> None:
    """Una curva monótona creciente no tiene drawdown."""
    from src.eval.metrics import max_drawdown

    assert max_drawdown(np.array([100.0, 101.0, 102.0, 103.0])) == pytest.approx(0.0)


# --- Fixtures de datos (F6-T4 / F6-T3) --------------------------------------


@pytest.fixture(scope="module")
def features_df() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        pytest.skip(f"{FEATURES_PATH} no existe — ejecuta scripts/02_build_features.py")
    return pd.read_parquet(FEATURES_PATH)


@pytest.fixture(scope="module")
def train_idx(features_df: pd.DataFrame) -> pd.DatetimeIndex:
    from src.data.splits import chronological_split

    return chronological_split(features_df)[0]


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    if not SYNTHETIC_PATH.exists():
        pytest.skip(f"{SYNTHETIC_PATH} no existe — ejecuta la Fase 4 (run_42)")
    return pd.read_parquet(SYNTHETIC_PATH)


# --- F6-T4: mezclador de datos real/sintético -------------------------------


def test_mixed_dataset_proporciones(
    features_df: pd.DataFrame, train_idx: pd.DatetimeIndex, synthetic_df: pd.DataFrame
) -> None:
    """F6-T4: 10000 elecciones de rama respetan synthetic_ratio (test chi-cuadrado)."""
    from scipy.stats import chisquare

    from src.data.mixed_dataset import MixedDataset

    mixed = MixedDataset(features_df, train_idx, synthetic_df, synthetic_ratio=2 / 3, seed=0)
    rng = np.random.default_rng(0)
    branches = [mixed.choose_branch(rng) for _ in range(10000)]
    n_real = branches.count("real")
    n_synth = branches.count("synthetic")
    assert n_real + n_synth == 10000, "choose_branch devolvió una rama no reconocida"
    _, p = chisquare([n_real, n_synth], [10000 / 3, 20000 / 3])
    assert p > 0.01, f"proporciones se desvían de 1:2 (real={n_real}, synth={n_synth}, p={p:.4f})"


def test_mixed_dataset_agente_a_solo_real(
    features_df: pd.DataFrame, train_idx: pd.DatetimeIndex
) -> None:
    """Con synthetic_ratio=0.0 (Agente A) todos los episodios son reales."""
    from src.data.mixed_dataset import MixedDataset

    mixed = MixedDataset(features_df, train_idx, synthetic_df=None, synthetic_ratio=0.0, seed=0)
    rng = np.random.default_rng(0)
    assert all(mixed.choose_branch(rng) == "real" for _ in range(1000))


def test_mixed_dataset_sample_episode_shape(
    features_df: pd.DataFrame, train_idx: pd.DatetimeIndex, synthetic_df: pd.DataFrame
) -> None:
    """sample_episode devuelve trayectoria de 54 filas y body_idx de 23 fechas operables."""
    from src.data.mixed_dataset import MixedDataset

    mixed = MixedDataset(features_df, train_idx, synthetic_df, synthetic_ratio=2 / 3, seed=0)
    traj, body_idx = mixed.sample_episode()
    assert traj.shape == (54, features_df.shape[1]), f"shape inesperado {traj.shape}"
    assert isinstance(traj.index, pd.DatetimeIndex)
    assert traj.index.is_monotonic_increasing
    assert len(body_idx) == 23, f"body_idx debe tener 23 fechas, tiene {len(body_idx)}"
    assert body_idx.isin(traj.index).all(), "body_idx no es subconjunto del índice de la trayectoria"
    assert list(traj.columns) == list(features_df.columns)


def test_mixed_dataset_episodio_sintetico_usa_secuencia(
    features_df: pd.DataFrame, train_idx: pd.DatetimeIndex, synthetic_df: pd.DataFrame
) -> None:
    """Un episodio sintético = warmup real (30 filas) + cuerpo = una secuencia sintética (24)."""
    from src.data.mixed_dataset import MixedDataset

    mixed = MixedDataset(features_df, train_idx, synthetic_df, synthetic_ratio=1.0, seed=1)
    traj, _ = mixed.sample_episode()
    assert mixed.last_branch == "synthetic"
    cuerpo = traj.iloc[30:].reset_index(drop=True)
    seq = synthetic_df.xs(mixed.last_seq_id, level="seq_id").reset_index(drop=True)
    assert np.allclose(
        cuerpo[features_df.columns].to_numpy(), seq[features_df.columns].to_numpy()
    ), "el cuerpo del episodio sintético no coincide con la secuencia sintética muestreada"


# --- F6-T3: factory de entornos ---------------------------------------------


@pytest.fixture(scope="module")
def env_config():
    """Config del entorno cargada desde el YAML de Hydra (F5-T6)."""
    from omegaconf import OmegaConf

    return OmegaConf.load(PROJECT_ROOT / "configs" / "env" / "portfolio_default.yaml")


def test_make_train_env_es_vecenv(
    features_df: pd.DataFrame, train_idx: pd.DatetimeIndex, env_config
) -> None:
    """make_train_env devuelve un VecEnv de SB3 con los shapes de obs/acción correctos."""
    from stable_baselines3.common.vec_env import VecEnv

    from src.agents.env_factory import make_train_env
    from src.data.mixed_dataset import MixedDataset

    mixed = MixedDataset(features_df, train_idx, synthetic_ratio=0.0, seed=0)
    venv = make_train_env(mixed, env_config, seed=0)
    try:
        assert isinstance(venv, VecEnv)
        assert venv.observation_space.shape == (30 * 139 + 7,)
        assert venv.action_space.shape == (7,)
        venv.reset()  # arranca sin error
    finally:
        venv.close()


def test_make_train_env_vecnormalize_opcional(
    features_df: pd.DataFrame, train_idx: pd.DatetimeIndex, env_config
) -> None:
    """use_vecnormalize=True envuelve en VecNormalize; False deja DummyVecEnv."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from src.agents.env_factory import make_train_env
    from src.data.mixed_dataset import MixedDataset

    mixed = MixedDataset(features_df, train_idx, synthetic_ratio=0.0, seed=0)
    con = make_train_env(mixed, env_config, seed=0, use_vecnormalize=True)
    sin = make_train_env(mixed, env_config, seed=0, use_vecnormalize=False)
    try:
        assert isinstance(con, VecNormalize)
        assert isinstance(sin, DummyVecEnv)
    finally:
        con.close()
        sin.close()


def test_make_eval_env_recorrido_completo(features_df: pd.DataFrame, env_config) -> None:
    """make_eval_env devuelve un PortfolioEnv que recorre todo el split de val."""
    from src.agents.env_factory import make_eval_env
    from src.data.splits import chronological_split
    from src.envs.portfolio_env import PortfolioEnv

    _, val_idx, _ = chronological_split(features_df)
    env = make_eval_env(features_df, val_idx, env_config)
    assert isinstance(env, PortfolioEnv)
    assert env.n_tradeable_days == len(val_idx)


# --- F6-T6: evaluación periódica sobre validación ----------------------------


def test_evaluate_on_env_devuelve_metricas(
    features_df: pd.DataFrame, train_idx: pd.DatetimeIndex, env_config
) -> None:
    """evaluate_on_env recorre un episodio de val y devuelve val_sharpe/mdd/return finitos."""
    from stable_baselines3 import PPO

    from src.agents.callbacks import evaluate_on_env
    from src.agents.env_factory import make_eval_env, make_train_env
    from src.data.mixed_dataset import MixedDataset
    from src.data.splits import chronological_split

    _, val_idx, _ = chronological_split(features_df)
    mixed = MixedDataset(features_df, train_idx, synthetic_ratio=0.0, seed=0)
    train_env = make_train_env(mixed, env_config, seed=0, use_vecnormalize=False)
    model = PPO("MlpPolicy", train_env, n_steps=64, batch_size=32, seed=0)
    eval_env = make_eval_env(features_df, val_idx, env_config)
    try:
        metrics = evaluate_on_env(model, eval_env)
    finally:
        train_env.close()
    assert set(metrics) == {"val_sharpe", "val_mdd", "val_return"}
    assert all(np.isfinite(v) for v in metrics.values()), f"métrica no finita: {metrics}"
    assert metrics["val_mdd"] <= 0.0, "el drawdown debe ser no positivo"


# --- F6-T5: helper de configuración del trainer PPO --------------------------


def test_build_policy_kwargs_resuelve_activation() -> None:
    """build_policy_kwargs convierte activation_fn 'tanh' en la clase torch.nn.Tanh."""
    from omegaconf import OmegaConf

    from src.agents.ppo_trainer import build_policy_kwargs

    ppo_cfg = OmegaConf.create(
        {
            "policy_kwargs": {
                "net_arch": {"pi": [128, 128], "vf": [128, 128]},
                "activation_fn": "tanh",
                "ortho_init": True,
            }
        }
    )
    pk = build_policy_kwargs(ppo_cfg)
    assert pk["activation_fn"] is torch.nn.Tanh
    assert pk["net_arch"] == {"pi": [128, 128], "vf": [128, 128]}
    assert pk["ortho_init"] is True
