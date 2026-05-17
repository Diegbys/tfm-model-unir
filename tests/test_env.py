"""Tests del entorno de trading PortfolioEnv — F5-T7…F5-T10.

Cubren la API Gymnasium, la proyección al simplex vía softmax, la consistencia
de la recompensa y el cálculo de costes de transacción. Siguen las convenciones
de ``tests/test_data.py``: fixture de módulo con ``pytest.skip`` si faltan
artefactos, asserts con mensaje y docstrings en español.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env
from omegaconf import OmegaConf

from src.data.download import PROJECT_ROOT
from src.data.splits import chronological_split
from src.envs.portfolio_env import PortfolioEnv
from src.envs.rewards import log_return_minus_costs
from src.envs.transaction_costs import compute_turnover, transaction_cost

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
ENV_CONFIG_PATH = PROJECT_ROOT / "configs" / "env" / "portfolio_default.yaml"


def test_compute_turnover_sin_cambio_es_cero() -> None:
    """Si los pesos nuevos coinciden con los drifted, el turnover es 0."""
    w = np.array([0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2], dtype=np.float64)
    assert compute_turnover(w, w) == pytest.approx(0.0)


def test_compute_turnover_rotacion_total_es_uno() -> None:
    """Rotar toda la cartera de un activo a otro da turnover 1 (bilateral/2)."""
    w_a = np.array([1.0, 0, 0, 0, 0, 0, 0])
    w_b = np.array([0, 1.0, 0, 0, 0, 0, 0])
    assert compute_turnover(w_a, w_b) == pytest.approx(1.0)


def test_transaction_costs() -> None:
    """F5-T10: turnover 0 ⇒ coste 0; turnover 1 ⇒ coste 0.001 (10 bps default)."""
    w = np.array([0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2])
    assert transaction_cost(compute_turnover(w, w), 0.001) == pytest.approx(0.0)

    w_a = np.array([1.0, 0, 0, 0, 0, 0, 0])
    w_b = np.array([0, 1.0, 0, 0, 0, 0, 0])
    turnover = compute_turnover(w_a, w_b)
    assert turnover == pytest.approx(1.0)
    assert transaction_cost(turnover, 0.001) == pytest.approx(0.001)


def test_transaction_cost_no_negativo_y_suma_slippage() -> None:
    """El coste nunca es negativo y el slippage se suma a la comisión."""
    assert transaction_cost(0.5, 0.001) == pytest.approx(0.0005)
    assert transaction_cost(0.5, 0.001, slippage_pct=0.001) == pytest.approx(0.001)
    assert transaction_cost(0.0, 0.001) >= 0.0


# --- F5-T5: función de recompensa --------------------------------------------


def test_reward_sin_retorno_ni_coste_es_cero() -> None:
    """Sin retorno ni coste, la recompensa (log-retorno) es 0."""
    assert log_return_minus_costs(0.0, 0.0) == pytest.approx(0.0)


def test_reward_es_log_del_factor_de_riqueza() -> None:
    """reward = log(1 + portfolio_return - cost)."""
    assert log_return_minus_costs(0.10, 0.0) == pytest.approx(math.log(1.10))
    assert log_return_minus_costs(0.0, 0.01) == pytest.approx(math.log(0.99))
    assert log_return_minus_costs(0.05, 0.001) == pytest.approx(math.log(1.049))


def test_reward_clip_en_factor_no_positivo() -> None:
    """Factor de riqueza <= 0 (ruina) se clipa: la recompensa es finita."""
    reward = log_return_minus_costs(-2.0, 0.0)
    assert math.isfinite(reward), "la recompensa debe ser finita incluso en ruina"
    assert reward < 0.0


# --- F5-T2: proyección al simplex vía softmax --------------------------------


def test_action_to_weights_zeros_es_uniforme() -> None:
    """softmax(0) reparte el peso uniformemente entre los 7 componentes."""
    w = PortfolioEnv._action_to_weights(np.zeros(7, dtype=np.float32))
    assert w == pytest.approx(np.full(7, 1 / 7))


def test_action_space() -> None:
    """F5-T8: 100 acciones aleatorias en [-1,1] -> pesos en [0,1] que suman 1."""
    rng = np.random.default_rng(0)
    for _ in range(100):
        action = rng.uniform(-1.0, 1.0, size=7).astype(np.float32)
        w = PortfolioEnv._action_to_weights(action)
        assert w.shape == (7,)
        assert np.all(w >= 0.0), f"peso negativo: {w}"
        assert np.all(w <= 1.0), f"peso >1: {w}"
        assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_action_to_weights_estable_con_valores_extremos() -> None:
    """El softmax no desborda con acciones grandes (resta del máximo)."""
    w = PortfolioEnv._action_to_weights(
        np.array([100, -100, 0, 0, 0, 0, 0], dtype=np.float32)
    )
    assert w.sum() == pytest.approx(1.0)
    assert np.all(np.isfinite(w))
    assert w[0] > w[1]


# --- F5-T3: drift de pesos ---------------------------------------------------


def test_drift_weights_retorno_cero_es_identidad() -> None:
    """F5-T3: con retornos cero, el drift deja los pesos intactos."""
    w = np.array([0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2])
    drifted = PortfolioEnv._drift_weights(w, np.zeros(7))
    assert drifted == pytest.approx(w)


def test_drift_weights_suma_uno() -> None:
    """F5-T3: tras el drift con retornos no nulos, los pesos siguen sumando 1."""
    w = np.array([0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2])
    r = np.array([0.05, -0.02, 0.01, 0.03, -0.01, 0.0, 0.0])
    drifted = PortfolioEnv._drift_weights(w, r)
    assert drifted.sum() == pytest.approx(1.0)


def test_drift_weights_caso_conocido() -> None:
    """El activo que sube gana peso relativo tras el drift."""
    w = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    r = np.array([0.10, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    drifted = PortfolioEnv._drift_weights(w, r)
    # (0.5*1.1, 0.5*1.0) / (0.55 + 0.50) = (0.55, 0.50) / 1.05
    assert drifted[0] == pytest.approx(0.55 / 1.05)
    assert drifted[1] == pytest.approx(0.50 / 1.05)


# --- Fixtures y tests del entorno completo (F5-T1/T6/T7/T9) ------------------


@pytest.fixture(scope="module")
def env_config():
    """Config del entorno cargada desde el YAML de Hydra (F5-T6)."""
    return OmegaConf.load(ENV_CONFIG_PATH)


@pytest.fixture(scope="module")
def features_df() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        pytest.skip(f"{FEATURES_PATH} no existe — ejecuta scripts/02_build_features.py")
    return pd.read_parquet(FEATURES_PATH)


@pytest.fixture
def train_env(features_df: pd.DataFrame, env_config) -> PortfolioEnv:
    """PortfolioEnv instanciado sobre el split de train de datos reales."""
    train_idx, _, _ = chronological_split(features_df)
    return PortfolioEnv(features_df, train_idx, env_config)


def test_config_valores_respetados(train_env: PortfolioEnv, env_config) -> None:
    """F5-T6: al instanciar el entorno los valores de la config se respetan."""
    assert train_env.window == env_config.window == 30
    assert train_env.n_assets == env_config.n_assets == 6


def test_espacios_shape_correcto(train_env: PortfolioEnv) -> None:
    """F5-T1: observation_space = (window*139 + 7,); action_space = (7,)."""
    assert train_env.observation_space.shape == (30 * 139 + 7,)
    assert train_env.action_space.shape == (7,)
    assert train_env.action_space.low.min() == pytest.approx(-1.0)
    assert train_env.action_space.high.max() == pytest.approx(1.0)


def test_reset_devuelve_obs_e_info(train_env: PortfolioEnv) -> None:
    """F5-T1: reset() devuelve (obs, info) con shape y dtype correctos."""
    obs, info = train_env.reset(seed=0)
    assert obs.shape == train_env.observation_space.shape
    assert obs.dtype == np.float32
    assert train_env.observation_space.contains(obs)
    assert isinstance(info, dict)


def test_step_devuelve_5_tupla(train_env: PortfolioEnv) -> None:
    """F5-T1: step(action) devuelve (obs, reward, terminated, truncated, info)."""
    train_env.reset(seed=0)
    action = train_env.action_space.sample()
    obs, reward, terminated, truncated, info = train_env.step(action)
    assert obs.shape == train_env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_gymnasium_compliance(train_env: PortfolioEnv) -> None:
    """F5-T7: el check_env oficial de Gymnasium no detecta errores de API."""
    check_env(train_env)


def test_reward_consistency(train_env: PortfolioEnv) -> None:
    """F5-T9: episodio completo con acción fija ⇒ V_T/V_0 ≈ exp(Σ rewards)."""
    train_env.reset(seed=0)
    accion_fija = np.zeros(train_env.action_space.shape, dtype=np.float32)
    rewards: list[float] = []
    info: dict = {"V_t": 1.0}
    terminated = truncated = False
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = train_env.step(accion_fija)
        rewards.append(reward)
    assert terminated, "el episodio debe terminar al final del split"
    assert info["V_t"] == pytest.approx(math.exp(sum(rewards)), abs=1e-3)


def test_episodio_recorre_todo_el_split(train_env: PortfolioEnv) -> None:
    """El número de pasos del episodio coincide con los días operables del split."""
    train_env.reset(seed=0)
    accion = np.zeros(train_env.action_space.shape, dtype=np.float32)
    pasos = 0
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = train_env.step(accion)
        pasos += 1
    assert pasos == train_env.n_tradeable_days


# --- F6: modo episódico (sampler de trayectorias) ----------------------------


@pytest.fixture
def episodic_env(features_df: pd.DataFrame, env_config) -> PortfolioEnv:
    """PortfolioEnv en modo episódico, alimentado por un MixedDataset solo-real."""
    from src.data.mixed_dataset import MixedDataset

    train_idx, _, _ = chronological_split(features_df)
    mixed = MixedDataset(features_df, train_idx, synthetic_ratio=0.0, seed=0)
    return PortfolioEnv(config=env_config, episode_sampler=mixed.sample_episode)


def test_modo_episodico_23_dias_operables(episodic_env: PortfolioEnv) -> None:
    """Cada episodio episódico tiene 23 días operables (cuerpo de 24 menos el baseline)."""
    episodic_env.reset(seed=0)
    assert episodic_env.n_tradeable_days == 23


def test_modo_episodico_reset_sin_seed_remuestrea(episodic_env: PortfolioEnv) -> None:
    """Dos reset() sin seed cargan trayectorias distintas (se re-muestrea el episodio)."""
    obs1, _ = episodic_env.reset()
    obs2, _ = episodic_env.reset()
    assert not np.array_equal(obs1, obs2), "dos reset dieron la misma obs — no se re-muestrea"


def test_modo_episodico_reset_con_seed_reproducible(episodic_env: PortfolioEnv) -> None:
    """reset(seed=N) es reproducible: dos llamadas con el mismo seed dan la misma obs."""
    obs1, _ = episodic_env.reset(seed=7)
    obs2, _ = episodic_env.reset(seed=7)
    assert np.array_equal(obs1, obs2)


def test_modo_episodico_obs_shape_estable(episodic_env: PortfolioEnv) -> None:
    """La observación conserva el shape del observation_space tras re-muestrear."""
    for _ in range(3):
        obs, _ = episodic_env.reset()
        assert obs.shape == episodic_env.observation_space.shape
        assert episodic_env.observation_space.contains(obs)


def test_modo_episodico_gymnasium_compliance(episodic_env: PortfolioEnv) -> None:
    """F5-T7 extendido: check_env de Gymnasium pasa también en modo episódico."""
    check_env(episodic_env)


def test_modo_episodico_episodio_termina_en_23_pasos(episodic_env: PortfolioEnv) -> None:
    """Un episodio episódico recorre exactamente sus 23 días operables."""
    episodic_env.reset(seed=0)
    accion = np.zeros(episodic_env.action_space.shape, dtype=np.float32)
    pasos = 0
    terminated = truncated = False
    while not (terminated or truncated):
        _, _, terminated, truncated, _ = episodic_env.step(accion)
        pasos += 1
    assert pasos == 23
