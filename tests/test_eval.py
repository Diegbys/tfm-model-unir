"""Tests de la evaluación / backtesting — Fase 7.

Cubren ``run_backtest`` (F7-T1), los baselines pasivos (F7-T3) y el runner de
agregación (F7-T4/T5). Siguen las convenciones de ``tests/test_env.py``: fixture
de módulo con ``pytest.skip`` si faltan artefactos, asserts con mensaje y
docstrings en español.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from src.data.download import PROJECT_ROOT
from src.data.splits import load_splits

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
ENV_CONFIG_PATH = PROJECT_ROOT / "configs" / "env" / "portfolio_default.yaml"
RUN_DIR = PROJECT_ROOT / "outputs" / "ppo_runs" / "agent_a" / "seed42"


@pytest.fixture(scope="module")
def artefactos():
    """Carga features, config de entorno, split de test y un modelo entrenado."""
    if not FEATURES_PATH.exists():
        pytest.skip(f"falta {FEATURES_PATH}")
    if not (RUN_DIR / "best_model.zip").exists():
        pytest.skip(f"falta el modelo entrenado en {RUN_DIR}")
    features_df = pd.read_parquet(FEATURES_PATH)
    env_config = OmegaConf.load(ENV_CONFIG_PATH)
    _, _, test_idx = load_splits()
    return features_df, env_config, test_idx


# --- F7-T1: run_backtest -----------------------------------------------------

def test_run_backtest_dataframe_bien_formado(artefactos) -> None:
    """``run_backtest`` devuelve un DataFrame sin NaN con las columnas del ADR."""
    from src.agents.env_factory import make_eval_env
    from src.eval.backtest import load_run_model, run_backtest

    features_df, env_config, test_idx = artefactos
    model = load_run_model(RUN_DIR, checkpoint="best")
    env = make_eval_env(features_df, test_idx, env_config)

    df = run_backtest(model, env)

    columnas = {"action_weights", "portfolio_return", "cost", "turnover",
                "wealth", "drawdown"}
    assert columnas.issubset(df.columns), f"faltan columnas: {columnas - set(df.columns)}"
    assert df.index.name == "date"
    assert len(df) == env.n_tradeable_days, "una fila por día operable del split"
    numericas = ["portfolio_return", "cost", "turnover", "wealth", "drawdown"]
    assert not df[numericas].isna().any().any(), "el backtest no debe tener NaN"
    assert (df["drawdown"] <= 1e-9).all(), "el drawdown es ≤ 0"
    assert (df["cost"] >= 0.0).all(), "los costes son ≥ 0"


def test_run_backtest_determinista(artefactos) -> None:
    """Con ``deterministic=True`` dos backtests del mismo modelo son idénticos."""
    from src.agents.env_factory import make_eval_env
    from src.eval.backtest import load_run_model, run_backtest

    features_df, env_config, test_idx = artefactos
    model = load_run_model(RUN_DIR, checkpoint="best")

    df1 = run_backtest(model, make_eval_env(features_df, test_idx, env_config))
    df2 = run_backtest(model, make_eval_env(features_df, test_idx, env_config))

    pd.testing.assert_series_equal(df1["wealth"], df2["wealth"])


def test_run_backtest_wealth_consistente_con_retornos(artefactos) -> None:
    """La riqueza final coincide con acumular los retornos diarios netos."""
    from src.agents.env_factory import make_eval_env
    from src.eval.backtest import load_run_model, run_backtest

    features_df, env_config, test_idx = artefactos
    model = load_run_model(RUN_DIR, checkpoint="best")
    df = run_backtest(model, make_eval_env(features_df, test_idx, env_config))

    wealth_recompuesta = float(np.prod(1.0 + df["portfolio_return"].to_numpy()))
    assert wealth_recompuesta == pytest.approx(df["wealth"].iloc[-1], rel=1e-6)


# --- F7-T3: baselines pasivos ------------------------------------------------

def test_baselines_esquema_y_sin_nan(artefactos) -> None:
    """Los 3 baselines devuelven el mismo esquema que ``run_backtest``, sin NaN."""
    from src.eval.baselines import (
        buy_and_hold_spy,
        equal_weight_6assets,
        portfolio_60_40,
    )

    features_df, _, test_idx = artefactos
    columnas = {"action_weights", "portfolio_return", "cost", "turnover",
                "wealth", "drawdown"}
    for fn in (equal_weight_6assets, buy_and_hold_spy, portfolio_60_40):
        df = fn(features_df, test_idx)
        assert columnas.issubset(df.columns), f"{fn.__name__}: faltan columnas"
        assert len(df) == len(test_idx), f"{fn.__name__}: una fila por fecha"
        numericas = ["portfolio_return", "cost", "turnover", "wealth", "drawdown"]
        assert not df[numericas].isna().any().any(), f"{fn.__name__} no debe tener NaN"
        assert (df["cost"] >= 0.0).all(), f"{fn.__name__}: costes ≥ 0"


def test_buy_and_hold_solo_paga_coste_el_primer_dia(artefactos) -> None:
    """Buy-and-hold compra el día 1 y deja correr: coste solo en el primer día."""
    from src.eval.baselines import buy_and_hold_spy

    features_df, _, test_idx = artefactos
    df = buy_and_hold_spy(features_df, test_idx)
    assert df["cost"].iloc[0] > 0.0, "el día de entrada paga comisión"
    assert df["cost"].iloc[1:].sum() == pytest.approx(0.0, abs=1e-9), (
        "sin rebalanceo no hay turnover ni coste tras el día 1"
    )


def test_equal_weight_rebalancea_a_un_sexto(artefactos) -> None:
    """El baseline equiponderado mantiene 1/6 en cada activo y 0 en cash."""
    from src.eval.baselines import equal_weight_6assets

    features_df, _, test_idx = artefactos
    df = equal_weight_6assets(features_df, test_idx)
    pesos = np.stack(df["action_weights"].to_numpy())
    esperado = np.array([1 / 6] * 6 + [0.0])
    assert np.allclose(pesos, esperado, atol=1e-9), "todos los días deben ser 1/6"


# --- F7-T4: evaluate_runs ----------------------------------------------------

_METRICAS = {
    "annualized_return", "annualized_vol", "sharpe_ratio", "sortino_ratio",
    "max_drawdown", "calmar_ratio", "cvar_95", "annualized_turnover",
    "win_rate", "recovery_time",
}


def test_evaluate_runs_tabla_seed_x_metrica(artefactos, tmp_path) -> None:
    """``evaluate_runs`` agrega métricas por seed y persiste CSV + backtest diario."""
    from src.eval.runner import evaluate_runs

    tabla = evaluate_runs("agent_a", [42], split="test", output_dir=tmp_path)

    assert list(tabla.index) == [42], "una fila por seed"
    assert _METRICAS.issubset(tabla.columns), f"faltan métricas: {_METRICAS - set(tabla.columns)}"
    assert (tmp_path / "agent_a_test.csv").exists(), "persiste la tabla CSV"
    bt = tmp_path / "backtests" / "agent_a_seed42_test.parquet"
    assert bt.exists(), "persiste la serie diaria del backtest (la necesita F8)"
    assert "portfolio_return" in pd.read_parquet(bt).columns


# --- F7-T5: evaluate_stress_periods ------------------------------------------

def test_evaluate_stress_periods_una_fila_por_periodo(artefactos, tmp_path) -> None:
    """``evaluate_stress_periods`` recalcula métricas dentro de cada sub-período."""
    from src.eval.runner import evaluate_runs, evaluate_stress_periods

    evaluate_runs("agent_a", [42], split="test", output_dir=tmp_path)
    tabla = evaluate_stress_periods("agent_a", [42], split="test", output_dir=tmp_path)

    assert {"seed", "period"}.issubset(tabla.columns), "indexada por seed y período"
    assert set(tabla["period"]) == {"rates_shock_2023", "ai_rally_2024", "q1_2025_selloff"}
    assert _METRICAS.issubset(tabla.columns), "cada fila trae la batería de métricas"
    assert (tmp_path / "agent_a_stress_periods.csv").exists()


# --- F7-T6: plots ------------------------------------------------------------

def _backtest_sintetico(seed: int, n: int = 60) -> pd.DataFrame:
    """Backtest de juguete (sin modelo) para testear las funciones de plot."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.012, n)
    wealth = np.cumprod(1.0 + returns)
    running_max = np.maximum.accumulate(wealth)
    return pd.DataFrame(
        {
            "portfolio_return": returns,
            "wealth": wealth,
            "drawdown": (wealth - running_max) / running_max,
            "turnover": rng.uniform(0.0, 0.2, n),
            "cost": rng.uniform(0.0, 2e-4, n),
        },
        index=pd.date_range("2023-01-02", periods=n, freq="B"),
    )


def _metricas_sinteticas(seeds: list[int]) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    tabla = pd.DataFrame(
        {m: rng.normal(size=len(seeds)) for m in _METRICAS}, index=seeds
    )
    tabla.index.name = "seed"
    return tabla


def test_plots_devuelven_figuras_matplotlib() -> None:
    """Las 4 funciones de F7-T6 devuelven un ``matplotlib.figure.Figure``."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    from src.eval.plots import (
        plot_drawdown_curves,
        plot_equity_curves,
        plot_metrics_bars,
        plot_return_distributions,
    )

    bts_a = [_backtest_sintetico(s) for s in (0, 1, 42)]
    bts_b = [_backtest_sintetico(s + 100) for s in (0, 1, 42)]
    baselines = {"equal_weight": _backtest_sintetico(999)}
    met_a = _metricas_sinteticas([0, 1, 42])
    met_b = _metricas_sinteticas([0, 1, 42])

    assert isinstance(plot_equity_curves(bts_a, bts_b, baselines), Figure)
    assert isinstance(plot_drawdown_curves(bts_a, bts_b, baselines), Figure)
    assert isinstance(plot_return_distributions(bts_a, bts_b), Figure)
    assert isinstance(plot_metrics_bars(met_a, met_b), Figure)
