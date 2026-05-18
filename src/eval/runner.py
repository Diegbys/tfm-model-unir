"""Agregación del backtest sobre múltiples seeds — F7-T4 / F7-T5.

``evaluate_runs`` recorre los modelos entrenados de un experimento (Agente A o B),
los backtest-ea sobre un split real y agrega las 9 métricas en una tabla
``seed × métrica``. ``evaluate_stress_periods`` recalcula esas métricas dentro de
los sub-períodos de estrés de ``configs/eval/stress_subperiods.yaml``.

Además de la tabla de métricas, ``evaluate_runs`` **persiste la serie diaria** de
cada backtest (``outputs/eval/backtests/``): la Fase 8 la necesita para el test
de Diebold-Mariano y los plots de F7-T6 la consumen.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.agents.env_factory import make_eval_env
from src.data.download import EQUITY_TICKERS_YF, PROJECT_ROOT, _safe_filename
from src.data.splits import load_splits
from src.eval.backtest import equity_curve, load_run_model, run_backtest
from src.eval.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

PPO_RUNS_DIR = PROJECT_ROOT / "outputs" / "ppo_runs"
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
STRESS_YAML = PROJECT_ROOT / "configs" / "eval" / "stress_subperiods.yaml"

# Columnas de pesos al persistir el backtest (la columna `action_weights` es un
# ndarray por celda y no se serializa limpio a Parquet).
_WEIGHT_COLS = [f"w_{_safe_filename(t)}" for t in EQUITY_TICKERS_YF] + ["w_cash"]


def _split_index(split: str) -> pd.DatetimeIndex:
    """Devuelve el índice de fechas del split ``val`` o ``test``."""
    train_idx, val_idx, test_idx = load_splits()
    indices = {"train": train_idx, "val": val_idx, "test": test_idx}
    if split not in indices:
        raise ValueError(f"split debe ser uno de {list(indices)}, no {split!r}")
    return indices[split]


def _suffix(checkpoint: str) -> str:
    """Sufijo de los ficheros: vacío para ``best``, ``_final`` para el modelo final."""
    return "" if checkpoint == "best" else "_final"


def _backtest_path(
    output_dir: Path, experiment: str, seed: int, split: str, checkpoint: str
) -> Path:
    return output_dir / "backtests" / (
        f"{experiment}_seed{seed}_{split}{_suffix(checkpoint)}.parquet"
    )


def _persist_backtest(backtest_df: pd.DataFrame, path: Path) -> None:
    """Guarda la serie diaria del backtest expandiendo los pesos en columnas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = np.stack(backtest_df["action_weights"].to_numpy())
    out = backtest_df.drop(columns=["action_weights"]).copy()
    for i, col in enumerate(_WEIGHT_COLS):
        out[col] = weights[:, i]
    out.to_parquet(path)


def evaluate_runs(
    experiment: str,
    seeds: list[int],
    split: str = "test",
    checkpoint: str = "best",
    output_dir: str | Path = EVAL_DIR,
) -> pd.DataFrame:
    """Backtest-ea todos los seeds de un experimento y agrega métricas por seed.

    Parameters
    ----------
    experiment:
        ``"agent_a"`` o ``"agent_b"``.
    seeds:
        Lista de seeds a evaluar (p. ej. ``[0, 1, 42, 123, 1337]``).
    split:
        ``"val"`` o ``"test"`` — el split real sobre el que se backtest-ea.
    checkpoint:
        ``"best"`` (mejor Sharpe en val, primario) o ``"final"`` (modelo final).
    output_dir:
        Carpeta donde se persisten ``<exp>_<split>.csv`` y los backtests diarios.

    Returns
    -------
    pd.DataFrame
        Tabla indexada por ``seed`` con las 9 métricas (más ``recovery_time``).
    """
    output_dir = Path(output_dir)
    features_df = pd.read_parquet(FEATURES_PATH)
    split_idx = _split_index(split)

    rows: dict[int, dict] = {}
    for seed in seeds:
        run_dir = PPO_RUNS_DIR / experiment / f"seed{seed}"
        env_config = OmegaConf.load(run_dir / "config.yaml").env
        model = load_run_model(run_dir, checkpoint)
        env = make_eval_env(features_df, split_idx, env_config)
        backtest_df = run_backtest(model, env)
        rows[seed] = compute_all_metrics(
            backtest_df["portfolio_return"].to_numpy(),
            equity_curve(backtest_df),
            backtest_df["turnover"].to_numpy(),
        )
        _persist_backtest(
            backtest_df, _backtest_path(output_dir, experiment, seed, split, checkpoint)
        )
        logger.info(
            "evaluate_runs: %s seed=%d %s — Sharpe=%.4f, MDD=%.4f",
            experiment, seed, split, rows[seed]["sharpe_ratio"], rows[seed]["max_drawdown"],
        )

    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "seed"
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / f"{experiment}_{split}{_suffix(checkpoint)}.csv")
    return table


def evaluate_stress_periods(
    experiment: str,
    seeds: list[int],
    split: str = "test",
    checkpoint: str = "best",
    output_dir: str | Path = EVAL_DIR,
) -> pd.DataFrame:
    """Recalcula las métricas dentro de cada sub-período de estrés (F7-T5).

    Lee las series diarias persistidas por ``evaluate_runs``, filtra cada
    sub-período de ``stress_subperiods.yaml`` y **re-basa la riqueza a 1.0** al
    inicio de cada ventana antes de recomputar las métricas.

    Returns
    -------
    pd.DataFrame
        Una fila por ``(seed, período)`` con la batería de métricas.
    """
    output_dir = Path(output_dir)
    periods = OmegaConf.load(STRESS_YAML).stress_subperiods

    rows: list[dict] = []
    for seed in seeds:
        backtest_df = pd.read_parquet(
            _backtest_path(output_dir, experiment, seed, split, checkpoint)
        )
        for period in periods:
            start, end = pd.Timestamp(str(period.start)), pd.Timestamp(str(period.end))
            window = backtest_df.loc[
                (backtest_df.index >= start) & (backtest_df.index <= end)
            ]
            if window.empty:
                logger.warning(
                    "evaluate_stress_periods: %s sin datos en test", period.label
                )
                continue
            returns = window["portfolio_return"].to_numpy()
            equity = np.concatenate([[1.0], np.cumprod(1.0 + returns)])  # re-basada
            metrics = compute_all_metrics(returns, equity, window["turnover"].to_numpy())
            rows.append(
                {"seed": seed, "period": period.label, "n_days": len(window), **metrics}
            )

    table = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        output_dir / f"{experiment}_stress_periods{_suffix(checkpoint)}.csv", index=False
    )
    return table
