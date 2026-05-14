"""Test anti-regresión sobre los stylized facts de los datos reales (F2-T7).

Sirve doble propósito:
1. Confirma que el dataset alineado mantiene las propiedades estadísticas
   esperadas de retornos financieros (fat tails, volatility clustering).
2. Es el mismo criterio que se aplicará a los sintéticos en F4 — si falla con
   datos reales, hay un problema en el pipeline antes de mirar TimeGAN.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.alignment import PROCESSED_PATH
from src.eval.stylized_facts import compute_stylized_facts, log_returns

EQUITIES = ["GSPC", "NDX", "AAPL", "AMZN", "NFLX", "NVDA"]


@pytest.fixture(scope="module")
def aligned_df() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        pytest.skip(
            f"{PROCESSED_PATH} no existe — ejecuta scripts/01_download_data.py primero"
        )
    return pd.read_parquet(PROCESSED_PATH)


@pytest.fixture(scope="module")
def real_facts(aligned_df: pd.DataFrame) -> pd.DataFrame:
    adj_close = aligned_df[[f"{t}_AdjClose" for t in EQUITIES]]
    returns = log_returns(adj_close)
    return compute_stylized_facts(returns)


def test_real_data_has_fat_tails(real_facts: pd.DataFrame) -> None:
    """Todos los activos deben tener kurtosis > 3 (fat tails)."""
    bad = real_facts[real_facts["kurtosis"] <= 3]
    assert bad.empty, (
        f"Kurtosis ≤ 3 (sin fat tails) en: {bad.index.tolist()}; "
        f"valores: {bad['kurtosis'].to_dict()}"
    )


def test_real_data_has_volatility_clustering(real_facts: pd.DataFrame) -> None:
    """ACF(|r|, lag=1) > ACF(r, lag=1) — volatility clustering domina sobre
    autocorrelación lineal en todos los activos."""
    bad = real_facts[real_facts["acf_abs_ret_lag1"] <= real_facts["acf_ret_lag1"]]
    assert bad.empty, (
        f"Sin volatility clustering en: {bad.index.tolist()}; "
        f"acf_ret_lag1: {bad['acf_ret_lag1'].to_dict()}; "
        f"acf_abs_ret_lag1: {bad['acf_abs_ret_lag1'].to_dict()}"
    )
