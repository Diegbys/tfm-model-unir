"""Tests de integridad para el dataset alineado producido por la Fase 1.

Línea de defensa contra regresiones cada vez que se re-descargue o se re-alinee
el dataset. NO testean lógica de red — eso es responsabilidad de smoke tests
manuales en F1-T3..T5.
"""
from __future__ import annotations

import pandas as pd
import pandas_market_calendars as mcal
import pytest

from src.data.alignment import PROCESSED_PATH
from src.data.download import (
    EQUITY_TICKERS_YF,
    END_DATE,
    MACRO_TICKERS_YF,
    START_DATE,
    _safe_filename,
)


@pytest.fixture(scope="module")
def aligned_df() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        pytest.skip(
            f"{PROCESSED_PATH} no existe — ejecuta scripts/01_download_data.py primero"
        )
    return pd.read_parquet(PROCESSED_PATH)


def test_aligned_dataset_row_count(aligned_df: pd.DataFrame) -> None:
    """Número de filas debe coincidir aproximadamente con el calendario XNYS."""
    nyse = mcal.get_calendar("XNYS")
    expected = len(nyse.schedule(start_date=START_DATE, end_date=END_DATE))
    # Permitimos hasta MAX_DROPPED_PCT (0.5%) de filas eliminadas por NaN en equities.
    assert 0.99 * expected <= len(aligned_df) <= expected, (
        f"Filas {len(aligned_df)} fuera del rango esperado "
        f"[{0.99 * expected:.0f}, {expected}]"
    )


def test_aligned_dataset_no_nans(aligned_df: pd.DataFrame) -> None:
    nan_counts = aligned_df.isna().sum()
    bad = nan_counts[nan_counts > 0]
    assert bad.empty, f"NaN restantes en columnas: {bad.to_dict()}"


def test_aligned_dataset_no_negative_prices(aligned_df: pd.DataFrame) -> None:
    price_like = [
        c
        for c in aligned_df.columns
        if any(
            c.endswith(suf) for suf in ("_Open", "_High", "_Low", "_Close", "_AdjClose")
        )
    ]
    assert price_like, "No se encontraron columnas de precio en el dataset"
    assert (aligned_df[price_like] > 0).all().all(), "Precios <=0 detectados"

    volume_cols = [c for c in aligned_df.columns if c.endswith("_Volume")]
    if volume_cols:
        assert (aligned_df[volume_cols] >= 0).all().all(), "Volúmenes negativos detectados"


def test_aligned_dataset_index_unique_sorted(aligned_df: pd.DataFrame) -> None:
    idx = aligned_df.index
    assert idx.is_unique, "Fechas duplicadas en el índice"
    assert idx.is_monotonic_increasing, "Índice no estrictamente creciente"


def test_aligned_dataset_expected_columns(aligned_df: pd.DataFrame) -> None:
    """6 activos × 6 OHLCV + 3 macros × 1 Close = 39 cols."""
    expected_equity_prefixes = {_safe_filename(t) for t in EQUITY_TICKERS_YF}
    for prefix in expected_equity_prefixes:
        for suffix in ("Open", "High", "Low", "Close", "AdjClose", "Volume"):
            assert f"{prefix}_{suffix}" in aligned_df.columns, (
                f"Columna esperada {prefix}_{suffix} no encontrada"
            )
    for macro in MACRO_TICKERS_YF:
        assert f"{macro}_Close" in aligned_df.columns, (
            f"Columna macro {macro}_Close no encontrada"
        )
