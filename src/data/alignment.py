"""Alineación de equities + macros al calendario XNYS de NYSE/NASDAQ.

Regla del ADR (§F1, §research 1.4):
- Para equities: si algún activo tiene NaN en un día del XNYS, se elimina la fila entera.
  Loggear % eliminado. Debe ser <0.5%.
- Para macros: forward-fill sólo en huecos ≤3 días.
"""
from __future__ import annotations

import logging

import pandas as pd
import pandas_market_calendars as mcal

from src.data.download import (
    EQUITY_TICKERS_YF,
    END_DATE,
    MACRO_TICKERS_YF,
    PROJECT_ROOT,
    RAW_EQUITIES_DIR,
    RAW_MACRO_DIR,
    START_DATE,
    _safe_filename,
)

logger = logging.getLogger(__name__)

PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "aligned.parquet"
MAX_FFILL_DAYS = 3
MAX_DROPPED_PCT = 0.5


def _xnys_business_days(start: str, end: str) -> pd.DatetimeIndex:
    """Universo canónico de días hábiles XNYS entre start y end (ambos inclusive)."""
    nyse = mcal.get_calendar("XNYS")
    schedule = nyse.schedule(start_date=start, end_date=end)
    return pd.DatetimeIndex(schedule.index.normalize())


def _load_equity(ticker: str) -> pd.DataFrame:
    safe = _safe_filename(ticker)
    df = pd.read_parquet(RAW_EQUITIES_DIR / f"{safe}.parquet")
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df.rename(columns={c: f"{safe}_{c.replace(' ', '')}" for c in df.columns})


def _load_macro(name: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW_MACRO_DIR / f"{name}.parquet")
    df.index = pd.DatetimeIndex(df.index).normalize()
    return df[["Close"]].rename(columns={"Close": f"{name}_Close"})


def align_to_xnys() -> pd.DataFrame:
    """Combina equities + macros alineados al calendario XNYS, persiste y devuelve."""
    xnys = _xnys_business_days(START_DATE, END_DATE)
    logger.info(
        "Calendario XNYS: %d días hábiles entre %s y %s", len(xnys), START_DATE, END_DATE
    )

    equity_frames = [_load_equity(t).reindex(xnys) for t in EQUITY_TICKERS_YF]
    equities = pd.concat(equity_frames, axis=1)
    n_total = len(equities)
    equities_clean = equities.dropna(how="any")
    n_dropped = n_total - len(equities_clean)
    pct_dropped = 100.0 * n_dropped / n_total
    logger.info("Equities: eliminadas %d filas (%.3f%%) con NaN", n_dropped, pct_dropped)
    if pct_dropped > MAX_DROPPED_PCT:
        raise ValueError(
            f"Demasiadas filas con NaN en equities: {pct_dropped:.3f}% (umbral {MAX_DROPPED_PCT}%)"
        )

    macro_frames = [
        _load_macro(name).reindex(xnys).ffill(limit=MAX_FFILL_DAYS)
        for name in MACRO_TICKERS_YF
    ]
    macros = pd.concat(macro_frames, axis=1)
    macros_aligned = macros.loc[equities_clean.index]
    n_macro_nan = int(macros_aligned.isna().sum().sum())
    if n_macro_nan > 0:
        logger.warning(
            "Macros: %d valores NaN restantes tras ffill≤%d, eliminando esas filas",
            n_macro_nan,
            MAX_FFILL_DAYS,
        )
        valid_mask = macros_aligned.notna().all(axis=1)
        equities_clean = equities_clean.loc[valid_mask]
        macros_aligned = macros_aligned.loc[valid_mask]

    combined = pd.concat([equities_clean, macros_aligned], axis=1)
    combined.index.name = "date"
    logger.info("Dataset combinado: %d filas × %d columnas", *combined.shape)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(PROCESSED_PATH)
    logger.info("Persistido en %s", PROCESSED_PATH)
    return combined
