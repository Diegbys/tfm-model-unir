"""Descarga única de datos OHLCV diarios para el TFE.

Fuente primaria: yfinance. Fallback: Stooq vía pandas_datareader.
Persistencia: Parquet por ticker en data/raw/{equities,macro}/.

NO llamar a estas funciones en runtime de entrenamiento. Los datos se
descargan una vez y se versionan vía hash (F9).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr

logger = logging.getLogger(__name__)

EQUITY_TICKERS_YF = ["^GSPC", "^NDX", "AAPL", "AMZN", "NFLX", "NVDA"]
EQUITY_TICKERS_STOOQ = {
    "^GSPC": "^SPX",
    "^NDX": "^NDX",
    "AAPL": "AAPL.US",
    "AMZN": "AMZN.US",
    "NFLX": "NFLX.US",
    "NVDA": "NVDA.US",
}
MACRO_TICKERS_YF = {"VIX": "^VIX", "TNX": "^TNX", "DXY": "DX-Y.NYB"}
MACRO_TICKERS_STOOQ = {"VIX": "^VIX", "TNX": "10USY.B", "DXY": "DX.F"}

START_DATE = "2015-01-01"
END_DATE = "2025-04-30"  # último día deseado en el dataset (inclusive)
# yfinance trata su parámetro `end` como exclusivo, así que internamente
# pasamos END_DATE + 1 día. Stooq y pandas_market_calendars son inclusivos.
_YF_END_EXCLUSIVE = "2025-05-01"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_EQUITIES_DIR = PROJECT_ROOT / "data" / "raw" / "equities"
RAW_MACRO_DIR = PROJECT_ROOT / "data" / "raw" / "macro"

EXPECTED_OHLCV_COLS = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
MIN_EXPECTED_ROWS = 2500


def _safe_filename(ticker: str) -> str:
    return ticker.lstrip("^").replace(".", "_").replace("-", "_")


def _validate_ohlcv(df: pd.DataFrame, ticker: str) -> None:
    if df.empty:
        raise ValueError(f"{ticker}: dataframe vacío tras descarga")
    missing = EXPECTED_OHLCV_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}: faltan columnas {missing}")
    if df.index.duplicated().any():
        raise ValueError(f"{ticker}: fechas duplicadas en el índice")
    if not df.index.is_monotonic_increasing:
        raise ValueError(f"{ticker}: índice de fechas no ordenado")
    if len(df) < MIN_EXPECTED_ROWS:
        raise ValueError(f"{ticker}: solo {len(df)} filas, esperaba ≥{MIN_EXPECTED_ROWS}")
    price_cols = ["Open", "High", "Low", "Close", "Adj Close"]
    if (df[price_cols] <= 0).any().any():
        raise ValueError(f"{ticker}: precios <=0 detectados")
    if (df["Volume"] < 0).any():
        raise ValueError(f"{ticker}: volumen negativo detectado")


def _download_one_yf(ticker: str, retries: int = 3, backoff: float = 2.0) -> pd.DataFrame:
    """Descarga un único ticker desde yfinance con reintentos exponenciales."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            df = yf.download(
                ticker,
                start=START_DATE,
                end=_YF_END_EXCLUSIVE,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if not df.empty:
                return df
            raise ValueError(f"yfinance devolvió dataframe vacío para {ticker}")
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning(
                "yfinance fallo %s intento %d/%d: %s", ticker, attempt + 1, retries, exc
            )
            time.sleep(backoff ** attempt)
    raise RuntimeError(f"yfinance agotó reintentos para {ticker}") from last_err


def _download_one_stooq(stooq_ticker: str) -> pd.DataFrame:
    """Descarga vía Stooq. Recibe directamente el ticker Stooq ya mapeado.

    Stooq devuelve orden descendente y sin 'Adj Close'. Lo normalizamos.
    """
    df = pdr.DataReader(stooq_ticker, "stooq", start=START_DATE, end=END_DATE)
    df = df.sort_index()
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]


def download_equity_with_fallback(ticker_yf: str) -> pd.DataFrame:
    """Intenta yfinance; si falla, cae a Stooq usando EQUITY_TICKERS_STOOQ."""
    try:
        return _download_one_yf(ticker_yf)
    except Exception as exc:  # noqa: BLE001
        stooq_ticker = EQUITY_TICKERS_STOOQ[ticker_yf]
        logger.warning(
            "yfinance agotó reintentos para %s, intentando Stooq (%s): %s",
            ticker_yf,
            stooq_ticker,
            exc,
        )
        return _download_one_stooq(stooq_ticker)


def download_macro_with_fallback(name: str) -> pd.DataFrame:
    """Intenta yfinance con MACRO_TICKERS_YF[name]; fallback Stooq con MACRO_TICKERS_STOOQ[name]."""
    yf_ticker = MACRO_TICKERS_YF[name]
    try:
        return _download_one_yf(yf_ticker)
    except Exception as exc:  # noqa: BLE001
        stooq_ticker = MACRO_TICKERS_STOOQ[name]
        logger.warning(
            "yfinance agotó reintentos para macro %s (%s), intentando Stooq (%s): %s",
            name,
            yf_ticker,
            stooq_ticker,
            exc,
        )
        return _download_one_stooq(stooq_ticker)


def download_equities(out_dir: Path = RAW_EQUITIES_DIR) -> dict[str, Path]:
    """Descarga los 6 activos y persiste como Parquet. Devuelve mapping ticker -> path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for ticker in EQUITY_TICKERS_YF:
        logger.info("Descargando equity %s", ticker)
        df = download_equity_with_fallback(ticker)
        _validate_ohlcv(df, ticker)
        path = out_dir / f"{_safe_filename(ticker)}.parquet"
        df.to_parquet(path)
        logger.info("Guardado %s (%d filas) en %s", ticker, len(df), path)
        paths[ticker] = path
    return paths


def download_macro(out_dir: Path = RAW_MACRO_DIR) -> dict[str, Path]:
    """Descarga VIX, TNX, DXY y persiste como Parquet. Validación de rango para TNX."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in MACRO_TICKERS_YF:
        logger.info("Descargando macro %s (%s)", name, MACRO_TICKERS_YF[name])
        df = download_macro_with_fallback(name)

        if df.empty or len(df) < MIN_EXPECTED_ROWS:
            raise ValueError(f"{name}: macro con solo {len(df)} filas (umbral {MIN_EXPECTED_ROWS})")
        if name == "TNX":
            tnx_close = df["Close"].dropna()
            if not (tnx_close.between(0.5, 50)).all():
                raise ValueError(
                    f"TNX fuera de rango razonable: min={tnx_close.min()}, max={tnx_close.max()}"
                )
        if df.index.duplicated().any():
            raise ValueError(f"{name}: fechas duplicadas")

        path = out_dir / f"{name}.parquet"
        df.to_parquet(path)
        logger.info("Guardado %s (%d filas) en %s", name, len(df), path)
        paths[name] = path
    return paths
