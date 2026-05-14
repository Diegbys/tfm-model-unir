"""Feature engineering — F3-T1 a F3-T4.

Transforma el dataset alineado de la Fase 1 en el dataframe de features que
alimenta los dos modelos del TFE:

- **TimeGAN (Fase 4)**: un subconjunto de 9 columnas de retorno (6 log-returns
  de activos + 3 macro derivadas estacionarias).
- **PPO (Fase 5/6)**: el vector completo de 139 features escalado con
  RobustScaler (ver `src.data.scalers`).

Decisiones literales del ADR (`Plan_Implementacion_Fase2_TFE.md` Fase 3):

- Indicadores técnicos vía librería `ta` (Bukosabino) — sustituye a `pandas-ta`
  porque éste exige Python>=3.12 en 0.4.x y el proyecto está pinneado a 3.11.
- Set core compass §1.1: MACD(12,26,9), RSI(14), BBANDS(20,2σ), CCI(14),
  ADX(14), **EMA(10,30,60)** (3 horizontes, decisión confirmada), ATR(14), OBV.
- OBV y EMAs se almacenan crudos; el RobustScaler atenúa magnitud. Trade-off
  documentado en `outputs/preproc/preproc_summary.md`.
- Macro estacionaria: `VIX_log, VIX_delta, EMA(VIX,5), EMA(VIX,21), VIX>25 flag,
  TNX_delta_bps, DXY_log_ret`. ADF test informativo (no aborta).

Las funciones son puras (no leen/escriben disco); el script
`scripts/02_build_features.py` orquesta y persiste.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, CCIIndicator, EMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

from src.data.download import EQUITY_TICKERS_YF, _safe_filename
from src.eval.stylized_facts import log_returns

logger = logging.getLogger(__name__)

# Lookback máximo entre todos los indicadores; usado para descartar filas
# iniciales con NaN. EMA(60) domina sobre BBANDS(20), MACD(26), ADX(14)+14.
MAX_LOOKBACK = 60

# Constante de anualización para volatilidad realizada (252 días hábiles).
SQRT_252 = float(np.sqrt(252))


def compute_returns_and_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    """F3-T1: añade 4 columnas por activo derivadas de OHLCV.

    - ``{ticker}_log_ret``: log-return de AdjClose, reutiliza
      :func:`src.eval.stylized_facts.log_returns` pero conserva la primera fila
      como NaN para preservar el índice completo.
    - ``{ticker}_body_rel``: ``(Close - Open) / Open``.
    - ``{ticker}_range_rel``: ``(High - Low) / Close``.
    - ``{ticker}_log_volume``: ``log(1 + Volume)``.

    No modifica el dataframe de entrada (devuelve copia).
    """
    out = df.copy()
    adj_close_cols = [f"{_safe_filename(t)}_AdjClose" for t in EQUITY_TICKERS_YF]
    # log_returns valida precios > 0 y aplica `dropna(how="all")`; al
    # reindexar al índice original recuperamos la primera fila como NaN.
    returns = log_returns(out[adj_close_cols]).reindex(out.index)
    for ticker in EQUITY_TICKERS_YF:
        prefix = _safe_filename(ticker)
        open_ = out[f"{prefix}_Open"]
        high = out[f"{prefix}_High"]
        low = out[f"{prefix}_Low"]
        close = out[f"{prefix}_Close"]
        volume = out[f"{prefix}_Volume"]

        out[f"{prefix}_log_ret"] = returns[f"{prefix}_AdjClose"]
        out[f"{prefix}_body_rel"] = (close - open_) / open_
        out[f"{prefix}_range_rel"] = (high - low) / close
        out[f"{prefix}_log_volume"] = np.log1p(volume)
    return out


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """F3-T2: añade 15 columnas por activo (MACD×3, RSI, BB×4, CCI, ADX,
    EMA×3, ATR, OBV) usando librería ``ta`` con ``fillna=False`` (causal).

    Las primeras ``MAX_LOOKBACK = 60`` filas tendrán NaN (por EMA(60)).
    """
    out = df.copy()
    for ticker in EQUITY_TICKERS_YF:
        prefix = _safe_filename(ticker)
        high = out[f"{prefix}_High"]
        low = out[f"{prefix}_Low"]
        close = out[f"{prefix}_Close"]
        volume = out[f"{prefix}_Volume"]

        macd = MACD(close=close, window_slow=26, window_fast=12, window_sign=9, fillna=False)
        out[f"{prefix}_macd"] = macd.macd()
        out[f"{prefix}_macd_signal"] = macd.macd_signal()
        out[f"{prefix}_macd_hist"] = macd.macd_diff()

        out[f"{prefix}_rsi"] = RSIIndicator(close=close, window=14, fillna=False).rsi()

        bb = BollingerBands(close=close, window=20, window_dev=2, fillna=False)
        out[f"{prefix}_bb_upper"] = bb.bollinger_hband()
        out[f"{prefix}_bb_lower"] = bb.bollinger_lband()
        out[f"{prefix}_bb_width"] = bb.bollinger_wband()
        out[f"{prefix}_bb_percent"] = bb.bollinger_pband()

        out[f"{prefix}_cci"] = CCIIndicator(
            high=high, low=low, close=close, window=14, fillna=False
        ).cci()
        out[f"{prefix}_adx"] = ADXIndicator(
            high=high, low=low, close=close, window=14, fillna=False
        ).adx()

        out[f"{prefix}_ema_10"] = EMAIndicator(close=close, window=10, fillna=False).ema_indicator()
        out[f"{prefix}_ema_30"] = EMAIndicator(close=close, window=30, fillna=False).ema_indicator()
        out[f"{prefix}_ema_60"] = EMAIndicator(close=close, window=60, fillna=False).ema_indicator()

        out[f"{prefix}_atr"] = AverageTrueRange(
            high=high, low=low, close=close, window=14, fillna=False
        ).average_true_range()
        out[f"{prefix}_obv"] = OnBalanceVolumeIndicator(
            close=close, volume=volume, fillna=False
        ).on_balance_volume()
    return out


def compute_realized_vol(df: pd.DataFrame) -> pd.DataFrame:
    """F3-T3: añade 3 columnas por activo basadas en log-returns previos.

    - ``{ticker}_rv_5d``: ``std(log_ret, 5) * sqrt(252)``.
    - ``{ticker}_rv_21d``: ``std(log_ret, 21) * sqrt(252)``.
    - ``{ticker}_rv_ratio``: ``rv_5d / rv_21d`` (régimen vol relativa).

    Requiere que :func:`compute_returns_and_ohlcv_features` se haya ejecutado
    previamente (necesita ``{ticker}_log_ret``).
    """
    out = df.copy()
    for ticker in EQUITY_TICKERS_YF:
        prefix = _safe_filename(ticker)
        log_ret_col = f"{prefix}_log_ret"
        if log_ret_col not in out.columns:
            raise ValueError(
                f"compute_realized_vol: falta {log_ret_col}. "
                "Ejecuta compute_returns_and_ohlcv_features primero."
            )
        rv_5d = out[log_ret_col].rolling(window=5).std() * SQRT_252
        rv_21d = out[log_ret_col].rolling(window=21).std() * SQRT_252
        out[f"{prefix}_rv_5d"] = rv_5d
        out[f"{prefix}_rv_21d"] = rv_21d
        out[f"{prefix}_rv_ratio"] = rv_5d / rv_21d.replace(0, np.nan)
    return out


def compute_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """F3-T4: añade 7 columnas globales (no replicadas por activo).

    - ``VIX_log``: ``log(VIX_Close)`` (nivel, deliberadamente no estacionario).
    - ``VIX_delta``: ``VIX_Close.diff()`` (puntos).
    - ``VIX_ema_5``, ``VIX_ema_21``: EMA del VIX en horizontes corto/medio.
    - ``VIX_high_flag``: 1 si ``VIX_Close > 25`` (régimen estrés).
    - ``TNX_delta_bps``: cambio de yield 10Y en puntos básicos (TNX en %).
    - ``DXY_log_ret``: log-return del índice del dólar.

    Aplica un test ADF informativo a las 3 columnas "returns-like" (delta y
    log_ret). Loguea WARNING si alguna no rechaza raíz unitaria; nunca aborta.
    """
    out = df.copy()
    vix = out["VIX_Close"]
    tnx = out["TNX_Close"]
    dxy = out["DXY_Close"]

    if (vix <= 0).any() or (dxy <= 0).any():
        raise ValueError("compute_macro_features: VIX o DXY contienen valores <=0")

    out["VIX_log"] = np.log(vix)
    out["VIX_delta"] = vix.diff()
    out["VIX_ema_5"] = EMAIndicator(close=vix, window=5, fillna=False).ema_indicator()
    out["VIX_ema_21"] = EMAIndicator(close=vix, window=21, fillna=False).ema_indicator()
    out["VIX_high_flag"] = (vix > 25).astype(int)
    out["TNX_delta_bps"] = tnx.diff() * 100.0
    out["DXY_log_ret"] = np.log(dxy / dxy.shift(1))

    for col in ("VIX_delta", "TNX_delta_bps", "DXY_log_ret"):
        clean = out[col].dropna()
        if len(clean) < 50:
            logger.warning("ADF %s: solo %d obs, salto", col, len(clean))
            continue
        adf_stat, pvalue = adfuller(clean.to_numpy(), autolag="AIC")[:2]
        if pvalue < 0.05:
            logger.info(
                "ADF %s ✓ estacionaria (stat=%.3f, p=%.4f)", col, adf_stat, pvalue
            )
        else:
            logger.warning(
                "ADF %s ✗ NO estacionaria (stat=%.3f, p=%.4f)", col, adf_stat, pvalue
            )
    return out


# Columnas crudas que se descartan al final (no entran al estado del PPO).
# AdjClose se conserva en cada activo como referencia para backtesting (F7).
_RAW_OHLCV_SUFFIXES_TO_DROP = ("_Open", "_High", "_Low", "_Close", "_Volume")
_RAW_MACRO_COLS_TO_DROP = ("VIX_Close", "TNX_Close", "DXY_Close")


def _drop_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina precios/volúmenes crudos. Conserva ``{ticker}_AdjClose``."""
    to_drop: list[str] = []
    for ticker in EQUITY_TICKERS_YF:
        prefix = _safe_filename(ticker)
        for suffix in _RAW_OHLCV_SUFFIXES_TO_DROP:
            col = f"{prefix}{suffix}"
            if col in df.columns:
                to_drop.append(col)
    for col in _RAW_MACRO_COLS_TO_DROP:
        if col in df.columns:
            to_drop.append(col)
    return df.drop(columns=to_drop)


def build_features(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo F3-T1 → F3-T4 + descarte de filas con NaN inicial.

    Devuelve dataframe con **139 features + 6 AdjClose** = 145 columnas. Las
    primeras filas se eliminan porque las ventanas móviles producen NaN durante
    el periodo de calentamiento (`MAX_LOOKBACK ≈ 60` por EMA(60)).
    """
    logger.info("build_features: input shape %s", aligned_df.shape)
    df = compute_returns_and_ohlcv_features(aligned_df)
    df = compute_technical_indicators(df)
    df = compute_realized_vol(df)
    df = compute_macro_features(df)
    df = _drop_raw_columns(df)

    n_before = len(df)
    df = df.dropna(how="any")
    n_after = len(df)
    n_dropped = n_before - n_after
    pct = 100.0 * n_dropped / n_before
    logger.info(
        "build_features: descartadas %d filas (%.2f%%) por NaN inicial (lookback EMA60=%d)",
        n_dropped, pct, MAX_LOOKBACK,
    )
    if n_dropped > MAX_LOOKBACK + 5:
        logger.warning(
            "build_features: más filas descartadas (%d) que el lookback esperado (~%d)",
            n_dropped, MAX_LOOKBACK,
        )
    logger.info("build_features: output shape %s", df.shape)
    return df


# Lista canónica de las 9 columnas que entran a TimeGAN (compass §1.6 opción A).
TIMEGAN_COLUMNS: tuple[str, ...] = tuple(
    f"{_safe_filename(t)}_log_ret" for t in EQUITY_TICKERS_YF
) + ("VIX_delta", "TNX_delta_bps", "DXY_log_ret")


def get_ppo_feature_columns(features_df: pd.DataFrame) -> list[str]:
    """Lista de las 139 columnas que entran al estado del PPO.

    Excluye explícitamente las 6 ``{ticker}_AdjClose`` (referencia, no estado).
    El orden es el del DataFrame para reproducibilidad.
    """
    return [c for c in features_df.columns if not c.endswith("_AdjClose")]
