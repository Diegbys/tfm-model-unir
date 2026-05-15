"""Reconstrucción OHLCV sintético + macros desde log-returns — F4-T6.

TimeGAN solo produce 9 valores por paso:

- 6 log-returns de equity (AdjClose): GSPC, NDX, AAPL, AMZN, NFLX, NVDA
- 3 deltas macro: VIX_delta, TNX_delta_bps, DXY_log_ret

Pero :func:`src.data.features.build_features` espera columnas OHLCV crudas
(``{ticker}_{Open,High,Low,Close,AdjClose,Volume}``) y macros en niveles
(``VIX_Close``, ``TNX_Close``, ``DXY_Close``). Este módulo reconstruye ese
schema completo (39 columnas, idéntico al ``aligned.parquet`` de Fase 1).

Decisiones del plan F4:

- **4.1**: Open/High/Low/Volume no son producidas por TimeGAN. Las sintetizamos
  con ruido gaussiano calibrado sobre TRAIN del aligned (``range_rel_std``,
  ``body_rel_std`` por ticker).
- **4.2**: Volume sintético = ``exp(N(μ_train, σ_train))`` por ticker,
  **independiente del log_return** (limitación documentada).
- **4.3**: Para macros, integramos las deltas desde un valor inicial real de
  TRAIN para evitar drift acumulativo absurdo en horizontes largos.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.data.download import EQUITY_TICKERS_YF, _safe_filename

logger = logging.getLogger(__name__)


# Orden canónico de las 9 columnas TimeGAN (debe coincidir con TIMEGAN_COLUMNS).
EQUITY_PREFIXES = ["GSPC", "NDX", "AAPL", "AMZN", "NFLX", "NVDA"]


@dataclass
class AlignedTrainStats:
    """Stats calculadas sobre TRAIN del aligned.parquet usadas para sintetizar
    Open/High/Low/Volume y los niveles iniciales de macros.

    Construir vía :func:`compute_aligned_train_stats`.
    """

    # Por ticker: stats de las diferencias relativas y del volumen
    body_rel_std: dict[str, float] = field(default_factory=dict)
    range_rel_mean: dict[str, float] = field(default_factory=dict)
    range_rel_std: dict[str, float] = field(default_factory=dict)
    log_volume_mean: dict[str, float] = field(default_factory=dict)
    log_volume_std: dict[str, float] = field(default_factory=dict)
    # Niveles iniciales (último valor de TRAIN) para arrancar las series
    adj_close_init: dict[str, float] = field(default_factory=dict)
    # Macros: nivel inicial de TRAIN
    vix_close_init: float = 0.0
    tnx_close_init: float = 0.0
    dxy_close_init: float = 0.0
    # Stats macros (no usadas por reconstrucción directa pero útiles para validación)
    vix_close_median: float = 0.0
    tnx_close_median: float = 0.0
    dxy_close_median: float = 0.0


def compute_aligned_train_stats(
    aligned_df: pd.DataFrame, train_idx: pd.DatetimeIndex
) -> AlignedTrainStats:
    """Calcula las stats necesarias para sintetizar OHLCV+macros.

    Todo se mide sobre ``aligned_df.loc[train_idx]`` para mantener
    consistencia anti-leakage.
    """
    if not train_idx.isin(aligned_df.index).all():
        raise ValueError("compute_aligned_train_stats: train_idx no está en aligned_df.index")
    sub = aligned_df.loc[train_idx]
    stats = AlignedTrainStats()

    for prefix in EQUITY_PREFIXES:
        open_ = sub[f"{prefix}_Open"]
        high = sub[f"{prefix}_High"]
        low = sub[f"{prefix}_Low"]
        close = sub[f"{prefix}_Close"]
        adj_close = sub[f"{prefix}_AdjClose"]
        volume = sub[f"{prefix}_Volume"]

        body_rel = (close - open_) / open_
        range_rel = (high - low) / close
        log_volume = np.log1p(volume)

        stats.body_rel_std[prefix] = float(body_rel.std())
        stats.range_rel_mean[prefix] = float(range_rel.mean())
        stats.range_rel_std[prefix] = float(range_rel.std())
        stats.log_volume_mean[prefix] = float(log_volume.mean())
        stats.log_volume_std[prefix] = float(log_volume.std())
        # Punto de arranque: mediana de train (más robusto que último valor).
        stats.adj_close_init[prefix] = float(adj_close.median())

    # Macros: usar mediana de TRAIN como nivel inicial (evita arrancar en outliers)
    stats.vix_close_init = float(sub["VIX_Close"].median())
    stats.tnx_close_init = float(sub["TNX_Close"].median())
    stats.dxy_close_init = float(sub["DXY_Close"].median())
    stats.vix_close_median = stats.vix_close_init
    stats.tnx_close_median = stats.tnx_close_init
    stats.dxy_close_median = stats.dxy_close_init

    logger.info("AlignedTrainStats computadas sobre %d filas de train", len(sub))
    return stats


def reconstruct_aligned_like(
    synthetic_returns: np.ndarray,
    stats: AlignedTrainStats,
    *,
    initial_price: float = 100.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Reconstruye 39 columnas con el mismo schema que ``aligned.parquet`` a
    partir de una **única** secuencia sintética.

    Parameters
    ----------
    synthetic_returns:
        Array shape ``(T, 9)`` con log-returns ya en espacio original
        (post-``scaler.inverse_transform``). Orden de columnas: las 6
        ``{ticker}_log_ret`` seguidas de ``VIX_delta, TNX_delta_bps, DXY_log_ret``.
    stats:
        AlignedTrainStats calculadas sobre TRAIN.
    initial_price:
        Precio inicial para cada equity (decisión 4.1: irrelevante para
        indicadores normalizados, fijado a 100).
    rng:
        Generator para los componentes estocásticos (Open/High/Low/Volume).
        Si ``None``, se crea uno con seed aleatorio.

    Returns
    -------
    DataFrame
        Shape ``(T, 39)``, índice DatetimeIndex de fechas hábiles sintéticas
        (rango arbitrario, ``2099-01-01`` + ``BDay`` para no chocar con datos reales).
    """
    if rng is None:
        rng = np.random.default_rng()
    if synthetic_returns.ndim != 2 or synthetic_returns.shape[1] != 9:
        raise ValueError(
            f"reconstruct_aligned_like: shape esperado (T, 9), got {synthetic_returns.shape}"
        )

    T = synthetic_returns.shape[0]
    # Índice DatetimeIndex sintético (no overlapping con datos reales). Usamos
    # 2099 como base — fuera del rango histórico, claramente sintético.
    idx = pd.date_range("2099-01-01", periods=T, freq=pd.tseries.offsets.BusinessDay())
    out = pd.DataFrame(index=idx)

    # --- Equity: 6 tickers × 6 cols (Open, High, Low, Close, AdjClose, Volume) ---
    for i, prefix in enumerate(EQUITY_PREFIXES):
        log_ret = synthetic_returns[:, i]  # (T,)

        # Close acumulativo: Close[0] = initial_price; Close[t] = Close[t-1]*exp(log_ret[t])
        close = np.empty(T, dtype=np.float64)
        close[0] = initial_price
        for t in range(1, T):
            close[t] = close[t - 1] * np.exp(log_ret[t])

        # Open: t=0 → initial_price; t>0 → Close[t-1] (continuidad simple)
        open_ = np.empty(T, dtype=np.float64)
        open_[0] = initial_price
        open_[1:] = close[:-1]

        # Range relative gaussiano truncado (decisión 4.1)
        # range = |N(0, range_rel_std)| * mean(O,C); High = max(O,C)+range/2, Low=min(O,C)-range/2
        range_rel_std = max(stats.range_rel_std[prefix], 1e-4)
        range_rel = np.abs(rng.normal(0.0, range_rel_std, size=T))
        mid = 0.5 * (open_ + close)
        range_abs = range_rel * mid
        max_oc = np.maximum(open_, close)
        min_oc = np.minimum(open_, close)
        high = max_oc + 0.5 * range_abs
        low = np.maximum(min_oc - 0.5 * range_abs, 1e-6)  # nunca negativo

        # AdjClose = Close (sin dividendos sintéticos; aligned ya es post-ajuste)
        adj_close = close.copy()

        # Volume sintético = exp(N(μ, σ)) - 1 (inverso de log1p)
        log_vol = rng.normal(stats.log_volume_mean[prefix], stats.log_volume_std[prefix], size=T)
        volume = np.maximum(np.expm1(log_vol), 0.0)

        out[f"{prefix}_Open"] = open_
        out[f"{prefix}_High"] = high
        out[f"{prefix}_Low"] = low
        out[f"{prefix}_Close"] = close
        out[f"{prefix}_AdjClose"] = adj_close
        out[f"{prefix}_Volume"] = volume

    # --- Macros: 3 columnas en nivel ---
    vix_delta = synthetic_returns[:, 6]        # VIX_delta
    tnx_delta_bps = synthetic_returns[:, 7]    # TNX_delta_bps
    dxy_log_ret = synthetic_returns[:, 8]      # DXY_log_ret

    # VIX_Close: integrar deltas desde nivel inicial
    vix_close = np.empty(T, dtype=np.float64)
    vix_close[0] = stats.vix_close_init
    for t in range(1, T):
        vix_close[t] = max(vix_close[t - 1] + vix_delta[t], 5.0)  # VIX nunca < 5

    # TNX_Close: TNX_delta_bps = tnx.diff() * 100 ⇒ tnx_close[t] = tnx_close[t-1] + delta_bps/100
    tnx_close = np.empty(T, dtype=np.float64)
    tnx_close[0] = stats.tnx_close_init
    for t in range(1, T):
        tnx_close[t] = max(tnx_close[t - 1] + tnx_delta_bps[t] / 100.0, 0.01)  # yield ≥ 0

    # DXY_Close: log-return acumulativo
    dxy_close = np.empty(T, dtype=np.float64)
    dxy_close[0] = stats.dxy_close_init
    for t in range(1, T):
        dxy_close[t] = dxy_close[t - 1] * np.exp(dxy_log_ret[t])

    out["VIX_Close"] = vix_close
    out["TNX_Close"] = tnx_close
    out["DXY_Close"] = dxy_close

    return out
