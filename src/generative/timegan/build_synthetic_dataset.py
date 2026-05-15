"""Dataset sintético con indicadores técnicos — F4-T7.

Reutiliza :func:`src.data.features.build_features` para que el dataset
sintético tenga **exactamente el mismo schema** (145 columnas) que
``data/processed/features.parquet``.

**Decisión 4.3 del plan F4** (warmup de EMA(60)):

- Cada secuencia sintética tiene solo 24 pasos, pero ``MAX_LOOKBACK=60``
  requiere 60 filas previas para evitar NaN en EMA(60), realized vol, etc.
- Para cada secuencia: **prepend** 60 días reales aleatorios de TRAIN
  (seed-dependent), ejecutar ``build_features``, descartar las primeras 60.
- Alternativas peores: zero-padding (introduce indicadores falsos),
  descartar inicio (reduce N_synth).

Output: DataFrame MultiIndex ``(seq_id, step)`` con 145 columnas.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features import MAX_LOOKBACK, build_features
from src.generative.timegan.reconstruct_prices import (
    AlignedTrainStats,
    reconstruct_aligned_like,
)

logger = logging.getLogger(__name__)


def build_synthetic_dataset(
    synthetic_returns: np.ndarray,
    aligned_df: pd.DataFrame,
    train_idx: pd.DatetimeIndex,
    stats: AlignedTrainStats,
    *,
    initial_price: float = 100.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Ensambla el dataset sintético completo.

    Parameters
    ----------
    synthetic_returns:
        Array ``(n_seq, T, 9)`` con log-returns en espacio original.
    aligned_df:
        ``aligned.parquet`` completo (para muestrear el buffer histórico).
    train_idx:
        Fechas de train (anti-leakage: solo se muestrea de aquí).
    stats:
        AlignedTrainStats para reconstruir OHLCV+macros.
    initial_price:
        Precio inicial sintético (default 100.0).
    seed:
        Seed del muestreo de buffers y del ruido de range/volume.

    Returns
    -------
    DataFrame
        MultiIndex ``(seq_id, step)``, 145 columnas (139 features + 6 AdjClose).
    """
    n_seq, T, n_feat = synthetic_returns.shape
    if n_feat != 9:
        raise ValueError(f"synthetic_returns: shape (N, T, 9) esperado; got {synthetic_returns.shape}")

    rng = np.random.default_rng(seed)
    # Solo podemos muestrear buffers cuyas 60 filas previas caen en train_idx
    train_pos = np.array([aligned_df.index.get_loc(d) for d in train_idx])
    # Excluir las primeras MAX_LOOKBACK posiciones (no tienen warmup)
    valid_buffer_pos = train_pos[train_pos >= MAX_LOOKBACK]
    if len(valid_buffer_pos) == 0:
        raise ValueError(
            f"build_synthetic_dataset: no hay posiciones válidas en train para buffer "
            f"de MAX_LOOKBACK={MAX_LOOKBACK} (train_idx primer pos={train_pos.min()})"
        )

    logger.info(
        "build_synthetic_dataset: %d secuencias × T=%d, buffer=%d días, seed=%d",
        n_seq, T, MAX_LOOKBACK, seed,
    )

    pieces: list[pd.DataFrame] = []
    for seq_id in range(n_seq):
        # Buffer: 60 filas reales contiguas de TRAIN
        buf_end_pos = int(rng.choice(valid_buffer_pos))
        buf_start_pos = buf_end_pos - MAX_LOOKBACK + 1
        buffer_real = aligned_df.iloc[buf_start_pos : buf_end_pos + 1].copy()
        # Renombrar el índice del buffer a fechas sintéticas (continuas con la secuencia)
        # para que build_features no se confunda con saltos.
        # Las 60 filas del buffer ocuparán [2098-09-01 .. 2098-...], la seq [2099-01-01 ...]
        # Calcular fechas hábiles que terminen justo antes del 2099-01-01.
        buffer_idx = pd.date_range(
            end=pd.Timestamp("2098-12-31"),
            periods=MAX_LOOKBACK,
            freq=pd.tseries.offsets.BusinessDay(),
        )
        buffer_real.index = buffer_idx

        # Reconstrucción OHLCV+macros para los T pasos sintéticos
        synth_aligned = reconstruct_aligned_like(
            synthetic_returns[seq_id], stats, initial_price=initial_price, rng=rng,
        )

        # Concatenar buffer + sintético y ejecutar build_features
        full = pd.concat([buffer_real, synth_aligned], axis=0)
        try:
            features_df = build_features(full)
        except Exception as exc:  # noqa: BLE001
            logger.warning("seq_id=%d: build_features falló (%s) — saltando", seq_id, exc)
            continue

        # Descartar todo lo que provenga del buffer: solo conservamos las filas
        # del rango sintético (índice ≥ 2099-01-01).
        cutoff = pd.Timestamp("2099-01-01")
        synth_only = features_df.loc[features_df.index >= cutoff]
        if len(synth_only) < T:
            logger.warning(
                "seq_id=%d: tras descartar buffer quedan %d/%d filas",
                seq_id, len(synth_only), T,
            )
        synth_only = synth_only.head(T)  # asegurar T exactos

        # MultiIndex (seq_id, step)
        synth_only = synth_only.reset_index(drop=True)
        synth_only.index = pd.MultiIndex.from_product(
            [[seq_id], range(len(synth_only))],
            names=["seq_id", "step"],
        )
        pieces.append(synth_only)

        if (seq_id + 1) % 200 == 0:
            logger.info("  %d/%d secuencias procesadas", seq_id + 1, n_seq)

    if not pieces:
        raise RuntimeError(
            "build_synthetic_dataset: ninguna secuencia válida; build_features falló en todas"
        )

    dataset = pd.concat(pieces, axis=0)
    logger.info(
        "build_synthetic_dataset OK: %d secuencias × %d cols = %d filas totales",
        len(pieces), dataset.shape[1], dataset.shape[0],
    )
    return dataset


def persist_synthetic_dataset(df: pd.DataFrame, output_dir: Path) -> Path:
    """Persiste como Parquet en ``{output_dir}/synthetic_dataset.parquet``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "synthetic_dataset.parquet"
    df.to_parquet(path)
    logger.info("Synthetic dataset persistido en %s (%d filas × %d cols)",
                path, df.shape[0], df.shape[1])
    return path
