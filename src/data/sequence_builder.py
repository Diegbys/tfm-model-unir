"""Builder de secuencias para TimeGAN — F3-T10.

Produce el tensor de shape ``(N_seq, seq_len, n_dims)`` que consume TimeGAN
(Fase 4) durante el entrenamiento. Las secuencias son ventanas deslizantes
con ``stride=1`` sobre el subset de 9 columnas de train escaladas a [0,1].
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.data.download import PROJECT_ROOT

logger = logging.getLogger(__name__)

SEQUENCES_PATH = PROJECT_ROOT / "data" / "processed" / "timegan_train_sequences.npy"

DEFAULT_SEQ_LEN = 24
DEFAULT_STRIDE = 1


def build_timegan_sequences(
    features_df: pd.DataFrame,
    train_idx: pd.DatetimeIndex,
    scaler: MinMaxScaler,
    cols: list[str],
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: int = DEFAULT_STRIDE,
) -> np.ndarray:
    """Aplica ``scaler.transform`` al subset y genera ventanas deslizantes.

    Parameters
    ----------
    features_df:
        DataFrame completo con las 9 columnas TimeGAN (ya en escala original).
    train_idx:
        DatetimeIndex de train (compass §1.6: TimeGAN solo ve train).
    scaler:
        ``MinMaxScaler`` ya ajustado sobre train (ver
        :func:`src.data.scalers.fit_timegan_scaler`).
    cols:
        Lista de 9 columnas en orden canónico (devuelta por fit_timegan_scaler).
    seq_len:
        Longitud de cada secuencia. Default 24 (compass §1.6, paper Yoon 2019).
    stride:
        Paso entre secuencias consecutivas. Default 1 (overlapping, estándar).

    Returns
    -------
    np.ndarray
        Tensor ``(N_seq, seq_len, n_dims=9)`` dtype float32, valores ≈ [0,1]
        (train ⇒ exactamente [0,1]; otros datos podrían exceder, pero aquí
        solo escalamos train).
    """
    if seq_len < 2:
        raise ValueError(f"build_timegan_sequences: seq_len={seq_len} debe ser >=2")
    if stride < 1:
        raise ValueError(f"build_timegan_sequences: stride={stride} debe ser >=1")

    missing = [c for c in cols if c not in features_df.columns]
    if missing:
        raise ValueError(f"build_timegan_sequences: faltan columnas {missing}")

    train_data = features_df.loc[train_idx, cols]
    if train_data.isna().any().any():
        raise ValueError("build_timegan_sequences: NaN en train (debería estar limpio)")

    scaled = scaler.transform(train_data.to_numpy())
    n_rows = scaled.shape[0]
    if n_rows < seq_len:
        raise ValueError(
            f"build_timegan_sequences: train tiene {n_rows} filas < seq_len={seq_len}"
        )

    n_seq = (n_rows - seq_len) // stride + 1
    sequences = np.empty((n_seq, seq_len, len(cols)), dtype=np.float32)
    for i in range(n_seq):
        start = i * stride
        sequences[i] = scaled[start : start + seq_len]

    logger.info(
        "TimeGAN secuencias: shape %s (train=%d filas, seq_len=%d, stride=%d)",
        sequences.shape, n_rows, seq_len, stride,
    )
    return sequences


def persist_sequences(sequences: np.ndarray, path: Path = SEQUENCES_PATH) -> Path:
    """Persiste el tensor como ``.npy`` (nativo numpy)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, sequences)
    logger.info("Secuencias persistidas en %s (shape %s)", path, sequences.shape)
    return path


def load_sequences(path: Path = SEQUENCES_PATH) -> np.ndarray:
    """Carga el tensor persistido por :func:`persist_sequences`."""
    if not path.exists():
        raise FileNotFoundError(f"load_sequences: falta {path}")
    return np.load(path)
