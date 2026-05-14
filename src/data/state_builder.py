"""Builder de estado para el PPO — F3-T9.

Función pura que produce la observación del agente en un instante ``t`` a
partir del DataFrame de features **ya escalado con RobustScaler**. Devuelve
un tensor numpy de forma ``(window, n_features)`` con las observaciones de
los ``window`` días previos a (e incluyendo) ``t``.

Nota arquitectónica: aquí **NO** se concatenan los pesos actuales de la
cartera; ese trabajo es del `PortfolioEnv` (Fase 5), que añadirá un sub-tensor
de pesos en cada step. Esta separación deja `build_state` puro y testeable
sin depender del MDP.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_WINDOW = 30


def build_state(
    scaled_features_df: pd.DataFrame,
    t: pd.Timestamp,
    window: int = DEFAULT_WINDOW,
) -> np.ndarray:
    """Devuelve la ventana ``[t - window + 1, t]`` como ``np.ndarray``.

    Parameters
    ----------
    scaled_features_df:
        DataFrame ya transformado por ``RobustScaler`` (ver
        :func:`src.data.scalers.transform_with_scaler`). El índice debe ser
        ``DatetimeIndex`` y contener ``t``.
    t:
        Timestamp del día actual (inclusive).
    window:
        Longitud de la ventana. Default 30 según compass §3.6.

    Returns
    -------
    np.ndarray
        Tensor de shape ``(window, n_features)`` con dtype float32. La fila 0
        corresponde a ``t - window + 1`` y la última a ``t``.

    Raises
    ------
    ValueError
        Si ``t`` no está en el índice, si no hay suficiente historia previa
        (``t - window + 1 < index[0]``), o si la ventana contiene NaN.
    """
    if window < 1:
        raise ValueError(f"build_state: window={window} debe ser >=1")
    if t not in scaled_features_df.index:
        raise ValueError(f"build_state: t={t} no está en el índice del DataFrame")

    pos = scaled_features_df.index.get_loc(t)
    if isinstance(pos, slice) or not isinstance(pos, int):
        raise ValueError(f"build_state: t={t} aparece >1 vez en el índice (no único)")
    if pos < window - 1:
        raise ValueError(
            f"build_state: t={t} (pos={pos}) tiene <{window} filas previas; "
            "necesitas más historia o reducir window"
        )

    window_df = scaled_features_df.iloc[pos - window + 1 : pos + 1]
    if window_df.isna().any().any():
        raise ValueError(
            f"build_state: ventana [{window_df.index[0]}, {window_df.index[-1]}] contiene NaN"
        )
    arr = window_df.to_numpy(dtype=np.float32)
    if arr.shape != (window, scaled_features_df.shape[1]):
        # Defensa adicional contra reshape silencioso.
        raise ValueError(
            f"build_state: shape inesperado {arr.shape}, esperaba "
            f"({window}, {scaled_features_df.shape[1]})"
        )
    return arr
