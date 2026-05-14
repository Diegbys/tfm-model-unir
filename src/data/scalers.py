"""Scalers para TimeGAN y PPO — F3-T6, F3-T7.

Persisten dos scalers independientes ajustados **solo en train** (regla
cardinal anti-leakage del ADR §F3-T8 y compass §4.2):

- **MinMaxScaler** para TimeGAN: rango [0,1] requerido por la sigmoide del
  recovery network (compass §1.3). Subset de 9 columnas (compass §1.6 opción A).
- **RobustScaler** para PPO: basado en mediana/IQR, insensible a outliers
  (COVID 2020, NVDA 2023). Aplica a las 139 features del estado.

Cada scaler se persiste con joblib junto a un JSON con la lista de columnas
en orden, para que el consumidor pueda validar consistencia.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler

from src.data.download import PROJECT_ROOT
from src.data.features import TIMEGAN_COLUMNS, get_ppo_feature_columns

logger = logging.getLogger(__name__)

SCALERS_DIR = PROJECT_ROOT / "models" / "scalers"
TIMEGAN_SCALER_PATH = SCALERS_DIR / "timegan_minmax.joblib"
TIMEGAN_COLUMNS_PATH = SCALERS_DIR / "timegan_columns.json"
PPO_SCALER_PATH = SCALERS_DIR / "ppo_robust.joblib"
PPO_COLUMNS_PATH = SCALERS_DIR / "ppo_columns.json"


def fit_timegan_scaler(
    features_df: pd.DataFrame,
    train_idx: pd.DatetimeIndex,
    out_dir: Path = SCALERS_DIR,
) -> tuple[MinMaxScaler, list[str]]:
    """Ajusta MinMaxScaler sobre 9 columnas de train y persiste.

    Las columnas son las definidas en :data:`src.data.features.TIMEGAN_COLUMNS`:
    6 ``{ticker}_log_ret`` + ``VIX_delta`` + ``TNX_delta_bps`` + ``DXY_log_ret``.

    Persiste el scaler en ``timegan_minmax.joblib`` y la lista de columnas en
    ``timegan_columns.json`` (orden canónico inmutable).
    """
    cols = list(TIMEGAN_COLUMNS)
    missing = [c for c in cols if c not in features_df.columns]
    if missing:
        raise ValueError(f"fit_timegan_scaler: faltan columnas {missing}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = features_df.loc[train_idx, cols]
    if train_data.isna().any().any():
        raise ValueError(
            "fit_timegan_scaler: train contiene NaN en columnas TimeGAN; "
            "build_features debería haberlas eliminado"
        )

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(train_data.to_numpy())

    joblib.dump(scaler, out_dir / "timegan_minmax.joblib")
    with (out_dir / "timegan_columns.json").open("w") as f:
        json.dump(cols, f, indent=2)
    logger.info(
        "TimeGAN MinMaxScaler ajustado sobre %d filas × %d cols, persistido en %s",
        len(train_data), len(cols), out_dir,
    )
    return scaler, cols


def fit_ppo_scaler(
    features_df: pd.DataFrame,
    train_idx: pd.DatetimeIndex,
    out_dir: Path = SCALERS_DIR,
) -> tuple[RobustScaler, list[str]]:
    """Ajusta RobustScaler sobre las 139 features del estado del PPO.

    Excluye las 6 ``{ticker}_AdjClose`` (referencia, no estado). Quantiles
    ``(25, 75)`` por defecto = IQR robusto a outliers.
    """
    cols = get_ppo_feature_columns(features_df)
    if not cols:
        raise ValueError("fit_ppo_scaler: get_ppo_feature_columns devolvió lista vacía")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_data = features_df.loc[train_idx, cols]
    if train_data.isna().any().any():
        raise ValueError("fit_ppo_scaler: train contiene NaN")

    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    scaler.fit(train_data.to_numpy())

    joblib.dump(scaler, out_dir / "ppo_robust.joblib")
    with (out_dir / "ppo_columns.json").open("w") as f:
        json.dump(cols, f, indent=2)
    logger.info(
        "PPO RobustScaler ajustado sobre %d filas × %d cols, persistido en %s",
        len(train_data), len(cols), out_dir,
    )
    return scaler, cols


def load_timegan_scaler(in_dir: Path = SCALERS_DIR) -> tuple[MinMaxScaler, list[str]]:
    """Carga MinMaxScaler + columnas para TimeGAN."""
    scaler = joblib.load(in_dir / "timegan_minmax.joblib")
    with (in_dir / "timegan_columns.json").open() as f:
        cols = json.load(f)
    return scaler, cols


def load_ppo_scaler(in_dir: Path = SCALERS_DIR) -> tuple[RobustScaler, list[str]]:
    """Carga RobustScaler + columnas para PPO."""
    scaler = joblib.load(in_dir / "ppo_robust.joblib")
    with (in_dir / "ppo_columns.json").open() as f:
        cols = json.load(f)
    return scaler, cols


def transform_with_scaler(
    features_df: pd.DataFrame,
    scaler: MinMaxScaler | RobustScaler,
    cols: list[str],
) -> pd.DataFrame:
    """Aplica ``scaler.transform`` y devuelve DataFrame con mismo índice/cols.

    Helper común para val/test downstream. Lanza ``ValueError`` si faltan
    columnas, evitando errores silenciosos por reordering.
    """
    missing = [c for c in cols if c not in features_df.columns]
    if missing:
        raise ValueError(f"transform_with_scaler: faltan columnas {missing}")
    arr = scaler.transform(features_df[cols].to_numpy())
    return pd.DataFrame(arr, index=features_df.index, columns=cols)
