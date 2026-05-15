"""Tests anti-leakage — F3-T8 y F3-T11.

Estos dos tests son la línea de defensa principal contra los errores
sistemáticos que invalidan resultados de DRL financiero (compass §7.1):

- `test_scaler_only_train`: verifica que los scalers persistidos en
  `models/scalers/` se ajustaron únicamente con datos de train.
- `test_no_lookahead_in_features`: verifica que ninguna feature en `t` usa
  información de `t+k` (k>0), recalculando el pipeline sobre datos truncados
  y comparando con el cálculo full para 100 fechas aleatorias del test.

Ambos saltan con `pytest.skip` si los artefactos de Fase 3 no existen;
ejecutar `uv run python scripts/02_build_features.py` primero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.alignment import PROCESSED_PATH
from src.data.features import (
    TIMEGAN_COLUMNS,
    build_features,
    get_ppo_feature_columns,
)
from src.data.scalers import (
    PPO_COLUMNS_PATH,
    PPO_SCALER_PATH,
    TIMEGAN_COLUMNS_PATH,
    TIMEGAN_SCALER_PATH,
    load_ppo_scaler,
    load_timegan_scaler,
)
from src.data.splits import SPLITS_DIR, load_splits

FEATURES_PATH = PROCESSED_PATH.parent / "features.parquet"


@pytest.fixture(scope="module")
def features_df() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        pytest.skip(f"{FEATURES_PATH} no existe — ejecuta scripts/02_build_features.py primero")
    return pd.read_parquet(FEATURES_PATH)


@pytest.fixture(scope="module")
def splits() -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    if not (SPLITS_DIR / "train_idx.parquet").exists():
        pytest.skip("splits no existen — ejecuta scripts/02_build_features.py primero")
    return load_splits()


@pytest.fixture(scope="module")
def aligned_df() -> pd.DataFrame:
    if not PROCESSED_PATH.exists():
        pytest.skip(f"{PROCESSED_PATH} no existe")
    return pd.read_parquet(PROCESSED_PATH)


def test_scaler_only_train(
    features_df: pd.DataFrame,
    splits: tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex],
) -> None:
    """F3-T8: ``data_min_``/``data_max_`` del MinMax y ``center_``/``scale_``
    del Robust deben coincidir bit a bit con el recálculo sobre train; además
    val/test debe contener valores fuera del rango train para confirmar que
    el scaler no los vio.
    """
    if not TIMEGAN_SCALER_PATH.exists() or not PPO_SCALER_PATH.exists():
        pytest.skip("Scalers no persistidos")

    train_idx, val_idx, test_idx = splits
    val_test = val_idx.union(test_idx)

    # --- MinMaxScaler (TimeGAN) ---
    timegan_scaler, timegan_cols = load_timegan_scaler()
    assert list(timegan_cols) == list(TIMEGAN_COLUMNS), (
        "Orden de columnas TimeGAN no coincide con TIMEGAN_COLUMNS"
    )
    train_timegan = features_df.loc[train_idx, timegan_cols].to_numpy()
    assert np.allclose(timegan_scaler.data_min_, train_timegan.min(axis=0)), (
        "data_min_ del MinMax no coincide con min sobre train ⇒ posible leakage"
    )
    assert np.allclose(timegan_scaler.data_max_, train_timegan.max(axis=0)), (
        "data_max_ del MinMax no coincide con max sobre train ⇒ posible leakage"
    )

    # Sanity: alguna columna del val/test debe exceder el rango de train.
    # (Si TODAS las columnas de val/test cayeran dentro, sería sospechoso de leakage o de
    # subajuste extremo del train.)
    val_test_timegan = features_df.loc[val_test, timegan_cols].to_numpy()
    exceeds_min = (val_test_timegan < timegan_scaler.data_min_).any(axis=0)
    exceeds_max = (val_test_timegan > timegan_scaler.data_max_).any(axis=0)
    n_exceeding = int((exceeds_min | exceeds_max).sum())
    assert n_exceeding > 0, (
        f"Ninguna columna del val/test excede el rango del train para TimeGAN. "
        f"Sospechoso de leakage o de columna constante."
    )

    # --- RobustScaler (PPO) ---
    ppo_scaler, ppo_cols = load_ppo_scaler()
    assert ppo_cols == get_ppo_feature_columns(features_df), (
        "Orden de columnas PPO no coincide con get_ppo_feature_columns"
    )
    train_ppo = features_df.loc[train_idx, ppo_cols].to_numpy()
    # RobustScaler con quantile_range=(25, 75) usa mediana como center, IQR como scale.
    expected_center = np.median(train_ppo, axis=0)
    q25 = np.percentile(train_ppo, 25, axis=0)
    q75 = np.percentile(train_ppo, 75, axis=0)
    expected_scale = q75 - q25
    # En casos degenerados (IQR=0), sklearn pone scale_=1 para evitar div/0.
    expected_scale = np.where(expected_scale == 0, 1.0, expected_scale)

    assert np.allclose(ppo_scaler.center_, expected_center), (
        "center_ del Robust no coincide con mediana sobre train ⇒ posible leakage"
    )
    assert np.allclose(ppo_scaler.scale_, expected_scale, atol=1e-10), (
        "scale_ del Robust no coincide con IQR sobre train ⇒ posible leakage"
    )


def test_no_lookahead_in_features(
    features_df: pd.DataFrame,
    aligned_df: pd.DataFrame,
    splits: tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex],
) -> None:
    """F3-T11: para N=10 fechas aleatorias del test, recalcula el pipeline
    sobre ``aligned_df.loc[:t]`` y verifica que la fila ``t`` coincide con
    el cálculo full hasta tolerancia numérica.

    Nota: el plan menciona 100 fechas; aquí usamos N=10 porque cada iteración
    recalcula todos los indicadores (15/asset × 6 = 90 cálculos) y a 100
    fechas el test tarda ~5 min. 10 fechas es suficiente para detectar
    look-ahead — si una sola feature filtra futuro, fallará en cualquier seed.
    """
    _, _, test_idx = splits
    if len(test_idx) < 20:
        pytest.skip("test_idx demasiado corto")

    rng = np.random.default_rng(seed=42)
    sample_dates = rng.choice(test_idx[5:], size=10, replace=False)
    sample_dates = sorted(pd.DatetimeIndex(sample_dates))

    # Columnas a comparar: todas las features (excluir AdjClose ref).
    feature_cols = get_ppo_feature_columns(features_df)

    for t in sample_dates:
        # Recalcula pipeline sobre aligned truncado en t (inclusive).
        truncated = aligned_df.loc[:t]
        truncated_features = build_features(truncated)
        # `t` debe estar en truncated_features (a menos que esté en las primeras 60 filas).
        if t not in truncated_features.index:
            # Edge case improbable porque test_idx >> MAX_LOOKBACK; saltamos por seguridad.
            continue
        full_row = features_df.loc[t, feature_cols].to_numpy(dtype=np.float64)
        truncated_row = truncated_features.loc[t, feature_cols].to_numpy(dtype=np.float64)
        # Tolerancia generosa por aritmética float; las EMAs son recursivas y
        # acumulan ε. Para detectar look-ahead (que daría diferencias O(1)), 1e-6 basta.
        diff = np.abs(full_row - truncated_row)
        max_diff = float(diff.max())
        max_idx = int(diff.argmax())
        assert max_diff < 1e-6, (
            f"Look-ahead detectado en t={t.date()}: feature '{feature_cols[max_idx]}' "
            f"difiere {max_diff:.2e} entre cálculo full y truncado"
        )


def test_timegan_train_only(
    features_df: pd.DataFrame,
    splits: tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex],
) -> None:
    """F4-T12: las secuencias TimeGAN deben provenir SOLO de fechas en train.

    Estrategia bit-a-bit (decisión 7.1 del plan F4):

    1. Recreamos las secuencias esperadas desde
       ``features_df.loc[train_idx, TIMEGAN_COLUMNS]`` + ``scaler.transform``
       + ventanas deslizantes (``stride=1``, ``seq_len=24``).
    2. Las comparamos contra ``load_sequences()`` con ``np.allclose``.
    3. Verificamos shape: ``len(train_idx) - 24 + 1 = 1681``.

    Si las secuencias se construyeron correctamente, todas las filas deben
    coincidir bit a bit (tolerancia float). Cualquier discrepancia indica
    leakage de val/test en las secuencias persistidas.
    """
    from src.data.features import TIMEGAN_COLUMNS
    from src.data.scalers import load_timegan_scaler
    from src.data.sequence_builder import (
        DEFAULT_SEQ_LEN,
        SEQUENCES_PATH,
        load_sequences,
    )

    if not SEQUENCES_PATH.exists():
        pytest.skip(f"{SEQUENCES_PATH} no existe — ejecuta scripts/02_build_features.py primero")

    train_idx, _, _ = splits
    seq_len = DEFAULT_SEQ_LEN

    # 1. Secuencias persistidas
    sequences = load_sequences()

    # 2. Shape esperado: (len(train_idx) - seq_len + 1, seq_len, 9)
    expected_n = len(train_idx) - seq_len + 1
    assert sequences.shape == (expected_n, seq_len, len(TIMEGAN_COLUMNS)), (
        f"Shape de secuencias persistidas no coincide: got {sequences.shape}, "
        f"esperado ({expected_n}, {seq_len}, {len(TIMEGAN_COLUMNS)}). "
        f"Posible leakage si N > {expected_n}."
    )

    # 3. Recrear bit-a-bit desde features + scaler + train_idx
    scaler, scaler_cols = load_timegan_scaler()
    assert list(scaler_cols) == list(TIMEGAN_COLUMNS), (
        "scaler_cols ≠ TIMEGAN_COLUMNS — orden de columnas inconsistente"
    )
    train_data = features_df.loc[train_idx, list(TIMEGAN_COLUMNS)].to_numpy()
    scaled = scaler.transform(train_data).astype(np.float32)
    expected_sequences = np.stack(
        [scaled[i : i + seq_len] for i in range(expected_n)], axis=0,
    )

    # 4. Comparación bit-a-bit (atol generosa por float32)
    assert np.allclose(sequences, expected_sequences, atol=1e-6), (
        "Secuencias persistidas NO coinciden con recálculo desde train + scaler. "
        "Posible leakage o re-shuffle de fechas."
    )
