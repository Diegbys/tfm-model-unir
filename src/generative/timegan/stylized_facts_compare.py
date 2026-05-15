"""Comparativa de stylized facts real vs sintético — F4-T11.

Wrapper FINO sobre :func:`src.eval.stylized_facts.compare_real_vs_synthetic`
(ya implementado en Fase 2 con este uso en mente). Aquí solo:

1. Extraemos los log-returns reales del DataFrame de features de Fase 3.
2. Aplanamos las secuencias sintéticas a un DataFrame columnar.
3. Llamamos a la función existente.
4. Persistimos el CSV.

NO reimplementa kurtosis/ACF/leverage — vive en :mod:`src.eval.stylized_facts`.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.features import TIMEGAN_COLUMNS
from src.eval.stylized_facts import compare_real_vs_synthetic

logger = logging.getLogger(__name__)

# Columnas TimeGAN que son log-returns de equity (las 6 primeras de TIMEGAN_COLUMNS).
EQUITY_RETURN_COLS = [c for c in TIMEGAN_COLUMNS if c.endswith("_log_ret")]


def _flatten_returns(seqs: np.ndarray) -> pd.DataFrame:
    """De ``(N_seq, T, 9)`` a DataFrame ``(N_seq*T, 6)`` con las 6 cols equity.

    Las primeras 6 columnas de la última dim son las equity log-returns
    (orden TIMEGAN_COLUMNS). Flatten preserva la independencia entre
    secuencias sintéticas; los stylized facts son agregados estadísticos.
    """
    if seqs.ndim != 3:
        raise ValueError(f"_flatten_returns: shape (N, T, F) esperado; got {seqs.shape}")
    n_seq, T, n_feat = seqs.shape
    if n_feat < len(EQUITY_RETURN_COLS):
        raise ValueError(
            f"_flatten_returns: esperaba ≥{len(EQUITY_RETURN_COLS)} dims, got {n_feat}"
        )
    flat = seqs[:, :, : len(EQUITY_RETURN_COLS)].reshape(n_seq * T, len(EQUITY_RETURN_COLS))
    return pd.DataFrame(flat, columns=EQUITY_RETURN_COLS)


def compute_synthetic_stylized_facts(
    synthetic_returns_original: np.ndarray,
    real_features_df: pd.DataFrame,
    train_idx: pd.DatetimeIndex,
    output_dir: Path,
) -> pd.DataFrame:
    """Compara stylized facts real vs sintético, persiste tabla.

    Parameters
    ----------
    synthetic_returns_original:
        Array ``(N_seq, T, 9)`` con log-returns en espacio original.
    real_features_df:
        ``features.parquet`` cargado.
    train_idx:
        Solo se usan los retornos reales en train (anti-leakage).
    output_dir:
        Directorio para persistir ``stylized_facts_compare.csv``.

    Returns
    -------
    DataFrame
        MultiIndex ``(asset, source)`` con columnas:
        ``kurtosis, skew, shapiro_pvalue, acf_ret_lag1, acf_abs_ret_lag1,
        leverage_effect_k5``.
    """
    real_returns = real_features_df.loc[train_idx, EQUITY_RETURN_COLS].copy()
    synth_returns = _flatten_returns(synthetic_returns_original)

    # Renombrar para que ambos tengan los mismos nombres de columnas.
    # compare_real_vs_synthetic exige columnas idénticas; ya lo son.
    logger.info(
        "compute_synthetic_stylized_facts: real=%d filas, synth=%d filas, 6 activos",
        len(real_returns), len(synth_returns),
    )
    table = compare_real_vs_synthetic(real_returns, synth_returns)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "stylized_facts_compare.csv"
    table.to_csv(csv_path)
    logger.info("stylized_facts_compare.csv persistido en %s", csv_path)
    return table
