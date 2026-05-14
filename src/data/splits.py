"""Split cronológico train/val/test — F3-T5.

Persiste los **índices** (no los datos) para que todos los módulos posteriores
(scalers, builders, agentes) lean exactamente el mismo split sin riesgo de
desincronización. La regla del ADR es 70/10/20 cronológico:

- **Train**: 2015-01-01 → 2021-12-31 (incluye crash COVID 2020-03).
- **Val**:   2022-01-01 → 2022-12-31 (sell-off por rates shock).
- **Test**:  2023-01-01 → 2025-04-30 (INTOCABLE hasta la evaluación final).

El split se aplica sobre el dataframe de features ya con las primeras filas
de calentamiento descartadas (ver `src.data.features.build_features`), por
lo que la longitud total es ≈2540 filas, no 2600.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.data.download import PROJECT_ROOT

logger = logging.getLogger(__name__)

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

# Bordes inclusivos siguiendo compass §4.1 y ADR F3-T5.
TRAIN_START = pd.Timestamp("2015-01-01")
TRAIN_END = pd.Timestamp("2021-12-31")
VAL_START = pd.Timestamp("2022-01-01")
VAL_END = pd.Timestamp("2022-12-31")
TEST_START = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2025-04-30")

# Sanity check: 2020-03-23 fue el mínimo intradía del S&P 500 durante COVID.
# Si tras el descarte de calentamiento no está en train, algo se rompió.
COVID_BOTTOM = pd.Timestamp("2020-03-23")


def chronological_split(
    df: pd.DataFrame,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Calcula los 3 índices (train/val/test) a partir del índice de ``df``.

    No persiste; usar :func:`persist_splits` para escribir a disco. Valida
    sin solapamiento y que la suma cubre todo el dataframe.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"chronological_split: índice debe ser DatetimeIndex, got {type(df.index)}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("chronological_split: índice no monotónicamente creciente")

    idx = df.index
    train_idx = idx[(idx >= TRAIN_START) & (idx <= TRAIN_END)]
    val_idx = idx[(idx >= VAL_START) & (idx <= VAL_END)]
    test_idx = idx[(idx >= TEST_START) & (idx <= TEST_END)]

    n_total = len(idx)
    n_split = len(train_idx) + len(val_idx) + len(test_idx)
    if n_split != n_total:
        raise ValueError(
            f"chronological_split: suma de splits {n_split} != longitud total {n_total}. "
            f"Probable causa: el índice contiene fechas fuera de [{TRAIN_START.date()}, "
            f"{TEST_END.date()}]."
        )
    if COVID_BOTTOM not in train_idx:
        # COVID_BOTTOM puede no estar si fue eliminado en la alineación, pero su
        # ausencia merece un WARNING porque cambia el carácter del train.
        logger.warning(
            "chronological_split: %s (mínimo COVID) no presente en train", COVID_BOTTOM.date()
        )

    logger.info(
        "Split cronológico: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(train_idx), 100 * len(train_idx) / n_total,
        len(val_idx), 100 * len(val_idx) / n_total,
        len(test_idx), 100 * len(test_idx) / n_total,
    )
    return train_idx, val_idx, test_idx


def persist_splits(
    train_idx: pd.DatetimeIndex,
    val_idx: pd.DatetimeIndex,
    test_idx: pd.DatetimeIndex,
    out_dir: Path = SPLITS_DIR,
) -> dict[str, Path]:
    """Persiste cada índice como Parquet single-column en ``out_dir``.

    Devuelve mapping ``{split_name: path}``. El parquet contiene una sola
    columna ``date`` para que cualquier consumidor haga
    ``pd.read_parquet(path).set_index('date').index``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, idx in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        path = out_dir / f"{name}_idx.parquet"
        pd.DataFrame({"date": idx}).to_parquet(path, index=False)
        logger.info("Persistido %s_idx (%d fechas) en %s", name, len(idx), path)
        paths[name] = path
    return paths


def load_splits(
    in_dir: Path = SPLITS_DIR,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Carga los 3 índices persistidos por :func:`persist_splits`."""
    out: list[pd.DatetimeIndex] = []
    for name in ("train", "val", "test"):
        path = in_dir / f"{name}_idx.parquet"
        if not path.exists():
            raise FileNotFoundError(f"load_splits: falta {path}")
        df = pd.read_parquet(path)
        out.append(pd.DatetimeIndex(df["date"]))
    return out[0], out[1], out[2]
