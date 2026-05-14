"""scripts/02_build_features.py — orquesta F3-T1 → F3-T10 + summary doc.

Uso:
    uv run python scripts/02_build_features.py
    uv run python scripts/02_build_features.py --force

Lee:
    data/processed/aligned.parquet  (output de Fase 1)

Escribe:
    data/processed/features.parquet                 (139 features + 6 AdjClose)
    data/processed/timegan_train_sequences.npy      (N, 24, 9)
    data/splits/{train,val,test}_idx.parquet
    models/scalers/timegan_minmax.joblib + timegan_columns.json
    models/scalers/ppo_robust.joblib + ppo_columns.json
    outputs/preproc/preproc_summary.md

Idempotente: si los artefactos ya existen, no recalcula (usa --force para sobreescribir).
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.alignment import PROCESSED_PATH  # noqa: E402
from src.data.download import PROJECT_ROOT  # noqa: E402
from src.data.features import (  # noqa: E402
    MAX_LOOKBACK,
    TIMEGAN_COLUMNS,
    build_features,
    get_ppo_feature_columns,
)
from src.data.scalers import (  # noqa: E402
    PPO_COLUMNS_PATH,
    PPO_SCALER_PATH,
    TIMEGAN_COLUMNS_PATH,
    TIMEGAN_SCALER_PATH,
    fit_ppo_scaler,
    fit_timegan_scaler,
)
from src.data.sequence_builder import (  # noqa: E402
    DEFAULT_SEQ_LEN,
    SEQUENCES_PATH,
    build_timegan_sequences,
    persist_sequences,
)
from src.data.splits import (  # noqa: E402
    SPLITS_DIR,
    chronological_split,
    persist_splits,
)

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
PREPROC_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "preproc" / "preproc_summary.md"


def _all_artifacts_exist() -> bool:
    paths = [
        FEATURES_PATH,
        SEQUENCES_PATH,
        SPLITS_DIR / "train_idx.parquet",
        SPLITS_DIR / "val_idx.parquet",
        SPLITS_DIR / "test_idx.parquet",
        TIMEGAN_SCALER_PATH,
        TIMEGAN_COLUMNS_PATH,
        PPO_SCALER_PATH,
        PPO_COLUMNS_PATH,
    ]
    return all(p.exists() and p.stat().st_size > 0 for p in paths)


def _write_summary(
    aligned_df: pd.DataFrame,
    features_df: pd.DataFrame,
    train_idx: pd.DatetimeIndex,
    val_idx: pd.DatetimeIndex,
    test_idx: pd.DatetimeIndex,
    sequences: np.ndarray,
    adf_results: dict[str, tuple[float, float]],
) -> None:
    """Genera `outputs/preproc/preproc_summary.md` con métricas de la fase."""
    n_dropped = len(aligned_df) - len(features_df)
    pct_dropped = 100.0 * n_dropped / len(aligned_df)
    n_features = features_df.shape[1] - 6  # -6 AdjClose de referencia
    n_ppo_features = len(get_ppo_feature_columns(features_df))

    adf_lines = []
    for col, (stat, pval) in adf_results.items():
        verdict = "✓ estacionaria" if pval < 0.05 else "✗ NO rechaza H0"
        adf_lines.append(f"- `{col}`: ADF stat={stat:.3f}, p={pval:.4f} {verdict}")

    content = f"""# Resumen Fase 3 — Preprocesamiento e ingeniería de features

Generado: {dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}

Insumo directo para el Cap. 5 sección "Fase 2: Preprocesamiento e ingeniería de características" de la memoria del TFE.

## 1. Calidad de datos

- **Filas en aligned.parquet** (Fase 1): {len(aligned_df)}
- **Filas tras `build_features`**: {len(features_df)}
- **Filas descartadas por NaN inicial**: {n_dropped} ({pct_dropped:.2f}%)
  - Causa: ventanas móviles de indicadores (EMA(60) domina con `MAX_LOOKBACK={MAX_LOOKBACK}`).

## 2. Shapes finales

- **features.parquet**: {features_df.shape[0]} filas × {features_df.shape[1]} columnas ({n_features} features + 6 AdjClose ref).
- **Columnas PPO** (entrada al RobustScaler): {n_ppo_features}.
- **Columnas TimeGAN**: {len(TIMEGAN_COLUMNS)} (6 log-returns + ΔVIX + ΔTNX_bps + DXY log_ret).
- **Splits**: train={len(train_idx)} ({100*len(train_idx)/len(features_df):.1f}%), val={len(val_idx)} ({100*len(val_idx)/len(features_df):.1f}%), test={len(test_idx)} ({100*len(test_idx)/len(features_df):.1f}%).
- **timegan_train_sequences.npy**: shape {tuple(sequences.shape)}, dtype {sequences.dtype}.
  - Rango de valores: [{sequences.min():.4f}, {sequences.max():.4f}] (esperado [0,1]).

## 3. Estacionariedad ADF (3 macros returns-like)

{chr(10).join(adf_lines)}

`VIX_log` no se testea (es nivel deliberadamente no estacionario, sirve como "régimen" lento). El RobustScaler atenuará su escala.

## 4. Distribución post-scaler

| Scaler | Cols | Min train | Max train | Mediana train |
|---|---|---|---|---|
| MinMax (TimeGAN, train) | {len(TIMEGAN_COLUMNS)} | 0.00 | 1.00 | ~0.5 (depende del activo) |
| Robust (PPO, train) | {n_ppo_features} | — | — | ≈0 (centrado en mediana) |

**Distribution shift esperada en val/test**: por construcción los scalers solo conocen train. Las EMAs y OBV (no estacionarios) excederán claramente el rango de train en test 2023-2025 (NVDA × ~10×). Esto es **aceptado por el ADR**: el PPO debe aprender a robustecerse frente a shift. La verificación cuantitativa se hace en `tests/test_no_leakage.py::test_scaler_only_train`.

## 5. Trade-offs documentados

- **EMAs y OBV crudos** (vs `Close - EMA` o `OBV.diff()`): siguiendo el ADR literal. El RobustScaler basa la magnitud en IQR de train; en test los valores estarán fuera del IQR pero el ordering relativo se conserva. *Future work*: experimentar con versiones relativas y comparar Sharpe del agente.
- **Librería `ta` (Bukosabino)** en lugar de `pandas-ta` (compass §5.4): pandas-ta 0.4.x exige Python>=3.12; el proyecto está pinneado a 3.11. `ta` cubre todos los indicadores del core set y se mantiene activamente.
- **EMA(60) añadida** al ADR original (EMA(10) + EMA(30)): alineación con compass §1.1 multi-horizonte. Aumenta `MAX_LOOKBACK` de 30 a 60 → 60 filas descartadas en lugar de ~30.
- **Stride=1** en secuencias TimeGAN: overlapping (estándar paper Yoon 2019). Genera correlación entre muestras consecutivas; tenerlo en cuenta al evaluar `discriminative_score` en Fase 4.

## 6. Artefactos persistidos (referencia)

- `data/processed/features.parquet`
- `data/processed/timegan_train_sequences.npy`
- `data/splits/{{train,val,test}}_idx.parquet`
- `models/scalers/{{timegan_minmax,ppo_robust}}.joblib` + `*_columns.json`

---

Estado de la Fase 3: **completada**. Próxima: Fase 4 (TimeGAN multivariado).
"""
    PREPROC_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREPROC_SUMMARY_PATH.write_text(content)
    logging.getLogger("F3-T12").info("Summary generado en %s", PREPROC_SUMMARY_PATH)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recalcula aunque los artefactos ya existan",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    log = logging.getLogger("F3-T12")
    log.info("Inicio build_features — %s UTC", dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))

    if not args.force and _all_artifacts_exist():
        log.info("Todos los artefactos ya existen, salto (--force para sobreescribir)")
        return 0

    if not PROCESSED_PATH.exists():
        log.error("No existe %s; ejecuta scripts/01_download_data.py primero", PROCESSED_PATH)
        return 1

    log.info("Cargando dataset alineado de %s", PROCESSED_PATH)
    aligned_df = pd.read_parquet(PROCESSED_PATH)

    # --- F3-T1 a F3-T4: features ---
    # Capturamos los ADF en un dict capturando warnings/info del logger de features.
    adf_results: dict[str, tuple[float, float]] = {}

    class _ADFCapture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            msg = record.getMessage()
            if msg.startswith("ADF "):
                # formato "ADF {col} ✓/✗ ...(stat=..., p=...)"
                try:
                    col = msg.split(" ", 2)[1]
                    stat = float(msg.split("stat=")[1].split(",")[0])
                    pval = float(msg.split("p=")[1].rstrip(")"))
                    adf_results[col] = (stat, pval)
                except (IndexError, ValueError):
                    pass

    features_logger = logging.getLogger("src.data.features")
    capture = _ADFCapture()
    features_logger.addHandler(capture)
    try:
        features_df = build_features(aligned_df)
    finally:
        features_logger.removeHandler(capture)

    log.info("Persistiendo features en %s", FEATURES_PATH)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_parquet(FEATURES_PATH)

    # --- F3-T5: split cronológico ---
    train_idx, val_idx, test_idx = chronological_split(features_df)
    persist_splits(train_idx, val_idx, test_idx)

    # --- F3-T6, F3-T7: scalers ---
    timegan_scaler, timegan_cols = fit_timegan_scaler(features_df, train_idx)
    _ppo_scaler, _ppo_cols = fit_ppo_scaler(features_df, train_idx)

    # --- F3-T10: secuencias TimeGAN ---
    sequences = build_timegan_sequences(
        features_df, train_idx, timegan_scaler, timegan_cols, seq_len=DEFAULT_SEQ_LEN
    )
    persist_sequences(sequences)

    # --- Summary doc ---
    _write_summary(aligned_df, features_df, train_idx, val_idx, test_idx, sequences, adf_results)

    log.info(
        "Fin OK — features %s, secuencias %s, splits %d/%d/%d",
        features_df.shape, sequences.shape, len(train_idx), len(val_idx), len(test_idx),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
