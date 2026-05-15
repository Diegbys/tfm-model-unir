"""scripts/03_train_timegan.py — Fase 4: entrenamiento + generación + dataset sintético.

Orquesta F4-T2 → F4-T7 + F4-T11 (stylized facts comparativa) por seed.
F4-T13 (multirun 3 seeds) se ejecuta con ``--multirun seed=0,42,123``.

Uso:
    # Smoke test (~15 min en M4 Pro): 1 seed con iteraciones reducidas
    uv run python scripts/03_train_timegan.py timegan/training=smoke seed=42

    # Single seed full
    uv run python scripts/03_train_timegan.py seed=42

    # Multirun 3 seeds (F4-T13, ~3-6h overnight en M4 Pro)
    uv run python scripts/03_train_timegan.py --multirun seed=0,42,123

    # Force retrain aunque artefactos existan
    uv run python scripts/03_train_timegan.py seed=0 force=true

    # Override device (e.g., para Kaggle)
    uv run python scripts/03_train_timegan.py --multirun seed=0,42,123 timegan.device=cuda

Inputs:
    data/processed/timegan_train_sequences.npy
    data/processed/features.parquet
    data/processed/aligned.parquet
    data/splits/train_idx.parquet
    models/scalers/timegan_minmax.joblib

Outputs (por seed):
    models/timegan/run_<seed>/best.pt
    models/timegan/run_<seed>/train_log.json
    data/synthetic/run_<seed>/synthetic_returns.npy
    data/synthetic/run_<seed>/synthetic_returns_scaled.npy
    data/synthetic/run_<seed>/synthetic_dataset.parquet
    data/synthetic/run_<seed>/stylized_facts_compare.csv
    data/synthetic/run_<seed>/metadata.json
    mlruns/<exp_id>/<run_id>/
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path

# Permitir imports del proyecto cuando Hydra cambia cwd (cfg.hydra.job.chdir=false).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

from src.data.alignment import PROCESSED_PATH  # noqa: E402
from src.data.download import PROJECT_ROOT  # noqa: E402
from src.data.scalers import load_timegan_scaler  # noqa: E402
from src.data.sequence_builder import load_sequences  # noqa: E402
from src.data.splits import load_splits  # noqa: E402
from src.generative.timegan.build_synthetic_dataset import (  # noqa: E402
    build_synthetic_dataset,
    persist_synthetic_dataset,
)
from src.generative.timegan.generate import generate_synthetic  # noqa: E402
from src.generative.timegan.reconstruct_prices import compute_aligned_train_stats  # noqa: E402
from src.generative.timegan.stylized_facts_compare import (  # noqa: E402
    compute_synthetic_stylized_facts,
)
from src.generative.timegan.train import train_timegan  # noqa: E402

logger = logging.getLogger("F4-T13")

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MLRUNS_PATH = PROJECT_ROOT / "mlruns"


def _artifacts_exist_for_seed(seed: int) -> bool:
    """Idempotencia: si todos los artefactos del seed existen, podemos saltar."""
    paths = [
        PROJECT_ROOT / f"models/timegan/run_{seed}/best.pt",
        PROJECT_ROOT / f"data/synthetic/run_{seed}/synthetic_returns.npy",
        PROJECT_ROOT / f"data/synthetic/run_{seed}/synthetic_returns_scaled.npy",
        PROJECT_ROOT / f"data/synthetic/run_{seed}/synthetic_dataset.parquet",
        PROJECT_ROOT / f"data/synthetic/run_{seed}/stylized_facts_compare.csv",
        PROJECT_ROOT / f"data/synthetic/run_{seed}/metadata.json",
    ]
    return all(p.exists() and p.stat().st_size > 0 for p in paths)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    seed = int(cfg.seed)
    force = bool(cfg.get("force", False))

    logger.info("=== Fase 4 — seed=%d ===", seed)
    logger.info("Config resuelto:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    if not force and _artifacts_exist_for_seed(seed):
        logger.info("Artefactos del seed=%d ya existen; salto (--force para sobreescribir)", seed)
        return 0

    # --- Cargas ---
    logger.info("Cargando inputs...")
    sequences = load_sequences()
    features_df = pd.read_parquet(FEATURES_PATH)
    aligned_df = pd.read_parquet(PROCESSED_PATH)
    train_idx, _, _ = load_splits()
    scaler, scaler_cols = load_timegan_scaler()
    logger.info(
        "  sequences=%s, features=%s, aligned=%s, train_idx=%d",
        sequences.shape, features_df.shape, aligned_df.shape, len(train_idx),
    )

    # --- MLflow ---
    MLRUNS_PATH.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file://{MLRUNS_PATH}")
    mlflow.set_experiment("timegan_phase4")
    run_name = f"seed_{seed}_{dt.datetime.now():%Y%m%d_%H%M%S}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "seed": seed,
            **_flatten_for_mlflow("timegan", OmegaConf.to_container(cfg.timegan, resolve=True)),
        })
        logger.info("MLflow run: %s (id=%s)", run_name, run.info.run_id)

        # --- F4-T2 + F4-T4: entrenamiento ---
        result = train_timegan(sequences, cfg, seed=seed, mlflow_logger=mlflow)
        mlflow.log_metric("best_disc_score", result.best_disc_score)
        mlflow.log_metric("best_iter", result.best_iter)
        mlflow.log_metric("seconds_total", result.seconds_total)
        mlflow.log_metric("early_stopped", float(result.early_stopped))
        mlflow.log_artifact(result.checkpoint_path)

        # --- F4-T5: generación ---
        logger.info("Generando sintéticos (K=%d)...", cfg.timegan.generation.K)
        synth_scaled, synth_original = generate_synthetic(
            Path(result.checkpoint_path), cfg, n_real=len(sequences), seed=seed,
        )

        output_dir = PROJECT_ROOT / f"data/synthetic/run_{seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "synthetic_returns_scaled.npy", synth_scaled)
        np.save(output_dir / "synthetic_returns.npy", synth_original)
        mlflow.log_artifact(str(output_dir / "synthetic_returns.npy"))

        # --- F4-T6 + F4-T7: reconstrucción + dataset ---
        logger.info("Calculando stats de aligned/train y construyendo dataset sintético...")
        stats = compute_aligned_train_stats(aligned_df, train_idx)
        synthetic_dataset = build_synthetic_dataset(
            synth_original, aligned_df, train_idx, stats,
            initial_price=float(cfg.timegan.generation.initial_price), seed=seed,
        )
        persist_synthetic_dataset(synthetic_dataset, output_dir)
        mlflow.log_artifact(str(output_dir / "synthetic_dataset.parquet"))

        # --- F4-T11: stylized facts ---
        logger.info("Calculando stylized facts sintético vs real...")
        sf_table = compute_synthetic_stylized_facts(
            synth_original, features_df, train_idx, output_dir,
        )
        mlflow.log_artifact(str(output_dir / "stylized_facts_compare.csv"))

        # --- Metadata por run ---
        metadata = {
            "seed": seed,
            "K": int(cfg.timegan.generation.K),
            "n_real_sequences": int(len(sequences)),
            "n_synth_sequences": int(synth_original.shape[0]),
            "seq_len": int(cfg.timegan.seq_len),
            "n_features": int(cfg.timegan.n_features),
            "initial_price": float(cfg.timegan.generation.initial_price),
            "timegan_checkpoint": str(result.checkpoint_path),
            "best_iter": int(result.best_iter),
            "best_disc_score": float(result.best_disc_score),
            "seconds_total": float(result.seconds_total),
            "early_stopped": bool(result.early_stopped),
            "mlflow_run_id": run.info.run_id,
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        }
        with (output_dir / "metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)
        mlflow.log_artifact(str(output_dir / "metadata.json"))

        # Log summary table de stylized facts
        for (asset, source), row in sf_table.iterrows():
            for col, val in row.items():
                try:
                    mlflow.log_metric(f"sf_{source}_{asset}_{col}", float(val))
                except (ValueError, TypeError):
                    pass

        logger.info("=== seed=%d FIN (artefactos en %s) ===", seed, output_dir)

    return 0


def _flatten_for_mlflow(prefix: str, d: dict | list | object, sep: str = ".") -> dict:
    """Aplana un dict anidado para mlflow.log_params (acepta solo escalares)."""
    out: dict = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(_flatten_for_mlflow(f"{prefix}{sep}{k}", v, sep))
    elif isinstance(d, list):
        out[prefix] = str(d)
    else:
        out[prefix] = d
    return out


if __name__ == "__main__":
    sys.exit(main())
