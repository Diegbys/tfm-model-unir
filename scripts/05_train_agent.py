"""scripts/05_train_agent.py — Fase 6: entrenamiento de un agente PPO.

Orquesta F6-T5…T9 por (experimento, seed). F6-T10/T11 (5 seeds × Agente A/B) se
ejecutan con ``--multirun``.

Uso:
    # Smoke (~pocos min en CPU): 1 seed, timesteps reducidos
    uv run python scripts/05_train_agent.py experiment=agent_a ppo=smoke seed=42

    # Corrida completa de un seed
    uv run python scripts/05_train_agent.py experiment=agent_a seed=42

    # Multirun F6-T10/T11: 5 seeds × 2 agentes (10 corridas)
    uv run python scripts/05_train_agent.py -m experiment=agent_a,agent_b seed=0,1,42,123,1337

    # Force re-entrenar aunque model.zip ya exista
    uv run python scripts/05_train_agent.py experiment=agent_a seed=42 force=true

Inputs:
    data/processed/features.parquet
    data/splits/{train,val,test}_idx.parquet
    models/scalers/ppo_robust.joblib
    data/synthetic/run_42/synthetic_dataset.parquet   (solo Agente B)

Outputs (por experimento × seed):
    outputs/ppo_runs/<exp>/seed<seed>/{model,best_model}.zip
    outputs/ppo_runs/<exp>/seed<seed>/{vec_normalize.pkl,config.yaml,eval_history.json,progress.csv}
    mlruns/<exp_id>/<run_id>/
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Permitir imports del proyecto cuando Hydra compone (cfg.hydra.job.chdir=false).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from src.agents.ppo_trainer import train_ppo_agent  # noqa: E402

logger = logging.getLogger("F6")


@hydra.main(config_path="../configs", config_name="train_ppo", version_base=None)
def main(cfg: DictConfig) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    model = train_ppo_agent(cfg)

    # F6-T9: el Agente B ve un dataset mayor pero debe consumir los MISMOS
    # total_timesteps que el A. PPO sobrepasa el objetivo en < n_steps por rollout;
    # la tolerancia admite esa holgura además del ±2% del ADR.
    achieved = int(model.num_timesteps)
    target = int(cfg.ppo.total_timesteps)
    tol = max(0.02 * target, int(cfg.ppo.n_steps))
    if abs(achieved - target) > tol:
        logger.warning(
            "F6-T9: timesteps entrenados (%d) se desvían del objetivo (%d) más de la tolerancia",
            achieved, target,
        )
    else:
        logger.info("F6-T9 OK: timesteps entrenados=%d ≈ objetivo=%d", achieved, target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
