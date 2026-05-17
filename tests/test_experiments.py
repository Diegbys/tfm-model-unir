"""Tests de comparabilidad del experimento A vs B — F6-T12.

La validez de la comparativa del TFE depende de que el Agente A y el Agente B se
entrenen con TODO idéntico salvo el dataset (regla de oro, ADR §Fase 6). Estos
tests lo verifican sobre los ``config.yaml`` y ``eval_history.json`` que persiste
``scripts/05_train_agent.py``.

Hacen ``pytest.skip`` si aún no hay corridas de ambos agentes para un seed común
(ejecutar ``scripts/05_train_agent.py`` para ambos experimentos primero).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from src.data.download import PROJECT_ROOT

PPO_RUNS_DIR = PROJECT_ROOT / "outputs" / "ppo_runs"


def _seed_dirs(agent: str) -> dict[int, Path]:
    """Mapa ``seed -> carpeta`` de las corridas con ``config.yaml`` de un agente."""
    base = PPO_RUNS_DIR / agent
    if not base.is_dir():
        return {}
    return {
        int(d.name[4:]): d
        for d in base.iterdir()
        if d.is_dir() and d.name.startswith("seed") and (d / "config.yaml").exists()
    }


def _common_seed() -> tuple[int, Path, Path]:
    """Primer seed con corrida de agent_a y agent_b; ``pytest.skip`` si no hay."""
    dirs_a = _seed_dirs("agent_a")
    dirs_b = _seed_dirs("agent_b")
    common = sorted(set(dirs_a) & set(dirs_b))
    if not common:
        pytest.skip(
            "faltan corridas de agent_a y agent_b para un seed común — "
            "ejecuta scripts/05_train_agent.py para ambos experimentos"
        )
    seed = common[0]
    return seed, dirs_a[seed], dirs_b[seed]


def test_a_vs_b_comparable() -> None:
    """F6-T12: A y B comparten ppo/env/seed; difieren solo en experiment (el dataset)."""
    seed, dir_a, dir_b = _common_seed()
    cfg_a = OmegaConf.load(dir_a / "config.yaml")
    cfg_b = OmegaConf.load(dir_b / "config.yaml")

    assert cfg_a.seed == cfg_b.seed == seed, "los seeds comparados no coinciden"
    assert OmegaConf.to_container(cfg_a.ppo) == OmegaConf.to_container(cfg_b.ppo), (
        "los hiperparámetros PPO de A y B no coinciden — la comparativa no es válida"
    )
    assert OmegaConf.to_container(cfg_a.env) == OmegaConf.to_container(cfg_b.env), (
        "la config del entorno de A y B no coincide"
    )

    # El grupo `experiment` es lo ÚNICO que puede cambiar entre A y B.
    assert cfg_a.experiment.name == "agent_a"
    assert cfg_b.experiment.name == "agent_b"
    assert cfg_a.experiment.synthetic_ratio == 0.0, "Agente A no debe usar sintéticos"
    assert cfg_b.experiment.synthetic_ratio > 0.0, "Agente B debe usar sintéticos"
    assert cfg_a.experiment.synthetic_dataset_path is None
    assert cfg_b.experiment.synthetic_dataset_path is not None


def test_a_vs_b_timesteps_igualados() -> None:
    """F6-T9: A y B consumen los mismos total_timesteps (tolerancia ±2% o un rollout)."""
    seed, dir_a, dir_b = _common_seed()
    cfg_a = OmegaConf.load(dir_a / "config.yaml")
    target = int(cfg_a.ppo.total_timesteps)
    tol = max(0.02 * target, int(cfg_a.ppo.n_steps))
    for agent, run_dir in (("agent_a", dir_a), ("agent_b", dir_b)):
        history = json.loads((run_dir / "eval_history.json").read_text())
        achieved = int(history["num_timesteps"])
        assert abs(achieved - target) <= tol, (
            f"{agent} seed={seed}: timesteps entrenados {achieved} se desvían del "
            f"objetivo {target} más de la tolerancia {tol:.0f}"
        )
