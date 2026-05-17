"""Entrenamiento de un agente PPO con tracking MLflow — F6-T5.

``train_ppo_agent`` orquesta una corrida completa: fija seeds, carga datos,
construye el ``MixedDataset``, los entornos de train/val, el modelo PPO y el
callback de evaluación; entrena ``total_timesteps`` steps y persiste modelo,
estadísticas de ``VecNormalize``, config e historial de evaluación, tanto en
``outputs/ppo_runs/`` como en MLflow.

Es **idéntico para el Agente A y el B** — la única diferencia la introduce
``cfg.experiment`` (``synthetic_ratio`` y el dataset sintético). Reusa el patrón
MLflow file-based de ``scripts/03_train_timegan.py``.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import mlflow
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.logger import KVWriter, configure
from stable_baselines3.common.vec_env import VecNormalize

from src.agents.callbacks import ValidationEvalCallback, evaluate_on_env
from src.agents.env_factory import make_eval_env, make_train_env
from src.data.download import PROJECT_ROOT
from src.data.mixed_dataset import MixedDataset
from src.data.splits import load_splits
from src.utils.seeding import set_global_seed

logger = logging.getLogger(__name__)

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MLRUNS_PATH = PROJECT_ROOT / "mlruns"
PPO_RUNS_DIR = PROJECT_ROOT / "outputs" / "ppo_runs"
MLFLOW_EXPERIMENT = "ppo_phase6"

_ACTIVATIONS = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}


def build_policy_kwargs(ppo_cfg) -> dict:
    """Convierte la sección ``policy_kwargs`` del YAML en kwargs para SB3.

    Resuelve ``activation_fn`` (string → clase ``torch.nn``) y deja ``net_arch``
    como dict plano. SB3 espera la clase de activación, no su nombre.
    """
    pk = OmegaConf.to_container(ppo_cfg.policy_kwargs, resolve=True)
    act = str(pk.pop("activation_fn", "tanh")).lower()
    if act not in _ACTIVATIONS:
        raise ValueError(f"activation_fn no soportada: {act} (usa {list(_ACTIVATIONS)})")
    pk["activation_fn"] = _ACTIVATIONS[act]
    return pk


def _flatten(prefix: str, value, sep: str = ".") -> dict:
    """Aplana un dict/lista anidado para ``mlflow.log_params`` (solo escalares)."""
    out: dict = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(_flatten(f"{prefix}{sep}{k}", v, sep))
    elif isinstance(value, list):
        out[prefix] = str(value)
    else:
        out[prefix] = value
    return out


class _MLflowWriter(KVWriter):
    """Output format de SB3 que vuelca cada métrica escalar a MLflow.

    Se añade al logger de SB3 para que ``ep_rew_mean``, ``policy_loss``,
    ``value_loss``, ``entropy_loss``, etc. queden en MLflow por step (F6-T5). Las
    métricas ``val/*`` se omiten: ya las loguea ``ValidationEvalCallback`` con
    nombre limpio (``val_sharpe``…) y el step exacto de la evaluación.
    """

    def write(self, key_values, key_excluded, step: int = 0) -> None:  # noqa: ARG002
        if mlflow.active_run() is None:
            return
        for key, value in key_values.items():
            if key.startswith("val/"):
                continue
            try:
                mlflow.log_metric(key.replace("/", "_"), float(value), step=step)
            except (TypeError, ValueError):
                pass  # métricas no escalares (e.g. texto): se ignoran

    def close(self) -> None:
        pass


def _artifacts_exist(output_dir: Path) -> bool:
    """Idempotencia: el modelo final ya está persistido."""
    return (output_dir / "model.zip").exists()


def train_ppo_agent(cfg: DictConfig, output_dir: Path | None = None) -> PPO:
    """Entrena un agente PPO según la config Hydra y persiste artefactos + MLflow.

    Parameters
    ----------
    cfg:
        Config compuesta de ``configs/train_ppo.yaml`` (grupos ``ppo``, ``env``,
        ``experiment`` + ``seed``, ``force``).
    output_dir:
        Carpeta de artefactos. Por defecto ``outputs/ppo_runs/<experiment>/seed<seed>``.

    Returns
    -------
    PPO
        El modelo entrenado (también persistido como ``model.zip``).
    """
    seed = int(cfg.seed)
    exp_name = str(cfg.experiment.name)
    if output_dir is None:
        output_dir = PPO_RUNS_DIR / exp_name / f"seed{seed}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not bool(cfg.get("force", False)) and _artifacts_exist(output_dir):
        logger.info(
            "model.zip ya existe en %s; salto (force=true para sobreescribir)", output_dir
        )
        return PPO.load(output_dir / "model.zip")

    logger.info("=== F6: entrenando %s seed=%d ===", exp_name, seed)
    logger.info("Config resuelto:\n%s", OmegaConf.to_yaml(cfg, resolve=True))
    set_global_seed(seed)

    # --- datos ---
    features_df = pd.read_parquet(FEATURES_PATH)
    train_idx, val_idx, _ = load_splits()
    synthetic_df = None
    if cfg.experiment.synthetic_dataset_path is not None:
        synthetic_df = pd.read_parquet(
            PROJECT_ROOT / str(cfg.experiment.synthetic_dataset_path)
        )

    mixed = MixedDataset(
        features_df, train_idx, synthetic_df,
        synthetic_ratio=float(cfg.experiment.synthetic_ratio), seed=seed,
    )

    # --- entornos: train episódico (mezclador) + eval recorrido-completo (val real) ---
    train_env = make_train_env(
        mixed, cfg.env, seed=seed,
        use_vecnormalize=bool(cfg.ppo.use_vecnormalize), gamma=float(cfg.ppo.gamma),
    )
    eval_env = make_eval_env(features_df, val_idx, cfg.env)

    # --- modelo PPO (hiperparámetros literales del ADR §3.4) ---
    model = PPO(
        "MlpPolicy", train_env,
        learning_rate=float(cfg.ppo.learning_rate),
        n_steps=int(cfg.ppo.n_steps),
        batch_size=int(cfg.ppo.batch_size),
        n_epochs=int(cfg.ppo.n_epochs),
        gamma=float(cfg.ppo.gamma),
        gae_lambda=float(cfg.ppo.gae_lambda),
        clip_range=float(cfg.ppo.clip_range),
        ent_coef=float(cfg.ppo.ent_coef),
        vf_coef=float(cfg.ppo.vf_coef),
        max_grad_norm=float(cfg.ppo.max_grad_norm),
        policy_kwargs=build_policy_kwargs(cfg.ppo),
        seed=seed,
        device=str(cfg.ppo.device),
        verbose=1,
    )

    # --- MLflow file-based (mismo patrón que F4) ---
    MLRUNS_PATH.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file://{MLRUNS_PATH}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    run_name = f"{exp_name}_seed{seed}_{dt.datetime.now():%Y%m%d_%H%M%S}"

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "seed": seed,
            "experiment_name": exp_name,
            **_flatten("ppo", OmegaConf.to_container(cfg.ppo, resolve=True)),
            **_flatten("env", OmegaConf.to_container(cfg.env, resolve=True)),
            **_flatten("experiment", OmegaConf.to_container(cfg.experiment, resolve=True)),
        })
        logger.info("MLflow run: %s (id=%s)", run_name, run.info.run_id)

        # Logger SB3 → stdout + progress.csv + MLflow.
        sb3_logger = configure(str(output_dir), ["stdout", "csv"])
        sb3_logger.output_formats.append(_MLflowWriter())
        model.set_logger(sb3_logger)

        # Callback de evaluación periódica sobre val (F6-T6).
        eval_cb = ValidationEvalCallback(
            eval_env,
            eval_freq=int(cfg.ppo.eval_freq),
            best_model_path=output_dir / "best_model",
            verbose=1,
        )
        model.learn(total_timesteps=int(cfg.ppo.total_timesteps), callback=eval_cb)

        # --- persistencia de artefactos ---
        model.save(output_dir / "model.zip")
        if isinstance(train_env, VecNormalize):
            train_env.save(str(output_dir / "vec_normalize.pkl"))
        (output_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

        final_metrics = evaluate_on_env(model, eval_env)
        (output_dir / "eval_history.json").write_text(json.dumps(
            {
                "experiment": exp_name,
                "seed": seed,
                "num_timesteps": int(model.num_timesteps),  # F6-T9: igualación A/B
                "evaluations": eval_cb.evaluations,
                "final": final_metrics,
                "best_val_sharpe": eval_cb.best_sharpe,
            },
            indent=2,
        ))
        for key, value in final_metrics.items():
            mlflow.log_metric(f"final_{key}", value)
        mlflow.log_metric("best_val_sharpe", float(eval_cb.best_sharpe))

        for name in (
            "model.zip", "best_model.zip", "vec_normalize.pkl",
            "config.yaml", "eval_history.json", "progress.csv",
        ):
            path = output_dir / name
            if path.exists():
                mlflow.log_artifact(str(path))

        logger.info(
            "=== %s seed=%d FIN — best_val_sharpe=%.4f, artefactos en %s ===",
            exp_name, seed, eval_cb.best_sharpe, output_dir,
        )

    train_env.close()
    return model
