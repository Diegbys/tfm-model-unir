"""scripts/04_evaluate_synthetic.py — Fase 4: gate de calidad F4-T14.

Para cada seed:

1. Carga ``synthetic_returns_scaled.npy`` + ``load_sequences()`` (reales).
2. Calcula ``discriminative_score`` y ``predictive_score`` (config eval del ADR).
3. Genera ``tsne.png``.
4. Lee ``stylized_facts_compare.csv`` ya persistido por ``scripts/03_*.py``.
5. **Gate F4-T14**:
   - ``discriminative_score['mean'] ≤ cfg.eval.gate.max_discriminative_score`` (0.30)
   - ``predictive_score['gap'] ≤ cfg.eval.gate.max_predictive_gap`` (0.25)
   - ``acf_abs_synth / acf_abs_real ≥ cfg.eval.gate.min_acf_abs_ratio`` (0.5)
6. Si PASS: ``touch data/synthetic/run_<seed>/QUALITY_OK.flag``.
7. Si FAIL: log detallado con qué thresholds fallaron.

Tras todos los seeds (multirun): escribe ``outputs/timegan/timegan_summary.md``
consolidado (decisión 8.1 del plan F4).

Uso:
    uv run python scripts/04_evaluate_synthetic.py seed=42
    uv run python scripts/04_evaluate_synthetic.py --multirun seed=0,42,123

Exit code: 0 si TODOS los seeds del run pasan; 1 si alguno falla.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hydra  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from src.data.download import PROJECT_ROOT  # noqa: E402
from src.data.sequence_builder import load_sequences  # noqa: E402
from src.generative.timegan.metrics import (  # noqa: E402
    discriminative_score,
    predictive_score,
    tsne_plot,
)

logger = logging.getLogger("F4-T14")

MLRUNS_PATH = PROJECT_ROOT / "mlruns"
TIMEGAN_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "timegan"


def _scaled_real_sequences() -> np.ndarray:
    """Sequences reales en MinMax [0,1] (output de F3, ya escaladas)."""
    return load_sequences()


def evaluate_seed(cfg: DictConfig, seed: int) -> tuple[bool, dict]:
    """Evalúa un seed y devuelve ``(passed, report)``.

    ``report`` contiene todas las métricas + verdicts de cada threshold.
    Persiste métricas como JSON y figura t-SNE.
    """
    run_dir = PROJECT_ROOT / f"data/synthetic/run_{seed}"
    if not run_dir.exists():
        raise FileNotFoundError(
            f"evaluate_seed({seed}): falta {run_dir} — ejecuta scripts/03_*.py primero"
        )

    synth_scaled = np.load(run_dir / "synthetic_returns_scaled.npy")
    real_scaled = _scaled_real_sequences()
    sf_table = pd.read_csv(run_dir / "stylized_facts_compare.csv")

    eval_cfg = cfg.timegan.eval

    logger.info("[seed=%d] discriminative_score (n_repeats=%d, epochs=%d)",
                seed, eval_cfg.discriminative.n_repeats, eval_cfg.discriminative.epochs)
    disc = discriminative_score(
        real_seqs=real_scaled,
        synthetic_seqs=synth_scaled,
        n_repeats=int(eval_cfg.discriminative.n_repeats),
        test_size=float(eval_cfg.discriminative.test_size),
        epochs=int(eval_cfg.discriminative.epochs),
        batch_size=int(eval_cfg.discriminative.batch_size),
        lr=float(eval_cfg.discriminative.lr),
        hidden_dim=int(eval_cfg.discriminative.classifier_hidden),
        non_overlap_stride=int(eval_cfg.discriminative.non_overlap_stride),
        device=cfg.timegan.device,
        seed=seed,
    )
    logger.info("[seed=%d] discriminative: mean=%.4f std=%.4f", seed, disc["mean"], disc["std"])

    logger.info("[seed=%d] predictive_score (TSTR vs TRTR)", seed)
    pred = predictive_score(
        real_seqs=real_scaled,
        synthetic_seqs=synth_scaled,
        epochs=int(eval_cfg.predictive.epochs),
        batch_size=int(eval_cfg.predictive.batch_size),
        lr=float(eval_cfg.predictive.lr),
        hidden_dim=int(eval_cfg.predictive.forecaster_hidden),
        device=cfg.timegan.device,
        seed=seed,
    )
    logger.info("[seed=%d] predictive: gap=%.4f (tstr=%.6f, trtr=%.6f)",
                seed, pred["gap"], pred["mae_tstr"], pred["mae_trtr"])

    logger.info("[seed=%d] t-SNE plot...", seed)
    tsne_path = run_dir / "tsne.png"
    tsne_plot(
        real_seqs=real_scaled,
        synthetic_seqs=synth_scaled,
        output_path=tsne_path,
        n_samples=int(eval_cfg.tsne.n_samples),
        perplexity=int(eval_cfg.tsne.perplexity),
        random_state=int(eval_cfg.tsne.random_state),
    )

    # --- Gate F4-T14 ---
    gate = eval_cfg.gate
    # Ratio ACF|r| sintético / real, mínimo sobre los 6 activos
    sf_pivot = sf_table.pivot_table(index="asset", columns="source", values="acf_abs_ret_lag1")
    if not {"real", "synthetic"}.issubset(sf_pivot.columns):
        raise RuntimeError(
            f"stylized_facts_compare.csv malformado (source cols: {sf_pivot.columns.tolist()})"
        )
    # Ratio por activo. Si real ≈ 0, evitamos div por 0.
    ratios = sf_pivot["synthetic"] / sf_pivot["real"].replace(0, np.nan)
    min_acf_ratio = float(ratios.min())

    checks = {
        "discriminative_score_ok": disc["mean"] <= float(gate.max_discriminative_score),
        "predictive_gap_ok": pred["gap"] <= float(gate.max_predictive_gap),
        "acf_abs_ratio_ok": min_acf_ratio >= float(gate.min_acf_abs_ratio),
    }
    passed = all(checks.values())

    report = {
        "seed": seed,
        "discriminative_score": disc,
        "predictive_score": pred,
        "min_acf_abs_ratio": min_acf_ratio,
        "thresholds": {
            "max_discriminative_score": float(gate.max_discriminative_score),
            "max_predictive_gap": float(gate.max_predictive_gap),
            "min_acf_abs_ratio": float(gate.min_acf_abs_ratio),
        },
        "checks": checks,
        "passed": passed,
        "tsne_path": str(tsne_path),
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }

    # Persistir métricas
    with (run_dir / "metrics.json").open("w") as f:
        json.dump(report, f, indent=2)

    # Flag de calidad
    flag_path = run_dir / "QUALITY_OK.flag"
    if passed:
        flag_path.touch()
        logger.info("[seed=%d] ✅ GATE PASS → %s", seed, flag_path)
    else:
        if flag_path.exists():
            flag_path.unlink()
        failures = [k for k, v in checks.items() if not v]
        logger.error("[seed=%d] ❌ GATE FAIL: %s", seed, ", ".join(failures))

    return passed, report


def _write_consolidated_summary(reports: list[dict], output_path: Path) -> None:
    """Escribe ``outputs/timegan/timegan_summary.md`` con sección por seed +
    consolidado (decisión 8.1)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(reports)
    n_pass = sum(1 for r in reports if r["passed"])
    seeds = [r["seed"] for r in reports]

    rows: list[str] = []
    for r in reports:
        rows.append(
            f"| {r['seed']} | {r['discriminative_score']['mean']:.4f} | "
            f"{r['predictive_score']['gap']:.4f} | {r['min_acf_abs_ratio']:.4f} | "
            f"{'✅' if r['passed'] else '❌'} |"
        )
    table_metrics = "\n".join(rows)

    failure_details: list[str] = []
    for r in reports:
        if not r["passed"]:
            failed = [k for k, v in r["checks"].items() if not v]
            failure_details.append(f"- **seed={r['seed']}**: falló en {', '.join(failed)}")

    content = f"""# Resumen Fase 4 — TimeGAN multivariado

Generado: {dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}

Insumo directo para el Cap. 5 sección "Fase 4: Modelado generativo" de la memoria del TFE.

## 1. Resumen ejecutivo

- **Seeds evaluados**: {seeds} (total {n})
- **Seeds que aprueban el gate F4-T14**: {n_pass}/{n}
- **Criterio MVP**: 1 de 3 seeds aprobando el gate (ADR §F4-T14)
- **Veredicto MVP**: {"✅ CUMPLE" if n_pass >= 1 else "❌ NO CUMPLE"}

## 2. Métricas por seed

| seed | disc_score | pred_gap | min_acf|r|_ratio | gate |
|---|---|---|---|---|
{table_metrics}

Thresholds del gate (de configs/timegan/eval/default.yaml):
- discriminative_score ≤ {reports[0]['thresholds']['max_discriminative_score']:.2f}
- predictive_gap ≤ {reports[0]['thresholds']['max_predictive_gap']:.2f}
- min_acf|r|_synth/real ≥ {reports[0]['thresholds']['min_acf_abs_ratio']:.2f}

## 3. Fallos del gate (si aplica)

{chr(10).join(failure_details) if failure_details else "_Ninguno._"}

## 4. Artefactos por seed

Cada `data/synthetic/run_<seed>/`:
- `best.pt` — checkpoint TimeGAN (mejor según disc_score)
- `synthetic_returns.npy` — log-returns sintéticos (espacio original)
- `synthetic_returns_scaled.npy` — log-returns sintéticos en [0,1] (para métricas)
- `synthetic_dataset.parquet` — dataset completo (145 cols, MultiIndex seq_id/step)
- `stylized_facts_compare.csv` — comparativa real vs sintético (6 métricas × 6 activos × 2 source)
- `metrics.json` — discriminative + predictive scores + verdicts
- `tsne.png` — visualización 2D
- `QUALITY_OK.flag` — presente si pasa el gate

## 5. Decisiones arquitectónicas

Las 17 decisiones tomadas en este plan (ver Plan_Implementacion_Fase2_TFE.md §F4
sección "Desviaciones vs ADR" y `~/.claude/plans/en-base-a-este-synchronous-storm.md`).

Las más relevantes:

- **3.1**: Loss auxiliares L_G_moments y factor 100×L_G_supervised del paper Yoon original (sin ellos hay mode collapse).
- **3.3**: Iters embedding:supervised:joint = 1:1:1 (30k total) por restricción de tiempo en M4 Pro.
- **4.1/4.2**: Open/High/Low/Volume sintéticos por ruido gaussiano calibrado sobre TRAIN (TimeGAN no los modela).
- **4.3**: Buffer histórico de 60 días reales de TRAIN para warmup de EMA(60) por secuencia sintética.
- **5.2**: Subsample no-overlapping (stride=24) antes de discriminative_score para evitar correlación serial.

## 6. Próximo paso

- Si **GATE PASS**: el dataset sintético es apto para F5/F6 (entrenamiento del Agente B con PPO sobre `data/synthetic/run_<seed>/synthetic_dataset.parquet`).
- Si **GATE FAIL**: aplicar plan de contingencia (ver `~/.claude/plans/en-base-a-este-synchronous-storm.md` §"Plan de contingencia"). Resumen:
  1. Reducir lr a 1e-4, subir iters_joint, añadir label smoothing.
  2. Subir η de 10 a 20 (más peso a reconstrucción).
  3. Considerar LayerNorm en GRU o WGAN-GP (fuera de "vanilla").
  4. Plan B definitivo: migrar a ydata-synthetic en venv aislado.
"""
    output_path.write_text(content)
    logger.info("Summary consolidado escrito en %s", output_path)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    seed = int(cfg.seed)

    # MLflow opcional (gate puede correr standalone tras descarga de Kaggle)
    MLRUNS_PATH.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"file://{MLRUNS_PATH}")
    mlflow.set_experiment("timegan_phase4_eval")

    with mlflow.start_run(run_name=f"eval_seed_{seed}_{dt.datetime.now():%Y%m%d_%H%M%S}"):
        mlflow.log_param("seed", seed)
        passed, report = evaluate_seed(cfg, seed)
        mlflow.log_metric("gate_passed", float(passed))
        mlflow.log_metric("disc_score_mean", report["discriminative_score"]["mean"])
        mlflow.log_metric("pred_gap", report["predictive_score"]["gap"])
        mlflow.log_metric("min_acf_abs_ratio", report["min_acf_abs_ratio"])
        mlflow.log_artifact(str(PROJECT_ROOT / f"data/synthetic/run_{seed}/metrics.json"))
        mlflow.log_artifact(report["tsne_path"])

    # Multirun: cuando termina el último seed, escribir summary consolidado.
    # Recolectamos todos los metrics.json existentes que sean recientes.
    all_reports: list[dict] = []
    for run_dir in sorted((PROJECT_ROOT / "data/synthetic").glob("run_*")):
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            with metrics_path.open() as f:
                all_reports.append(json.load(f))
    if all_reports:
        _write_consolidated_summary(all_reports, TIMEGAN_OUTPUT_DIR / "timegan_summary.md")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
