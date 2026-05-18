"""scripts/06_backtest_agent.py — Fase 7: backtesting y evaluación.

Orquesta F7-T3…T6 sobre los modelos entrenados en F6: backtest-ea los 2 agentes
× 5 seeds sobre val y test, calcula los 3 baselines pasivos, evalúa los
sub-períodos de estrés y genera los 4 plots comparativos. La materia prima del
Cap. 5–6 y de la Fase 8 (tests estadísticos).

Es un script ``argparse`` (no Hydra): agrega corridas ya entrenadas, sin
composición de configs ni sweeps. La config del entorno la lee del ``config.yaml``
persistido en cada run.

Uso:
    uv run python scripts/06_backtest_agent.py
    uv run python scripts/06_backtest_agent.py --experiments agent_a --seeds 42
    uv run python scripts/06_backtest_agent.py --force

Inputs:
    data/processed/features.parquet
    data/processed/aligned.parquet            (proxy de bono del 60/40)
    data/splits/{val,test}_idx.parquet
    outputs/ppo_runs/<exp>/seed<seed>/{best_model,model}.zip + config.yaml
    configs/eval/stress_subperiods.yaml

Outputs:
    outputs/eval/<exp>_<split>[_final].csv     (tabla seed × métrica)
    outputs/eval/<exp>_stress_periods.csv      (seed × período × métrica)
    outputs/eval/baselines_<split>.csv         (métricas de los 3 baselines)
    outputs/eval/backtests/*.parquet           (series diarias — input de F8)
    outputs/plots/{equity_curves,drawdown_curves,return_distributions,metrics_bars}.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend headless: el script solo guarda PNGs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.download import PROJECT_ROOT  # noqa: E402
from src.eval.backtest import equity_curve  # noqa: E402
from src.eval.baselines import BASELINES  # noqa: E402
from src.eval.metrics import compute_all_metrics  # noqa: E402
from src.eval.plots import (  # noqa: E402
    plot_drawdown_curves,
    plot_equity_curves,
    plot_metrics_bars,
    plot_return_distributions,
)
from src.eval.runner import (  # noqa: E402
    EVAL_DIR,
    FEATURES_PATH,
    _backtest_path,
    _split_index,
    evaluate_runs,
    evaluate_stress_periods,
)

logger = logging.getLogger("F7")

PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
DEFAULT_EXPERIMENTS = ["agent_a", "agent_b"]
DEFAULT_SEEDS = [0, 1, 42, 123, 1337]
DEFAULT_SPLITS = ["val", "test"]


def run_baselines(split: str) -> dict[str, pd.DataFrame]:
    """Backtest-ea los 3 baselines pasivos sobre ``split`` y persiste sus métricas.

    Devuelve un dict ``{nombre: backtest_df}`` para superponerlos en los plots.
    """
    features_df = pd.read_parquet(FEATURES_PATH)
    split_idx = _split_index(split)
    backtests: dict[str, pd.DataFrame] = {}
    metrics_rows: dict[str, dict] = {}
    for name, fn in BASELINES.items():
        bt = fn(features_df, split_idx)
        backtests[name] = bt
        metrics_rows[name] = compute_all_metrics(
            bt["portfolio_return"].to_numpy(),
            equity_curve(bt),
            bt["turnover"].to_numpy(),
        )
        bt.drop(columns=["action_weights"]).to_parquet(
            EVAL_DIR / "backtests" / f"baseline_{name}_{split}.parquet"
        )
        logger.info(
            "baseline %s (%s): Sharpe=%.3f, retorno total=%.1f%%",
            name,
            split,
            metrics_rows[name]["sharpe_ratio"],
            100.0 * (bt["wealth"].iloc[-1] - 1.0),
        )
    pd.DataFrame.from_dict(metrics_rows, orient="index").to_csv(
        EVAL_DIR / f"baselines_{split}.csv", index_label="baseline"
    )
    return backtests


def load_backtests(experiment: str, seeds: list[int], split: str) -> list[pd.DataFrame]:
    """Recarga las series diarias persistidas por ``evaluate_runs`` (checkpoint best)."""
    return [
        pd.read_parquet(_backtest_path(EVAL_DIR, experiment, s, split, "best"))
        for s in seeds
    ]


def generate_plots(
    experiments: list[str], seeds: list[int], baselines_test: dict[str, pd.DataFrame]
) -> None:
    """Genera los 4 plots comparativos de F7-T6 sobre el test (checkpoint best)."""
    if set(experiments) != {"agent_a", "agent_b"}:
        logger.warning("plots A vs B: se omiten (faltan agent_a y/o agent_b)")
        return
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    bts_a = load_backtests("agent_a", seeds, "test")
    bts_b = load_backtests("agent_b", seeds, "test")
    metrics_a = pd.read_csv(EVAL_DIR / "agent_a_test.csv", index_col="seed")
    metrics_b = pd.read_csv(EVAL_DIR / "agent_b_test.csv", index_col="seed")

    figuras = {
        "equity_curves": plot_equity_curves(bts_a, bts_b, baselines_test),
        "drawdown_curves": plot_drawdown_curves(bts_a, bts_b, baselines_test),
        "return_distributions": plot_return_distributions(bts_a, bts_b),
        "metrics_bars": plot_metrics_bars(metrics_a, metrics_b),
    }
    for name, fig in figuras.items():
        fig.savefig(PLOTS_DIR / f"{name}.png", dpi=150)
        logger.info("plot guardado: outputs/plots/%s.png", name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fase 7 — backtesting y evaluación")
    parser.add_argument("--experiments", default=",".join(DEFAULT_EXPERIMENTS))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument(
        "--force", action="store_true", help="regenera aunque los outputs existan"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    experiments = [e.strip() for e in args.experiments.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    splits = [s.strip() for s in args.splits.split(",")]

    sentinel = PLOTS_DIR / "equity_curves.png"
    if sentinel.exists() and not args.force:
        logger.info(
            "Fase 7 ya generada (%s existe); usa --force para regenerar", sentinel
        )
        return 0

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    (EVAL_DIR / "backtests").mkdir(parents=True, exist_ok=True)

    # F7-T4: tabla seed × métrica por (experimento, split, checkpoint).
    for experiment in experiments:
        for split in splits:
            for checkpoint in ("best", "final"):
                table = evaluate_runs(experiment, seeds, split, checkpoint)
                logger.info(
                    "evaluate_runs %s %s [%s]:\n%s",
                    experiment,
                    split,
                    checkpoint,
                    table.round(4),
                )

    # F7-T5: evaluación condicional por sub-períodos de estrés (solo test).
    for experiment in experiments:
        if "test" in splits:
            evaluate_stress_periods(experiment, seeds, "test", "best")

    # F7-T3: baselines pasivos.
    baselines_by_split = {split: run_baselines(split) for split in splits}

    # F7-T6: plots comparativos sobre test.
    generate_plots(experiments, seeds, baselines_by_split.get("test", {}))

    logger.info(
        "=== Fase 7 completa — CSVs en outputs/eval/, plots en outputs/plots/ ==="
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
