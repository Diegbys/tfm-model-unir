"""Loop de entrenamiento TimeGAN — F4-T2, F4-T4.

3 fases secuenciales (paper Yoon 2019 §3.1, **obligatorias** compass §2.3):

1. **Embedding** (``iters_embedding``): ``E + R`` minimizan
   ``L_R = MSE(R(E(x)), x)``. Sin esta fase, joint diverge.

2. **Supervised** (``iters_supervised``): ``S`` aprende dinámica temporal
   en latente: ``L_S = MSE(S(E(x))[:, :-1, :], E(x)[:, 1:, :])``.

3. **Joint** (``iters_joint``): 4 optimizadores alternan según paper §3.1
   ecs. 5-7. Cada ``eval_every`` iters se evalúa ``discriminative_score``
   sobre el holdout (10 % final cronológico, decisión 3.2 del plan F4)
   y se persiste el ``best.pt``.

**Decisión 3.1 del plan F4**: incluimos ``L_G_moments`` (MSE de media y std
entre ``R(G(z))`` y ``x``) y factor 100× en ``L_G_supervised``. Son del
paper original (no del ADR literal); sin ellos hay mode collapse casi
seguro con N=1681 y seq_len=24.

**Decisión 3.4 del plan F4**: device auto-detect (MPS→CPU); ``warn_only=True``
para deterministic algorithms porque MPS no es 100 % determinista.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.generative.timegan.metrics import discriminative_score
from src.generative.timegan.model import TimeGAN

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resultado del entrenamiento
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """Resumen del entrenamiento, persistido en ``train_log.json``."""

    best_iter: int                  # iter de la fase joint donde se logró el best
    best_disc_score: float          # mejor discriminative_score (val) observado
    final_losses: dict              # losses al cierre del entrenamiento
    checkpoint_path: str            # ruta al best.pt (str para JSON)
    early_stopped: bool             # True si patience agotada antes de iters_joint
    total_iters_joint: int          # iteraciones realmente ejecutadas en joint
    seconds_total: float            # wall-clock del entrenamiento entero
    param_counts: dict              # nº de params por sub-red (debug/MLflow)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_device(device_cfg: str | torch.device) -> torch.device:
    """Resuelve ``'auto'`` → MPS si disponible, fallback CPU.

    Permite override explícito: ``'cuda'`` (Kaggle T4), ``'cpu'`` (debug).
    """
    if isinstance(device_cfg, torch.device):
        return device_cfg
    if device_cfg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_cfg)


def _set_global_seeds(seed: int) -> None:
    """Seeds para reproducibilidad. ``warn_only=True`` en deterministic
    algorithms porque MPS no es 100 % determinista en algunas ops."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:  # noqa: BLE001 (defensive — algunas versiones de torch fallan en MPS)
        logger.warning("use_deterministic_algorithms falló: %s — continuando sin él", exc)


def _split_train_holdout(
    sequences: np.ndarray, fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    """Decisión 3.2: holdout = ``fraction`` final cronológico.

    Cortar por el final (no aleatorio) evita leakage entre secuencias
    adyacentes (stride=1 ⇒ comparten 23 timesteps).
    """
    n_total = len(sequences)
    n_holdout = max(1, int(n_total * fraction))
    n_train = n_total - n_holdout
    return sequences[:n_train], sequences[n_train:]


def _make_loader(arr: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    """DataLoader sobre un tensor único (sin labels). ``drop_last=True`` para
    estabilidad de BatchNorm/momentos en joint."""
    tensor = torch.from_numpy(arr.astype(np.float32))
    ds = TensorDataset(tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def _iter_batches(loader: DataLoader):
    """Generador infinito sobre el loader (loop por iters, no por epochs)."""
    while True:
        for (batch,) in loader:
            yield batch


# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------


def train_timegan(
    sequences: np.ndarray,
    cfg: DictConfig,
    seed: int,
    mlflow_logger=None,
) -> TrainResult:
    """Entrena TimeGAN en 3 fases con early stopping.

    Parameters
    ----------
    sequences:
        Tensor ``(N, T, F)`` float32 en [0,1] de F3
        (output de :func:`src.data.sequence_builder.load_sequences`).
    cfg:
        Config Hydra completo. Espera ``cfg.timegan`` y ``cfg.timegan.training``.
    seed:
        Seed global (se inyecta en torch/numpy/random).
    mlflow_logger:
        Opcional. Objeto con ``.log_metric(name, value, step)`` y
        ``.log_params(dict)``; si ``None``, solo se loguea a stderr.
        Pasado por :mod:`scripts.03_train_timegan` con run MLflow activo.

    Returns
    -------
    TrainResult
    """
    t_start = time.time()
    _set_global_seeds(seed)

    cfg_t = cfg.timegan
    cfg_train = cfg_t.training
    dev = resolve_device(cfg_t.device)
    logger.info("train_timegan: device=%s, seed=%d, sequences shape=%s", dev, seed, sequences.shape)

    # Modelo
    model = TimeGAN.from_config(cfg_t).to(dev)
    param_counts = model.num_parameters()
    logger.info("Param counts: %s (total=%d)", param_counts, sum(param_counts.values()))

    # Split holdout
    train_arr, holdout_arr = _split_train_holdout(sequences, float(cfg_train.holdout_fraction))
    logger.info("Split: train=%d, holdout=%d (last %.0f%%)",
                len(train_arr), len(holdout_arr), 100 * cfg_train.holdout_fraction)
    loader = _make_loader(train_arr, batch_size=int(cfg_t.batch_size), shuffle=True)
    batch_iter = _iter_batches(loader)

    # Optimizadores (paper Yoon §3.1 ecs. 5-7)
    lr = float(cfg_t.lr)
    opt_embed = torch.optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=lr,
    )
    opt_sup = torch.optim.Adam(model.supervisor.parameters(), lr=lr)
    opt_g = torch.optim.Adam(
        list(model.generator.parameters()) + list(model.supervisor.parameters()), lr=lr,
    )
    opt_er = torch.optim.Adam(
        list(model.embedder.parameters()) + list(model.recovery.parameters()), lr=lr,
    )
    opt_d = torch.optim.Adam(model.discriminator.parameters(), lr=lr)

    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    gamma = float(cfg_t.gamma)
    eta = float(cfg_t.eta)
    moments_weight = float(cfg_train.moments_weight)
    sup_weight = float(cfg_train.supervised_weight)
    noise_dim = int(cfg_t.noise_dim)
    seq_len = int(cfg_t.seq_len)
    log_every = int(cfg_train.log_loss_every)

    def _sample_z(batch_size: int) -> torch.Tensor:
        """Ruido z ~ U(0,1) shape (B, T, noise_dim). Paper Yoon usa U(0,1)."""
        return torch.rand(batch_size, seq_len, noise_dim, device=dev)

    def _log(metric: str, value: float, step: int) -> None:
        if mlflow_logger is not None:
            try:
                mlflow_logger.log_metric(metric, value, step=step)
            except Exception as exc:  # noqa: BLE001
                logger.debug("mlflow.log_metric(%s) falló: %s", metric, exc)

    # -----------------------------------------------------------------------
    # Fase 1 — Embedding (E + R)
    # -----------------------------------------------------------------------
    logger.info("=== Fase 1: Embedding (iters=%d) ===", cfg_train.iters_embedding)
    model.train()
    final_loss_r = float("nan")
    for it in range(int(cfg_train.iters_embedding)):
        x = next(batch_iter).to(dev)
        x_hat = model.recovery(model.embedder(x))
        loss_r = mse(x_hat, x)

        opt_embed.zero_grad()
        loss_r.backward()
        opt_embed.step()

        if it % log_every == 0:
            final_loss_r = loss_r.item()
            _log("phase1_loss_R", final_loss_r, step=it)
            if it % (log_every * 10) == 0:
                logger.info("  emb it=%5d  L_R=%.6f", it, final_loss_r)

    # -----------------------------------------------------------------------
    # Fase 2 — Supervised (S)
    # -----------------------------------------------------------------------
    logger.info("=== Fase 2: Supervised (iters=%d) ===", cfg_train.iters_supervised)
    final_loss_s = float("nan")
    for it in range(int(cfg_train.iters_supervised)):
        x = next(batch_iter).to(dev)
        with torch.no_grad():
            h_real = model.embedder(x)
        h_pred = model.supervisor(h_real)
        # Shift: predice next-step
        loss_s = mse(h_pred[:, :-1, :], h_real[:, 1:, :])

        opt_sup.zero_grad()
        loss_s.backward()
        opt_sup.step()

        if it % log_every == 0:
            final_loss_s = loss_s.item()
            _log("phase2_loss_S", final_loss_s, step=it)
            if it % (log_every * 10) == 0:
                logger.info("  sup it=%5d  L_S=%.6f", it, final_loss_s)

    # -----------------------------------------------------------------------
    # Fase 3 — Joint (E, R, G, S, D) con early stopping
    # -----------------------------------------------------------------------
    logger.info("=== Fase 3: Joint (iters=%d, eval_every=%d) ===",
                cfg_train.iters_joint, cfg_train.eval_every)

    checkpoint_dir = Path(cfg_train.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"

    best_disc = float("inf")
    best_iter = -1
    patience_counter = 0
    early_stopped = False
    iters_done = 0
    final_losses: dict[str, float] = {}

    d_updates = int(cfg_train.d_updates_per_g)
    eval_every = int(cfg_train.eval_every)
    patience = int(cfg_train.patience)
    min_iters = int(cfg_train.min_iters_before_stop)
    iters_joint = int(cfg_train.iters_joint)

    for it in range(iters_joint):
        iters_done = it + 1
        x = next(batch_iter).to(dev)
        bs = x.size(0)

        # ---- Update D (d_updates veces) ----
        for _ in range(d_updates):
            z = _sample_z(bs)
            with torch.no_grad():
                h_real_det = model.embedder(x)
                h_fake_g = model.generator(z)
                h_fake_s = model.supervisor(h_fake_g)
            logits_real = model.discriminator(h_real_det)
            logits_fake = model.discriminator(h_fake_g)
            logits_fake_s = model.discriminator(h_fake_s)
            loss_d_real = bce(logits_real, torch.ones_like(logits_real))
            loss_d_fake = bce(logits_fake, torch.zeros_like(logits_fake))
            loss_d_fake_s = bce(logits_fake_s, torch.zeros_like(logits_fake_s))
            loss_d = loss_d_real + loss_d_fake + loss_d_fake_s

            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()

        # ---- Update G + S ----
        z = _sample_z(bs)
        h_real = model.embedder(x)               # se reusará en E+R update
        h_fake_g = model.generator(z)
        h_fake_s = model.supervisor(h_fake_g)
        x_fake = model.recovery(h_fake_s)

        # Adversarial (G quiere que D crea que es real)
        logits_fake = model.discriminator(h_fake_g)
        logits_fake_s = model.discriminator(h_fake_s)
        loss_g_adv = bce(logits_fake, torch.ones_like(logits_fake)) + bce(
            logits_fake_s, torch.ones_like(logits_fake_s)
        )

        # Supervised loss sobre G (decisión 3.1: factor 100×)
        loss_g_sup = mse(
            model.supervisor(h_real)[:, :-1, :], h_real[:, 1:, :].detach()
        )

        # Moments loss (decisión 3.1: factor 100·sqrt)
        # MSE entre media y std de R(G(z)) vs x (por feature)
        mean_x = x.mean(dim=(0, 1))
        std_x = x.std(dim=(0, 1))
        mean_xf = x_fake.mean(dim=(0, 1))
        std_xf = x_fake.std(dim=(0, 1))
        loss_g_moments = ((mean_xf - mean_x).pow(2).mean() + (std_xf - std_x).pow(2).mean())

        loss_g = (
            gamma * loss_g_adv
            + moments_weight * torch.sqrt(loss_g_moments + 1e-8)
            + sup_weight * loss_g_sup
        )

        opt_g.zero_grad()
        loss_g.backward()
        opt_g.step()

        # ---- Update E + R (mantener compatibilidad con S) ----
        h_real = model.embedder(x)
        x_hat = model.recovery(h_real)
        loss_r = mse(x_hat, x)
        loss_s_for_er = mse(
            model.supervisor(h_real).detach()[:, :-1, :], h_real[:, 1:, :]
        )
        loss_er = eta * torch.sqrt(loss_r + 1e-8) + 0.1 * loss_s_for_er

        opt_er.zero_grad()
        loss_er.backward()
        opt_er.step()

        # ---- Logging ----
        if it % log_every == 0:
            final_losses = {
                "loss_D": float(loss_d.item()),
                "loss_G_adv": float(loss_g_adv.item()),
                "loss_G_sup": float(loss_g_sup.item()),
                "loss_G_moments": float(loss_g_moments.item()),
                "loss_R": float(loss_r.item()),
                "loss_ER": float(loss_er.item()),
            }
            for k, v in final_losses.items():
                _log(f"joint_{k}", v, step=it)
            if it % (log_every * 10) == 0:
                logger.info(
                    "  joint it=%5d  L_D=%.4f  L_G_adv=%.4f  L_G_sup=%.4f  L_G_mom=%.4f  L_R=%.5f",
                    it,
                    final_losses["loss_D"],
                    final_losses["loss_G_adv"],
                    final_losses["loss_G_sup"],
                    final_losses["loss_G_moments"],
                    final_losses["loss_R"],
                )

        # ---- Early stopping eval ----
        if (it + 1) % eval_every == 0 and (it + 1) >= min_iters:
            disc_eval = _evaluate_discriminative(
                model=model,
                holdout=holdout_arr,
                cfg_eval=cfg_t.eval.discriminative,
                device=dev,
                noise_dim=noise_dim,
                seq_len=seq_len,
                seed=seed + it,
            )
            _log("eval_disc_score", disc_eval, step=it)
            logger.info("  [eval] it=%5d  disc_score=%.4f  (best=%.4f @%d)",
                        it + 1, disc_eval, best_disc, best_iter)

            if disc_eval < best_disc:
                best_disc = disc_eval
                best_iter = it + 1
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "best_disc_score": best_disc,
                        "best_iter": best_iter,
                        "config": _config_to_dict(cfg_t),
                        "seed": seed,
                    },
                    best_path,
                )
                patience_counter = 0
                logger.info("  ↳ checkpoint actualizado en %s", best_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping: %d evaluaciones sin mejora", patience)
                    early_stopped = True
                    break

    # Si nunca se guardó best (e.g., iters_joint < min_iters), persistir el final
    if best_iter < 0:
        logger.warning(
            "No se completaron evaluaciones de early stopping; guardando estado final"
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "best_disc_score": float("nan"),
                "best_iter": iters_done,
                "config": _config_to_dict(cfg_t),
                "seed": seed,
            },
            best_path,
        )
        best_iter = iters_done
        best_disc = float("nan")

    result = TrainResult(
        best_iter=best_iter,
        best_disc_score=float(best_disc),
        final_losses=final_losses,
        checkpoint_path=str(best_path),
        early_stopped=early_stopped,
        total_iters_joint=iters_done,
        seconds_total=time.time() - t_start,
        param_counts=param_counts,
    )

    # Persistir TrainResult como JSON
    log_path = checkpoint_dir / "train_log.json"
    with log_path.open("w") as f:
        json.dump(asdict(result), f, indent=2)
    logger.info(
        "train_timegan FIN: best_iter=%d, best_disc=%.4f, secs=%.1f, ckpt=%s",
        result.best_iter, result.best_disc_score, result.seconds_total, result.checkpoint_path,
    )
    return result


# ---------------------------------------------------------------------------
# Evaluación en holdout
# ---------------------------------------------------------------------------


def _evaluate_discriminative(
    *,
    model: TimeGAN,
    holdout: np.ndarray,
    cfg_eval: DictConfig,
    device: torch.device,
    noise_dim: int,
    seq_len: int,
    seed: int,
) -> float:
    """Genera un batch sintético del mismo tamaño que el holdout y mide
    ``discriminative_score`` (media de repeats). Usado para early stopping.

    Para acelerar (esto ocurre cada eval_every iters), reducimos epochs del
    classifier y n_repeats vs el eval final.
    """
    model.eval()
    n_h = len(holdout)
    with torch.no_grad():
        z = torch.rand(n_h, seq_len, noise_dim, device=device)
        h_fake = model.supervisor(model.generator(z))
        synth_scaled = model.recovery(h_fake).cpu().numpy()
    model.train()

    # Eval rápida: 1 repeat, 10 epochs (vs n_repeats=3, epochs=50 del eval final)
    score = discriminative_score(
        real_seqs=holdout,
        synthetic_seqs=synth_scaled,
        n_repeats=1,
        epochs=10,
        batch_size=int(cfg_eval.batch_size),
        lr=float(cfg_eval.lr),
        hidden_dim=int(cfg_eval.classifier_hidden),
        non_overlap_stride=int(cfg_eval.non_overlap_stride),
        device=device,
        seed=seed,
    )
    return score["mean"]


def _config_to_dict(cfg) -> dict:
    """Convierte DictConfig a dict puro (para JSON/torch.save)."""
    from omegaconf import OmegaConf

    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Carga de checkpoint (utilidad para generate.py)
# ---------------------------------------------------------------------------


def load_checkpoint(checkpoint_path: Path, cfg: DictConfig) -> TimeGAN:
    """Carga ``best.pt`` y devuelve el modelo en eval mode, en el device de cfg."""
    dev = resolve_device(cfg.timegan.device)
    model = TimeGAN.from_config(cfg.timegan).to(dev)
    payload = torch.load(checkpoint_path, map_location=dev, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    logger.info(
        "Checkpoint cargado de %s (best_iter=%s, best_disc=%s)",
        checkpoint_path, payload.get("best_iter"), payload.get("best_disc_score"),
    )
    return model
