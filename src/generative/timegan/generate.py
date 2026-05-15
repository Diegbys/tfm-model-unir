"""Generación de muestras sintéticas desde un TimeGAN entrenado — F4-T5.

Pipeline:

1. Cargar checkpoint en eval mode.
2. Sample ``z ~ U(0,1)`` shape ``(K * n_real, T, noise_dim)``.
3. Forward: ``H_fake = S(G(z))``, ``X_fake_scaled = R(H_fake)`` ∈ [0,1].
4. Inverse MinMaxScaler → log-returns reales.

Devuelve dos arrays:

- ``synthetic_scaled``: ``(K*n_real, T, 9)`` en [0,1] — para métricas
  (decisión 5.1 del plan F4).
- ``synthetic_original``: ``(K*n_real, T, 9)`` en log-returns reales — para
  reconstrucción OHLC (F4-T6).

Si más del 5 % de los valores sintéticos salen fuera de [-0.5, 0.5] tras el
inverse_transform, se loguea un WARNING (señal de mode collapse o ruido extremo).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig

from src.data.scalers import load_timegan_scaler
from src.generative.timegan.model import TimeGAN
from src.generative.timegan.train import load_checkpoint, resolve_device

logger = logging.getLogger(__name__)


def generate_synthetic(
    checkpoint_path: Path,
    cfg: DictConfig,
    n_real: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera ``K * n_real`` secuencias sintéticas desde el checkpoint.

    Parameters
    ----------
    checkpoint_path:
        Ruta al ``best.pt`` producido por :func:`src.generative.timegan.train.train_timegan`.
    cfg:
        Config Hydra completo. Espera ``cfg.timegan.{seq_len, noise_dim, device}``
        y ``cfg.timegan.generation.K``.
    n_real:
        Número de secuencias reales en el train set (típicamente 1681).
        El número de sintéticas será ``K * n_real``.
    seed:
        Seed del muestreo del ruido z.

    Returns
    -------
    (synthetic_scaled, synthetic_original)
        Ambos arrays con shape ``(K*n_real, T, 9)`` y dtype float32.
    """
    rng_seed = int(seed)
    torch.manual_seed(rng_seed)
    np.random.seed(rng_seed)

    dev = resolve_device(cfg.timegan.device)
    model = load_checkpoint(checkpoint_path, cfg).to(dev)
    model.eval()

    K = int(cfg.timegan.generation.K)
    n_synth = K * n_real
    seq_len = int(cfg.timegan.seq_len)
    noise_dim = int(cfg.timegan.noise_dim)

    logger.info(
        "generate_synthetic: %d secuencias × T=%d × F=9 (K=%d × n_real=%d)",
        n_synth, seq_len, K, n_real,
    )

    # Generar por lotes para no saturar memoria en MPS (M4 Pro tiene 18GB
    # pero el grafo de autograd consume mucho aunque estemos en no_grad).
    batch_size = 256
    chunks: list[np.ndarray] = []
    n_remaining = n_synth
    with torch.no_grad():
        while n_remaining > 0:
            bs = min(batch_size, n_remaining)
            z = torch.rand(bs, seq_len, noise_dim, device=dev)
            h_fake = model.supervisor(model.generator(z))
            x_scaled = model.recovery(h_fake)
            chunks.append(x_scaled.cpu().numpy().astype(np.float32))
            n_remaining -= bs

    synthetic_scaled = np.concatenate(chunks, axis=0)
    assert synthetic_scaled.shape == (n_synth, seq_len, 9), (
        f"shape inesperado {synthetic_scaled.shape}"
    )

    # Inverse MinMaxScaler → log-returns reales
    scaler, cols = load_timegan_scaler()
    # MinMaxScaler trabaja sobre 2D (n, F); aplanar (n*T, F), invertir, reshape.
    flat = synthetic_scaled.reshape(-1, 9)
    inv = scaler.inverse_transform(flat).astype(np.float32)
    synthetic_original = inv.reshape(n_synth, seq_len, 9)

    # Sanity check: log-returns típicamente en [-0.2, 0.2]
    abs_orig = np.abs(synthetic_original)
    pct_extreme = float((abs_orig > 0.5).mean()) * 100.0
    if pct_extreme > 5.0:
        logger.warning(
            "generate_synthetic: %.2f%% de valores |x|>0.5 (esperado < 5%%). "
            "Posible mode collapse o ruido extremo. Min/max=%.4f/%.4f",
            pct_extreme, synthetic_original.min(), synthetic_original.max(),
        )
    else:
        logger.info(
            "generate_synthetic OK: min=%.4f, max=%.4f, mean=%.4f, %.2f%% extremos (>0.5)",
            synthetic_original.min(), synthetic_original.max(),
            synthetic_original.mean(), pct_extreme,
        )

    return synthetic_scaled, synthetic_original
