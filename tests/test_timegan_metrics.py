"""Tests de las métricas TimeGAN — F4.

Regresión del bug de desbalanceo de clases en :func:`discriminative_score`
(detectado en el multirun de Kaggle: los 3 seeds reportaban exactamente
``disc_score=0.4796`` porque el classifier colapsaba a la clase mayoritaria
con 71 reales vs 3362 sintéticas).
"""
from __future__ import annotations

import numpy as np

from src.generative.timegan.metrics import discriminative_score


def _gaussian_seqs(
    n: int, T: int = 10, F: int = 3, loc: float = 0.0, scale: float = 1.0, seed: int = 0
) -> np.ndarray:
    """Genera ``n`` secuencias ``(T, F)`` ~ N(loc, scale)."""
    rng = np.random.default_rng(seed)
    return rng.normal(loc, scale, size=(n, T, F)).astype(np.float32)


def test_discriminative_score_real_vs_noise_high() -> None:
    """Control positivo: real vs ruido de distribución muy distinta →
    score alto (el classifier las distingue trivialmente)."""
    real = _gaussian_seqs(120, loc=0.0, seed=1)
    noise = _gaussian_seqs(120, loc=8.0, seed=2)
    out = discriminative_score(
        real, noise, n_repeats=1, epochs=15, non_overlap_stride=1,
        device="cpu", seed=0,
    )
    assert out["mean"] > 0.30, f"esperaba score alto para datos distintos, got {out['mean']}"


def test_discriminative_score_same_distribution_low() -> None:
    """Control negativo: real vs sintético de la MISMA distribución →
    score bajo (indistinguibles)."""
    real = _gaussian_seqs(120, loc=0.0, seed=1)
    synth = _gaussian_seqs(120, loc=0.0, seed=99)
    out = discriminative_score(
        real, synth, n_repeats=1, epochs=15, non_overlap_stride=1,
        device="cpu", seed=0,
    )
    assert out["mean"] < 0.25, f"esperaba score bajo para misma distribución, got {out['mean']}"


def test_discriminative_score_balanced_under_imbalance() -> None:
    """REGRESIÓN F4: con clases muy desbalanceadas (70 reales vs 1500
    sintéticas) y MISMA distribución, el score debe seguir siendo bajo.

    Antes del fix el classifier colapsaba a la clase mayoritaria
    (predecir siempre "sintético") → accuracy ≈ 0.95 → score ≈ 0.45
    artificial, sin ninguna relación con la calidad del modelo.
    """
    real = _gaussian_seqs(70, loc=0.0, seed=1)
    synth = _gaussian_seqs(1500, loc=0.0, seed=99)
    out = discriminative_score(
        real, synth, n_repeats=1, epochs=15, non_overlap_stride=1,
        device="cpu", seed=0,
    )
    assert out["mean"] < 0.25, (
        f"score degenerado por desbalanceo de clases: {out['mean']:.4f} "
        "(el classifier colapsó a la clase mayoritaria en vez de medir calidad)"
    )
