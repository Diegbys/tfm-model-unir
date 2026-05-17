"""Fijación de seeds global para reproducibilidad — F6-T7.

El entrenamiento de PPO toca varias fuentes de aleatoriedad: el muestreo de
episodios (``MixedDataset``), la inicialización de la política, el ruido de la
distribución de acciones y el rollout del entorno. ``set_global_seed`` las
siembra todas de una vez para que dos corridas con el mismo seed sean
reproducibles (ADR §5.7).
"""
from __future__ import annotations

import logging
import random

import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed

logger = logging.getLogger(__name__)


def set_global_seed(seed: int) -> None:
    """Siembra todas las fuentes de aleatoriedad del proceso.

    Cubre ``random``, ``numpy``, ``torch`` (CPU y CUDA) y el helper de
    Stable-Baselines3. Llamar **antes** de construir el ``MixedDataset``, el
    entorno y el modelo PPO.

    Parameters
    ----------
    seed:
        Semilla entera. Los seeds del experimento son ``{0, 1, 42, 123, 1337}``
        (ADR §Fase 6 / compass §4.7).
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Helper de SB3: re-siembra random/numpy/torch y, con using_cuda, fija el
    # determinismo de cuDNN. Se llama el último para que su estado prevalezca.
    set_random_seed(seed, using_cuda=torch.cuda.is_available())
    logger.info("set_global_seed: fuentes de aleatoriedad sembradas con seed=%d", seed)
