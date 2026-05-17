"""Mezclador de datos real/sintético para el Agente B — F6-T4.

El Agente A entrena solo con datos reales; el Agente B con una mezcla 1:2
(real:sintético, ADR §2.6). ``MixedDataset`` realiza ese muestreo a nivel de
episodio: en cada ``sample_episode`` decide —con probabilidad ``synthetic_ratio``—
si el episodio será sintético o real, y ensambla la trayectoria correspondiente.

Estructura de un episodio (decisión del plan F6)
------------------------------------------------
Cada trayectoria tiene ``window + body_len`` filas = 30 warmup + 24 cuerpo:

- **Episodio real**: 54 filas consecutivas del split de train real.
- **Episodio sintético**: 30 días reales de train (warmup, para llenar la ventana
  de observación de 30 días) + una de las 3362 secuencias sintéticas de 24 pasos.

Las 24 filas del cuerpo son operables salvo la primera: su retorno cruzaría la
frontera warmup→cuerpo (precios reales de miles vs sintéticos que arrancan en
100) y sería espurio. ``body_idx`` marca por eso 23 fechas operables.

Anti-leakage: el warmup y los episodios reales se muestrean SOLO de ``train_idx``
(ADR §4.2). val y test nunca pasan por aquí — se evalúan con el modo
recorrido-completo del entorno sobre los splits reales.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Índice sintético canónico: las trayectorias no tienen fechas reales, solo
# necesitan un DatetimeIndex monótono para el entorno. Año lejano para dejar
# claro que es ficticio (misma convención que build_synthetic_dataset).
_SYNTHETIC_INDEX_START = pd.Timestamp("2099-01-01")


class MixedDataset:
    """Muestreador de episodios real/sintético (ver docstring del módulo)."""

    def __init__(
        self,
        real_features_df: pd.DataFrame,
        train_idx: pd.DatetimeIndex,
        synthetic_df: pd.DataFrame | None = None,
        *,
        synthetic_ratio: float = 0.0,
        window: int = 30,
        body_len: int = 24,
        seed: int = 0,
    ) -> None:
        """Construye el mezclador.

        Parameters
        ----------
        real_features_df:
            DataFrame con el esquema de ``features.parquet`` (145 columnas,
            ``DatetimeIndex``).
        train_idx:
            Fechas de train. Warmups y episodios reales se muestrean solo de aquí.
        synthetic_df:
            ``synthetic_dataset.parquet`` con ``MultiIndex (seq_id, step)``, o
            ``None`` para el Agente A.
        synthetic_ratio:
            Probabilidad de que un episodio sea sintético (0.0 = A, ≈0.667 = B).
        window, body_len:
            Longitud del warmup y del cuerpo. La trayectoria mide ``window + body_len``.
        seed:
            Semilla del RNG interno (reproducibilidad de la secuencia de episodios).
        """
        if not 0.0 <= synthetic_ratio <= 1.0:
            raise ValueError(f"synthetic_ratio debe estar en [0,1], got {synthetic_ratio}")
        if synthetic_ratio > 0.0 and synthetic_df is None:
            raise ValueError(
                "synthetic_ratio>0 requiere synthetic_df (el Agente B necesita sintéticos)"
            )
        self.window = int(window)
        self.body_len = int(body_len)
        self.traj_len = self.window + self.body_len
        self.synthetic_ratio = float(synthetic_ratio)
        self._rng = np.random.default_rng(seed)

        # Train real: solo de aquí se muestrean warmups y episodios reales.
        self._columns = list(real_features_df.columns)
        self._real_train = real_features_df.loc[train_idx, self._columns]
        if self._real_train.isna().any().any():
            raise ValueError("MixedDataset: el train real contiene NaN")
        if len(self._real_train) < self.traj_len:
            raise ValueError(
                f"MixedDataset: train real ({len(self._real_train)} filas) < "
                f"longitud de trayectoria ({self.traj_len})"
            )
        self._real_values = self._real_train.to_numpy(dtype=np.float64)
        self._n_real = len(self._real_train)

        # Secuencias sintéticas válidas (las que tienen body_len pasos exactos).
        self._synthetic_df = synthetic_df
        self._seq_ids: list[int] = []
        if synthetic_df is not None:
            if not isinstance(synthetic_df.index, pd.MultiIndex):
                raise TypeError("synthetic_df debe tener MultiIndex (seq_id, step)")
            counts = synthetic_df.groupby(level="seq_id").size()
            self._seq_ids = [int(k) for k, n in counts.items() if n == self.body_len]
            if not self._seq_ids:
                raise ValueError(
                    f"MixedDataset: ninguna secuencia sintética tiene {self.body_len} pasos"
                )
            # Reordenar las columnas del sintético al orden canónico de features.
            self._synthetic_df = synthetic_df[self._columns]

        # Índice canónico y fechas operables (filas window+1 .. window+body_len-1).
        self._index = pd.date_range(
            start=_SYNTHETIC_INDEX_START,
            periods=self.traj_len,
            freq=pd.tseries.offsets.BusinessDay(),
        )
        self._body_idx = self._index[self.window + 1 :]

        # Trazabilidad del último episodio muestreado (tests / debugging).
        self.last_branch: str | None = None
        self.last_seq_id: int | None = None

        logger.info(
            "MixedDataset: train_real=%d filas, secuencias sintéticas=%d, "
            "synthetic_ratio=%.3f, trayectoria=%d (warmup=%d + cuerpo=%d)",
            self._n_real, len(self._seq_ids), self.synthetic_ratio,
            self.traj_len, self.window, self.body_len,
        )

    def choose_branch(self, rng: np.random.Generator) -> str:
        """Decide la rama del próximo episodio: ``"real"`` o ``"synthetic"``."""
        if self._synthetic_df is None or self.synthetic_ratio <= 0.0:
            return "real"
        return "synthetic" if rng.random() < self.synthetic_ratio else "real"

    def sample_episode(
        self, rng: np.random.Generator | None = None
    ) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
        """Muestrea un episodio y devuelve ``(trayectoria, body_idx)``.

        ``trayectoria`` es un DataFrame de ``traj_len`` filas con índice sintético
        canónico; ``body_idx`` son las 23 fechas operables. Pensado para usarse
        como ``episode_sampler`` del :class:`~src.envs.portfolio_env.PortfolioEnv`.
        """
        if rng is None:
            rng = self._rng
        branch = self.choose_branch(rng)
        if branch == "synthetic":
            values, seq_id = self._sample_synthetic(rng)
        else:
            values = self._sample_real(rng)
            seq_id = None
        self.last_branch = branch
        self.last_seq_id = seq_id
        traj = pd.DataFrame(values, index=self._index, columns=self._columns)
        return traj, self._body_idx

    def _sample_real(self, rng: np.random.Generator) -> np.ndarray:
        """Trayectoria real: ``traj_len`` filas consecutivas del train real."""
        start = int(rng.integers(0, self._n_real - self.traj_len + 1))
        return self._real_values[start : start + self.traj_len]

    def _sample_synthetic(self, rng: np.random.Generator) -> tuple[np.ndarray, int]:
        """Trayectoria sintética: warmup real (``window``) + secuencia sintética (``body_len``)."""
        warmup_start = int(rng.integers(0, self._n_real - self.window + 1))
        warmup = self._real_values[warmup_start : warmup_start + self.window]
        seq_id = int(self._seq_ids[rng.integers(0, len(self._seq_ids))])
        body = self._synthetic_df.xs(seq_id, level="seq_id").to_numpy(dtype=np.float64)
        return np.concatenate([warmup, body], axis=0), seq_id
