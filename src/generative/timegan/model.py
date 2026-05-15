"""TimeGAN — F4-T1: las 5 redes (Embedder, Recovery, Generator, Supervisor,
Discriminator) y un wrapper :class:`TimeGAN` que las agrupa.

Sigue paper Yoon et al. 2019 (NeurIPS) §3.1 y compass §2.3. Todas las redes
usan **GRU** (más estable que LSTM para series financieras, compass §2.3) con
``num_layers=3`` y ``hidden_dim=24`` (= ``seq_len``, parametrización original).

Salidas con activación sigmoide (E, R, G, S) porque:

- Los datos reales están en [0,1] post-MinMaxScaler (F3-T6).
- El paper Yoon usa sigmoide en R como capa de salida (matching del dominio).
- E también devuelve embeddings en [0,1] para que S y D operen en un
  espacio acotado (estabilidad de entrenamiento).

D devuelve **logits** (sin sigmoide); la pérdida se calcula con
``BCEWithLogitsLoss`` en :mod:`src.generative.timegan.train`.

Sin dropout: el paper original no lo usa. ``dropout=0.1`` se reserva como
contingencia si el gate F4-T14 falla por mode collapse.
"""
from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig
from torch import nn

logger = logging.getLogger(__name__)


def _build_gru(input_dim: int, hidden_dim: int, num_layers: int) -> nn.GRU:
    """GRU canónica con ``batch_first=True``. Aislada para que las 5 redes
    compartan exactamente la misma configuración salvo input_dim."""
    return nn.GRU(
        input_size=input_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        batch_first=True,
    )


class Embedder(nn.Module):
    """E: espacio real ``(B, T, n_features)`` → latente ``(B, T, hidden_dim)``.

    Output con sigmoide para que los embeddings vivan en [0,1] (estabilidad
    de S y D, paper Yoon §3.1).

    >>> e = Embedder(n_features=9, hidden_dim=24, num_layers=3)
    >>> e(torch.randn(4, 24, 9)).shape
    torch.Size([4, 24, 24])
    """

    def __init__(self, n_features: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.gru = _build_gru(n_features, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        return self.act(self.fc(h))


class Recovery(nn.Module):
    """R: latente ``(B, T, hidden_dim)`` → reconstrucción ``(B, T, n_features)``.

    Output con sigmoide porque los datos de entrada están en [0,1] (post-MinMax).

    >>> r = Recovery(n_features=9, hidden_dim=24, num_layers=3)
    >>> r(torch.randn(4, 24, 24)).shape
    torch.Size([4, 24, 9])
    """

    def __init__(self, n_features: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.gru = _build_gru(hidden_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, n_features)
        self.act = nn.Sigmoid()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(h)
        return self.act(self.fc(out))


class Generator(nn.Module):
    """G: ruido ``(B, T, noise_dim)`` → embeddings fake ``(B, T, hidden_dim)``.

    Output con sigmoide para que comparta dominio con ``E(x)`` (los embeddings
    reales también están en [0,1] gracias a la sigmoide de E).

    >>> g = Generator(noise_dim=32, hidden_dim=24, num_layers=3)
    >>> g(torch.randn(4, 24, 32)).shape
    torch.Size([4, 24, 24])
    """

    def __init__(self, noise_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.gru = _build_gru(noise_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Sigmoid()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(z)
        return self.act(self.fc(h))


class Supervisor(nn.Module):
    """S: latente ``(B, T, hidden_dim)`` → predicción next-step ``(B, T, hidden_dim)``.

    Aprende la dinámica temporal en el espacio latente: ``S(h)[t]`` predice
    ``h[t+1]``. Output con sigmoide (mismo dominio que E y G).

    La pérdida se aplica con shift en train.py:
    ``L_S = MSE(S(h)[:, :-1], h[:, 1:])``.

    >>> s = Supervisor(hidden_dim=24, num_layers=3)
    >>> s(torch.randn(4, 24, 24)).shape
    torch.Size([4, 24, 24])
    """

    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.gru = _build_gru(hidden_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.Sigmoid()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(h)
        return self.act(self.fc(out))


class Discriminator(nn.Module):
    """D: latente ``(B, T, hidden_dim)`` → logits ``(B, T, 1)``.

    **NO aplica sigmoide**: la pérdida se calcula con
    ``BCEWithLogitsLoss`` (más estable numéricamente que BCE+sigmoide
    separados).

    >>> d = Discriminator(hidden_dim=24, num_layers=3)
    >>> d(torch.randn(4, 24, 24)).shape
    torch.Size([4, 24, 1])
    """

    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.gru = _build_gru(hidden_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(h)
        return self.fc(out)


class TimeGAN(nn.Module):
    """Wrapper que agrupa las 5 redes.

    Facilita:

    - ``.to(device)`` mueve las 5 redes en una llamada.
    - ``.train()`` / ``.eval()`` propaga modos correctamente.
    - ``torch.save(model.state_dict(), path)`` guarda un único checkpoint
      con las 5 redes (F4-T4 lo necesita para persistir el best).

    Las 5 clases siguen disponibles sueltas para testeo aislado.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        num_layers: int,
        noise_dim: int,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.noise_dim = noise_dim
        self.embedder = Embedder(n_features, hidden_dim, num_layers)
        self.recovery = Recovery(n_features, hidden_dim, num_layers)
        self.generator = Generator(noise_dim, hidden_dim, num_layers)
        self.supervisor = Supervisor(hidden_dim, num_layers)
        self.discriminator = Discriminator(hidden_dim, num_layers)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "TimeGAN":
        """Construye desde ``cfg.timegan`` o equivalente DictConfig.

        Espera las claves ``n_features``, ``hidden_dim``, ``num_layers``,
        ``noise_dim``. Si el ``module`` no es ``gru``, lanza ``ValueError``
        (LSTM no soportado por decisión arquitectónica, compass §2.3).
        """
        module = str(cfg.get("module", "gru")).lower()
        if module != "gru":
            raise ValueError(
                f"TimeGAN.from_config: module={module!r} no soportado; "
                "usar 'gru' (compass §2.3: LSTM más inestable para finanzas)"
            )
        return cls(
            n_features=int(cfg.n_features),
            hidden_dim=int(cfg.hidden_dim),
            num_layers=int(cfg.num_layers),
            noise_dim=int(cfg.noise_dim),
        )

    def num_parameters(self) -> dict[str, int]:
        """Cuenta de parámetros por sub-red, útil para logs/MLflow."""
        return {
            "embedder": sum(p.numel() for p in self.embedder.parameters()),
            "recovery": sum(p.numel() for p in self.recovery.parameters()),
            "generator": sum(p.numel() for p in self.generator.parameters()),
            "supervisor": sum(p.numel() for p in self.supervisor.parameters()),
            "discriminator": sum(p.numel() for p in self.discriminator.parameters()),
        }
