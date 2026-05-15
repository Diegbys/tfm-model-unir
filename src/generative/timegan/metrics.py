"""Métricas TimeGAN — F4-T8, F4-T9, F4-T10.

Las 3 métricas estándar del paper Yoon et al. 2019 §5:

- :func:`discriminative_score` (F4-T8): clasificador GRU 2-capas entrenado a
  distinguir secuencias reales de sintéticas. Reporta ``|accuracy - 0.5|``.
  Score ≈ 0 ⇒ indistinguibles (sintéticos perfectos). Score → 0.5 ⇒ basura.

- :func:`predictive_score` (F4-T9): forecaster GRU entrenado en sintéticos y
  evaluado en reales (TSTR = Train-on-Synthetic, Test-on-Real). Reporta
  ``gap = |mae_tstr - mae_trtr| / mae_trtr``. Gap < 10 % bueno.

- :func:`tsne_plot` (F4-T10): proyecta 1000 reales + 1000 sintéticas a 2D
  (t-SNE) + PCA comparativo.

**Decisión 5.1**: las 3 métricas operan en espacio **MinMax-escalado [0,1]**,
NO en log-returns originales. Es lo que hace el paper Yoon §5 y los rangos
uniformes mejoran convergencia del classifier/forecaster.

**Decisión 5.2**: ``non_overlap_stride=24`` antes del classifier. El stride=1
de F3 genera correlación serial que infla ``accuracy`` trivialmente
(documentado en outputs/preproc/preproc_summary.md §5.4).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modelos auxiliares (NO confundir con las redes del TimeGAN — éstos son los
# classifier/forecaster post-hoc para las métricas).
# ---------------------------------------------------------------------------


class GRUClassifier(nn.Module):
    """Clasificador binario GRU 2-capas para discriminative_score.

    Recibe secuencia ``(B, T, F)`` y devuelve logit escalar ``(B,)`` ⇒
    probabilidad de ser real (1) vs sintético (0).
    """

    def __init__(self, n_features: int, hidden_dim: int = 24, num_layers: int = 2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)  # h_n: (num_layers, B, hidden)
        last = h_n[-1]        # (B, hidden) — última capa
        return self.fc(last).squeeze(-1)  # (B,)


class GRUForecaster(nn.Module):
    """Forecaster GRU 2-capas para predictive_score (TSTR).

    Recibe ``(B, T-1, F)`` y predice ``(B, F)`` = paso T.
    """

    def __init__(self, n_features: int, hidden_dim: int = 24, num_layers: int = 2) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------


def _resolve_device(device: torch.device | str | None) -> torch.device:
    """Resuelve None → auto (MPS o CPU). Aislada para reusar en train.py.

    NOTA: aquí duplicamos lógica de :func:`src.generative.timegan.train.resolve_device`
    para evitar dependencia circular (train.py importa metrics.py para el
    early stopping). Mantenerlas sincronizadas.
    """
    if device is None or device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def _subsample_non_overlap(seqs: np.ndarray, stride: int) -> np.ndarray:
    """Subsamplea no-overlapping (decisión 5.2). Para stride=24 con seq_len=24,
    cada par de secuencias consecutivas comparte 0 timesteps reales."""
    if stride <= 1:
        return seqs
    return seqs[::stride]


# ---------------------------------------------------------------------------
# F4-T8: Discriminative score
# ---------------------------------------------------------------------------


def discriminative_score(
    real_seqs: np.ndarray,
    synthetic_seqs: np.ndarray,
    *,
    n_repeats: int = 3,
    test_size: float = 0.20,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1.0e-3,
    hidden_dim: int = 24,
    non_overlap_stride: int = 24,
    device: torch.device | str | None = None,
    seed: int = 0,
) -> dict:
    """Calcula |accuracy - 0.5| de un classifier GRU sobre real vs synth.

    Paper Yoon 2019 §5: si el classifier no distingue ⇒ score ≈ 0 (perfecto).
    Si distingue perfectamente ⇒ score → 0.5.

    Parameters
    ----------
    real_seqs, synthetic_seqs:
        Arrays ``(N, T, F)`` en escala MinMax [0,1] (decisión 5.1).
    n_repeats:
        Repeticiones del entrenamiento con splits distintos, promediadas.
    non_overlap_stride:
        Subsample no-overlapping de ``real_seqs`` antes del classifier
        (decisión 5.2). Para ``stride=24`` con seq_len=24, sin solapamiento.

    Returns
    -------
    dict
        ``{'mean': float, 'std': float, 'repeats': [n_repeats floats]}``.
    """
    if real_seqs.ndim != 3 or synthetic_seqs.ndim != 3:
        raise ValueError(
            f"discriminative_score: arrays 3D requeridos; got "
            f"{real_seqs.shape} y {synthetic_seqs.shape}"
        )
    if real_seqs.shape[1:] != synthetic_seqs.shape[1:]:
        raise ValueError(
            f"discriminative_score: shapes (T, F) no coinciden: "
            f"{real_seqs.shape[1:]} vs {synthetic_seqs.shape[1:]}"
        )

    dev = _resolve_device(device)
    n_features = real_seqs.shape[2]

    real_sub = _subsample_non_overlap(real_seqs, non_overlap_stride)
    # Balanceo de clases (CRÍTICO): el classifier GRU colapsa a la clase
    # mayoritaria si los tamaños difieren mucho. Con 71 reales (tras stride)
    # vs 3362 sintéticas, predecir siempre "sintético" da accuracy ≈ 0.98 →
    # score artificial ≈ 0.48 sin ninguna relación con la calidad del modelo.
    # El paper Yoon 2019 §5 usa cantidades IGUALES por construcción.
    n_balanced = min(len(real_sub), len(synthetic_seqs))
    logger.info(
        "discriminative_score: %d reales (post stride=%d), %d sintéticas → "
        "balanceado a %d/clase, %d repeats × %d epochs en %s",
        len(real_sub), non_overlap_stride, len(synthetic_seqs), n_balanced,
        n_repeats, epochs, dev,
    )

    repeats: list[float] = []
    for rep_idx in range(n_repeats):
        rep_seed = seed + rep_idx
        torch.manual_seed(rep_seed)
        np.random.seed(rep_seed)
        rng = np.random.default_rng(rep_seed)

        # Sub-muestreo balanceado: n_balanced de cada clase, sin reemplazo.
        # Cada repeat muestrea distinto ⇒ mejor estimación de media/std.
        real_pick = rng.choice(len(real_sub), size=n_balanced, replace=False)
        synth_pick = rng.choice(len(synthetic_seqs), size=n_balanced, replace=False)
        real_bal = real_sub[real_pick]
        synth_bal = synthetic_seqs[synth_pick]

        # Mix etiquetado (clases balanceadas)
        X = np.concatenate([real_bal, synth_bal], axis=0).astype(np.float32)
        y = np.concatenate(
            [np.ones(n_balanced, dtype=np.float32),
             np.zeros(n_balanced, dtype=np.float32)],
            axis=0,
        )
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=rep_seed, stratify=y,
        )

        # DataLoader
        ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        ds_te = TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te))
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)
        dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)

        # Train classifier
        clf = GRUClassifier(n_features=n_features, hidden_dim=hidden_dim).to(dev)
        opt = torch.optim.Adam(clf.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss()
        clf.train()
        for _ in range(epochs):
            for xb, yb in dl_tr:
                xb, yb = xb.to(dev), yb.to(dev)
                opt.zero_grad()
                logits = clf(xb)
                loss = bce(logits, yb)
                loss.backward()
                opt.step()

        # Eval accuracy
        clf.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in dl_te:
                xb, yb = xb.to(dev), yb.to(dev)
                preds = (torch.sigmoid(clf(xb)) > 0.5).float()
                correct += (preds == yb).sum().item()
                total += yb.numel()
        accuracy = correct / total
        score = abs(accuracy - 0.5)
        repeats.append(float(score))
        logger.info(
            "discriminative_score rep %d/%d: accuracy=%.4f → score=%.4f",
            rep_idx + 1, n_repeats, accuracy, score,
        )

    arr = np.array(repeats)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "repeats": [float(r) for r in repeats],
    }


# ---------------------------------------------------------------------------
# F4-T9: Predictive score (TSTR)
# ---------------------------------------------------------------------------


def _train_forecaster(
    train_seqs: np.ndarray,
    *,
    n_features: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> GRUForecaster:
    """Entrena un forecaster GRU sobre ``train_seqs``.

    Input: ``x[:, :-1, :]`` shape ``(B, T-1, F)``.
    Target: ``x[:, -1, :]`` shape ``(B, F)``.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    x = torch.from_numpy(train_seqs[:, :-1, :].astype(np.float32))
    y = torch.from_numpy(train_seqs[:, -1, :].astype(np.float32))
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = GRUForecaster(n_features=n_features, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = mse(pred, yb)
            loss.backward()
            opt.step()
    return model


def _eval_forecaster_mae(model: GRUForecaster, seqs: np.ndarray, device: torch.device) -> float:
    """MAE del forecaster sobre ``seqs`` (paso T predicho desde paso 1..T-1)."""
    model.eval()
    x = torch.from_numpy(seqs[:, :-1, :].astype(np.float32)).to(device)
    y = torch.from_numpy(seqs[:, -1, :].astype(np.float32)).to(device)
    with torch.no_grad():
        pred = model(x)
        mae = (pred - y).abs().mean().item()
    return float(mae)


def predictive_score(
    real_seqs: np.ndarray,
    synthetic_seqs: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1.0e-3,
    hidden_dim: int = 24,
    device: torch.device | str | None = None,
    seed: int = 0,
) -> dict:
    """Predictive score TSTR (Train-on-Synthetic, Test-on-Real).

    Pipeline:

    1. Entrena forecaster_S sobre ``synthetic_seqs``. Evalúa MAE en real → ``mae_tstr``.
    2. Baseline: entrena forecaster_R sobre ``real_seqs``. Evalúa MAE en real → ``mae_trtr``.
    3. ``gap = |mae_tstr - mae_trtr| / mae_trtr``.

    Returns
    -------
    dict
        ``{'mae_tstr': float, 'mae_trtr': float, 'gap': float}``.
    """
    dev = _resolve_device(device)
    n_features = real_seqs.shape[2]

    logger.info(
        "predictive_score: training TSTR (synth=%d) y TRTR (real=%d), %d epochs en %s",
        len(synthetic_seqs), len(real_seqs), epochs, dev,
    )

    # TSTR
    fc_s = _train_forecaster(
        synthetic_seqs, n_features=n_features, hidden_dim=hidden_dim,
        epochs=epochs, batch_size=batch_size, lr=lr, device=dev, seed=seed,
    )
    mae_tstr = _eval_forecaster_mae(fc_s, real_seqs, dev)

    # TRTR (baseline)
    fc_r = _train_forecaster(
        real_seqs, n_features=n_features, hidden_dim=hidden_dim,
        epochs=epochs, batch_size=batch_size, lr=lr, device=dev, seed=seed + 1,
    )
    mae_trtr = _eval_forecaster_mae(fc_r, real_seqs, dev)

    gap = abs(mae_tstr - mae_trtr) / max(mae_trtr, 1e-12)
    logger.info(
        "predictive_score: mae_tstr=%.6f, mae_trtr=%.6f, gap=%.4f",
        mae_tstr, mae_trtr, gap,
    )
    return {"mae_tstr": float(mae_tstr), "mae_trtr": float(mae_trtr), "gap": float(gap)}


# ---------------------------------------------------------------------------
# F4-T10: t-SNE / PCA visualization
# ---------------------------------------------------------------------------


def tsne_plot(
    real_seqs: np.ndarray,
    synthetic_seqs: np.ndarray,
    output_path: Path,
    *,
    n_samples: int = 1000,
    perplexity: int = 30,
    random_state: int = 42,
) -> Path:
    """Proyecta n_samples reales + n_samples sintéticas a 2D y plotea
    (t-SNE + PCA en panel doble).

    Flatten ``(N, T, F)`` → ``(N, T*F)`` antes de la reducción de dimensión.
    """
    import matplotlib.pyplot as plt  # lazy import
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    rng = np.random.default_rng(random_state)
    n_real = min(n_samples, len(real_seqs))
    n_synth = min(n_samples, len(synthetic_seqs))
    idx_real = rng.choice(len(real_seqs), size=n_real, replace=False)
    idx_synth = rng.choice(len(synthetic_seqs), size=n_synth, replace=False)

    real_flat = real_seqs[idx_real].reshape(n_real, -1)
    synth_flat = synthetic_seqs[idx_synth].reshape(n_synth, -1)
    X = np.concatenate([real_flat, synth_flat], axis=0)
    labels = np.array(["real"] * n_real + ["synth"] * n_synth)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=110)

    # PCA
    pca = PCA(n_components=2, random_state=random_state)
    coords_pca = pca.fit_transform(X)
    axes[0].scatter(
        coords_pca[labels == "real", 0], coords_pca[labels == "real", 1],
        s=8, alpha=0.5, c="#3b6ea5", label=f"real (n={n_real})",
    )
    axes[0].scatter(
        coords_pca[labels == "synth", 0], coords_pca[labels == "synth", 1],
        s=8, alpha=0.5, c="#c0504d", label=f"synth (n={n_synth})",
    )
    axes[0].set_title("PCA — 2D")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend(loc="upper right")

    # t-SNE
    tsne = TSNE(
        n_components=2, perplexity=perplexity, random_state=random_state,
        init="pca", learning_rate="auto",
    )
    coords_tsne = tsne.fit_transform(X)
    axes[1].scatter(
        coords_tsne[labels == "real", 0], coords_tsne[labels == "real", 1],
        s=8, alpha=0.5, c="#3b6ea5", label=f"real (n={n_real})",
    )
    axes[1].scatter(
        coords_tsne[labels == "synth", 0], coords_tsne[labels == "synth", 1],
        s=8, alpha=0.5, c="#c0504d", label=f"synth (n={n_synth})",
    )
    axes[1].set_title(f"t-SNE — 2D (perplexity={perplexity})")
    axes[1].set_xlabel("dim 1"); axes[1].set_ylabel("dim 2")
    axes[1].legend(loc="upper right")

    fig.suptitle("TimeGAN — reales vs sintéticas (escala MinMax [0,1])")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("tsne_plot guardado en %s", output_path)
    return output_path
