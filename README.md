# TFE Máster IA UNIR — Robustez DRL + TimeGAN

**Equipo**: Diegbys Mudarra y Andrés Pereira
**Director**: Eduardo Muñoz Lorenzo
**Tipología**: Comparativa de soluciones (Tipo 4 UNIR)

Comparativa de un agente PPO entrenado solo con datos reales (Agente A) vs un agente PPO entrenado con datos reales + sintéticos TimeGAN multivariado (Agente B), para gestión de carteras sobre 6 activos USA + 3 macros, frecuencia diaria, período 2015–2025.

## Stack

- Python 3.11
- Dependencias: `uv` + `pyproject.toml`
- Datos: yfinance (primario) + Stooq (fallback), persistidos en Parquet
- ML (entra en fases posteriores): PyTorch 2.x, Stable-Baselines3 ≥ 2.3, Gymnasium, Hydra, MLflow

## Reproducir

```bash
uv sync --extra dev                            # instala deps
uv run python scripts/01_download_data.py      # descarga y alinea dataset
uv run pytest                                  # corre tests
```

## Snapshot de datos

- **Fecha de descarga**: <SE_RELLENA_AL_EJECUTAR_F1-T7>
- **Fuente primaria**: yfinance
- **Fuente fallback**: Stooq
- **Rango**: 2015-01-01 a 2025-04-30
- **Activos**: ^GSPC, ^NDX, AAPL, AMZN, NFLX, NVDA
- **Macros**: ^VIX, ^TNX, DX-Y.NYB
