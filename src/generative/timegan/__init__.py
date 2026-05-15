"""TimeGAN multivariado — Fase 4 del TFE.

Módulos:
- ``model``: las 5 redes (Embedder, Recovery, Generator, Supervisor, Discriminator)
  + wrapper :class:`TimeGAN` (F4-T1).
- ``train``: loop de 3 fases (embedding → supervised → joint) con early stopping
  por discriminative score (F4-T2, F4-T4).
- ``generate``: muestreo desde un checkpoint entrenado (F4-T5).
- ``reconstruct_prices``: OHLCV sintético desde log-returns (F4-T6).
- ``build_synthetic_dataset``: features completas sobre OHLCV sintético (F4-T7).
- ``metrics``: discriminative score, predictive score, t-SNE (F4-T8, T9, T10).
- ``stylized_facts_compare``: wrapper sobre :func:`src.eval.stylized_facts.compare_real_vs_synthetic`
  (F4-T11).
"""
