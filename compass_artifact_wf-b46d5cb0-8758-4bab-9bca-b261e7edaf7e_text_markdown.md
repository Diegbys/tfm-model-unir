# Investigación Técnica: Robustez de Agentes DRL (PPO) con Aumentación TimeGAN para Gestión de Carteras

> **Alcance**: Reporte técnico para TFE Máster IA UNIR — comparativa de Agente A (PPO + datos reales) vs Agente B (PPO + datos reales + sintéticos TimeGAN multivariado) sobre S&P 500, NASDAQ-100, AAPL, AMZN, NFLX, NVDA con frecuencia diaria, periodo 2015–2025. Recomendaciones accionables para construir el prompt de Claude Cowork.

---

## 1. Selección de features y variables

### 1.1 Indicadores técnicos: cuáles SÍ y cuáles descartar

La revisión sistemática de papers 2023–2026 (FinRL config por defecto, MDPI 2024 *"A Systematic Approach to Portfolio Optimization"*, arXiv 2503.04143 *"MTS: A DRL Portfolio Management Framework"*, opastpublishers 2024 *"Portfolio Optimization through a Multi-modal DRL Framework"*) muestra una convergencia bastante clara sobre el "core set" de indicadores en frecuencia diaria:

**Set recomendado (mínimo viable validado en literatura):**

| Indicador | Periodo típico (diario) | Justificación |
|---|---|---|
| **MACD** (línea, signal, histograma) | (12, 26, 9) | Momento; presente en FinRL `INDICATORS` por defecto y en casi todos los papers DRL recientes |
| **RSI** | 14 | Sobrecompra/sobreventa; indicador de momento mejor cubierto en literatura |
| **Bollinger Bands** (upper, lower, ancho %B) | 20 con 2σ | Volatilidad y reversión a la media |
| **CCI** (Commodity Channel Index) | 14 o 20 | FinRL default; complementa RSI |
| **ADX / DMI** | 14 | Fuerza de tendencia (separar de dirección); usado en FinRL y MTS 2025 |
| **SMA / EMA** | 10, 30, 60 | Tendencia multi-horizonte; usar **EMA** preferentemente (menos lag) |
| **ATR** (Average True Range) | 14 | Volatilidad absoluta para posibles ajustes de tamaño |
| **OBV** (On-Balance Volume) | — | Único indicador de volumen ampliamente validado |
| **Volatilidad realizada** (rolling std de log-returns) | 5, 21 | **Sí añadir**; es la feature de riesgo más usada en papers recientes |

**Indicadores a DESCARTAR para este proyecto:**

- **Money Flow Index (MFI)**, **Stochastic Oscillator**, **Williams %R**: alta correlación con RSI, aportan poco valor marginal y aumentan dimensionalidad — crítico para TimeGAN.
- **Fibonacci Retracements / Pivot Points**: calculados sobre puntos pivote subjetivos, difíciles de incorporar como vector de features estable.
- **Ichimoku Cloud**: 5 sub-componentes, redundante con MACD+SMA, infla dimensionalidad.
- **VWAP**: con datos diarios (no intradía) pierde sentido — VWAP requiere datos a granularidad sub-diaria para tener significado.
- **Indicadores propietarios o "exóticos"** (TRIX, KST, Vortex, etc.): poca evidencia en literatura DRL y no replicables fuera de TA-Lib específico.

**Justificación arquitectónica**: con 6 activos × ~10 indicadores + OHLCV (5) + macro (3–4) → estado de ~110–130 features. Pasar de eso complica drásticamente el entrenamiento del TimeGAN (mode collapse) y del PPO (curse of dimensionality).

### 1.2 Variables de contexto de mercado (VIX, ^TNX, DX-Y.NYB)

Buena evidencia en MDPI 2025 *"Stock Market Volatility Forecasting: Exploring the Power of Deep Learning"* de que VIX + cambios diarios del DXY + Treasury yields mejoran significativamente el desempeño de modelos profundos sobre activos USA.

**Cómo incorporarlos correctamente como features de contexto:**

1. **No como precios crudos** — usar transformaciones estacionarias:
   - VIX: `log(VIX_t)` y `ΔVIX_t = VIX_t - VIX_{t-1}` (cambio absoluto, en puntos)
   - ^TNX (10Y yield): `Δyield_t` (cambios en puntos básicos), no el yield bruto
   - DXY: `log_return_DXY_t` (retorno logarítmico)
2. **Variables compartidas (no por activo)**: estas tres series son macro/cross-section; añadirlas como features globales repetidas en cada paso temporal (no replicar por activo).
3. **Lags y EMAs**: añadir EMA(VIX, 5) y EMA(VIX, 21) para capturar régimen de volatilidad.
4. **Régimen booleano** (opcional): indicador binario `VIX > 25` como flag de estrés.

### 1.3 Transformaciones y normalización

**Recomendación basada en práctica estándar:**

- **Retornos**: usar **log-returns** (`r_t = log(P_t/P_{t-1})`) para precios. Aditividad temporal y cuasi-gaussianidad parcial. Para retornos del portfolio en la reward también log-returns.
- **Precios crudos OHLCV en estado**: NO incluir precios absolutos. Reemplazar por:
  - `(High - Low) / Close` (rango relativo)
  - `(Close - Open) / Open` (cuerpo de vela relativo)
  - `log_volume = log(1 + Volume)` y/o volumen normalizado por su media móvil (`Volume / SMA(Volume, 20)`)
- **Indicadores acotados** (RSI 0–100, %B): Min-Max scaling a [-1, 1] o [0, 1].
- **Indicadores no acotados** (MACD, ATR, OBV): **z-score con ventana móvil** (ej. 252 días) o **RobustScaler** (basado en mediana/IQR). Evitar z-score global (data leakage al usar estadísticas del test).
- **Para TimeGAN**: la práctica universal en la literatura (Yoon et al. 2019, ydata-synthetic, papers 2023+) es escalar **TODAS** las features a `[0, 1]` con MinMaxScaler **fit ÚNICAMENTE en datos de entrenamiento**. TimeGAN asume datos en ese rango por su uso de sigmoide en la salida del recovery network.

**Punto crítico anti-leakage**: ajustar todos los scalers (MinMax, RobustScaler, etc.) **solo en train**, persistir el scaler (joblib/pickle), aplicar `.transform()` (no `.fit_transform()`) en validación/test.

### 1.4 Alineación temporal y feriados

- S&P 500, NASDAQ-100, AAPL/AMZN/NFLX/NVDA: todos NYSE/NASDAQ → **mismo calendario**. No hay desalineación entre estos activos.
- VIX, ^TNX, DXY: también cotizan en USA en días hábiles del NYSE → calendario común.
- **Estrategia recomendada**: usar `pandas_market_calendars` (XNYS) para obtener el universo canónico de fechas hábiles. Hacer `reindex()` al calendario y aplicar `forward-fill` solo para variables macro/exógenas (VIX, yields) en caso de huecos puntuales. Para los precios de activos, **eliminar filas** donde algún activo no tenga dato (caso muy raro en este universo). Documentar el % de días eliminados (debe ser <0.5%).
- Cuidado especial: NVDA solo tiene historia con ese ticker tal cual desde su IPO 1999 → sin problemas para 2015–2025. NFLX cotiza desde 2002. Todos los activos cubren el periodo completo sin survivorship bias para este universo concreto (ver §7).

### 1.5 Volatilidad histórica realizada

**Sí, añadirla como feature**. Es prácticamente obligatoria en papers recientes:
- `realized_vol_5d = std(log_returns, 5) * sqrt(252)` (anualizada)
- `realized_vol_21d` (mensual)
- Opcionalmente, ratio `vol_5d / vol_21d` (régimen de volatilidad reciente vs medio plazo)

### 1.6 Decisiones específicas para TimeGAN

**Sobre qué entrenar TimeGAN:**

Hay dos alternativas viables y ambas se ven en literatura:

- **Opción A (recomendada como mínimo viable):** entrenar TimeGAN sobre **log-returns multivariados de los 6 activos + las 3 variables macro (VIX, ΔTNX, DXY return)** = ~9–12 dimensiones. Después, los indicadores técnicos se **recalculan determinísticamente** sobre las series de retornos sintéticas reconstruidas (acumulando para obtener "precios sintéticos"). Esto evita que TimeGAN tenga que aprender la relación funcional entre indicadores y precios (que es matemática y conocida), reduce dimensionalidad y mejora estabilidad.
- **Opción B (más ambiciosa):** entrenar TimeGAN sobre el vector completo de features ya normalizadas (~110–130 dims). Mucho más costoso y propenso a mode collapse. **No recomendado para mínimo viable.**

**Sequence length (ventana temporal del TimeGAN):**

- El paper original (Yoon et al. 2019, NeurIPS) usa `seq_len = 24` para datos de stock.
- ydata-synthetic mantiene ese default (24).
- En literatura financiera reciente, los rangos típicos son **20–60 días** para datos diarios. Recomendación: **`seq_len = 24`** (un mes hábil, alineado con horizonte de análisis de momento) como punto de partida; experimentar con 60 si hay tiempo (un trimestre).

**Normalización antes/después**: TimeGAN se entrena sobre features **ya normalizadas** a [0,1]. Después se invierte la transformación con el scaler persistido para volver al espacio de retornos.

---

## 2. Implementación de TimeGAN multivariado

### 2.1 Implementaciones de referencia (estado 2024–2026)

Tres opciones realistas para Python:

| Repositorio | Framework | Mantenimiento | Pros | Contras |
|---|---|---|---|---|
| **`jsyoon0823/TimeGAN`** (oficial) | TensorFlow 1 (con port a TF2 disponible) | Mantenimiento mínimo, código de referencia | Es el código del paper original; sirve para validar contra los benchmarks reportados | TF1 deprecado; requiere migración a TF2; código educativo, no production-ready |
| **`ydataai/ydata-synthetic`** | TensorFlow 2 / Keras | Activo (pero el repo principal `ydata-synthetic` ha tenido renombres recientes; alternativa actual: `Data-Centric-AI-Community/fg-data-synthetic`) | API de alto nivel: `TimeSeriesSynthesizer(modelname='timegan')`; ejemplo oficial con datos de Yahoo Finance; bien documentado | Menos control de bajo nivel; añade dependencia "pesada" |
| **`stefan-jansen/machine-learning-for-trading`** (cap. 21) | TensorFlow 2 | Actualizado | Port limpio del original a TF2; ejemplo financiero completo con metricas; libro de referencia (ml4trading.io) | Requiere adaptación |
| **`zwzhang123/TimeGAN-pytorch`** / **`benearnthof/TimeGAN`** | PyTorch | Comunitario, no oficial | PyTorch (preferible si el resto del stack es PyTorch + Stable-Baselines3) | Repos pequeños; benearnthof reporta dificultad de reproducir resultados originales |

**Recomendación**: dado que **Stable-Baselines3 es PyTorch**, ideal usar **PyTorch en TimeGAN** para no mezclar frameworks. Mejor opción práctica: partir del fork de `zwzhang123/TimeGAN-pytorch` y validar contra el oficial. Alternativa más cómoda y robusta: usar `ydata-synthetic` (TF2) **aislado en un módulo separado** (genera datos sintéticos, los persiste como Parquet, y luego el módulo PPO PyTorch los consume). El acoplamiento entre los dos frameworks es nulo si la comunicación es por archivos.

### 2.2 Hiperparámetros típicos

Del paper original y replicaciones (jsyoon0823, ydata-synthetic, gebob19.github.io, Yoon et al. NeurIPS 2019):

```
seq_len      = 24            # 24 días = ~1 mes hábil
hidden_dim   = 24            # igual a seq_len, default original
num_layers   = 3             # GRU layers en E, R, G, S
batch_size   = 128
lr           = 5e-4
epochs/iter  = 10000–50000   # con early stopping basado en discriminative score
gamma (GAN)  = 1             # peso del adversarial loss
eta (recon)  = 10            # peso reconstrucción típico
module       = 'gru'         # GRU > LSTM en estabilidad para este caso
noise_dim    = 32            # dimensión de z
```

Nota importante: el paper original entrena durante **50 000 iteraciones** que pueden tomar **>1 día en V100** según TimeVAE (arXiv 2111.08095). Para un TFE, 10 000–20 000 iteraciones con monitoreo de discriminative score suele ser suficiente.

### 2.3 Pitfalls comunes y mitigaciones

Documentados ampliamente (TimeVAE 2021, SeriesGAN arXiv 2410.21203, benearnthof README, ORNL 2024):

| Problema | Síntoma | Mitigación |
|---|---|---|
| **Mode collapse** | Generador produce siempre la misma "forma" (gancho, recta) | (i) Reducir lr; (ii) usar GRU en vez de LSTM; (iii) sustituir adversarial loss por **WGAN-GP**; (iv) aumentar diversidad del batch |
| **Inestabilidad de entrenamiento** | Losses oscilan, discriminator domina | Entrenamiento por etapas (embedding → supervised → joint) **es obligatorio**, no opcional. Es como lo implementa el paper. Saltarse la fase supervisada degrada calidad drásticamente |
| **Discriminator demasiado fuerte** | D loss → 0, G loss → ∞ | Reducir nº de updates de D por update de G; añadir noise a inputs de D |
| **Reproductibilidad pobre entre runs** | Varianza enorme entre seeds | Reportar 5–8 seeds; reportar **mejor modelo según validación discriminative score**, no el último |
| **Overfitting al periodo de entrenamiento** | Sintéticos que replican secuencias completas | Verificar que ninguna muestra sintética sea "casi-copia" de una real (Nearest Neighbor distance check) |

### 2.4 Métricas de evaluación de calidad de datos sintéticos

**Set mínimo a reportar (estándar del paper TimeGAN y todos los benchmarks posteriores):**

1. **Discriminative score** (TimeGAN paper, Yoon 2019): entrenar un clasificador GRU/LSTM de 2 capas para distinguir real vs sintético sobre 80% de los datos; reportar `|accuracy - 0.5|` en el 20% test. Cuanto más cerca de 0, mejor (clasificador no distingue). Valores razonables en literatura: 0.05–0.15 es bueno; <0.05 excelente.
2. **Predictive score** (TSTR — Train on Synthetic, Test on Real): entrenar GRU forecaster sobre sintéticos, evaluar MAE/MSE de predicción de step+1 en datos reales. Comparar contra el mismo modelo entrenado en reales. Diferencia pequeña ⇒ los sintéticos capturan dinámica predictiva.
3. **t-SNE / PCA visualización**: proyectar a 2D muestras reales vs sintéticas; deben solaparse visualmente.
4. **Hechos estilizados financieros** (Cont 2001; revisitados arXiv 2311.07738; SFAG arXiv 2601.12990) — **críticos para finanzas, no se reportan en TimeGAN genérico**:
   - **Distribución de retornos**: comparar histogramas, kurtosis (debe ser >3, fat tails) y asimetría
   - **Volatility clustering**: ACF de `|r_t|` (debe decaer lento, ley de potencia)
   - **Ausencia de autocorrelación de retornos**: ACF de `r_t` ≈ 0 a partir de lag 1
   - **Leverage effect**: `Corr(r_t, vol_{t+k})` debe ser negativo
   - **Cross-correlations** entre activos: comparar matrices de correlación real vs sintética (Frobenius distance)
   - Para multivariado: matrices de covarianza y autocovarianza cruzada

Reportar estas 4 categorías como tabla y figuras es lo que diferencia un análisis de TFE serio de uno superficial.

### 2.5 ¿Sigue siendo TimeGAN SOTA en 2024–2026?

**Respuesta breve: NO es SOTA, pero sigue siendo el estándar de comparación y suficiente para un mínimo viable.** Evidencia:

- **TimeVAE** (arXiv 2111.08095): VAE más estable y mucho más rápido de entrenar, resultados comparables o superiores en discriminative/predictive scores.
- **Quant GANs** (Wiese et al. 2020, *Quantitative Finance*): específicamente diseñado para finanzas; mejor reproducción de stylized facts univariados.
- **Modelos de difusión**: Diffusion-TS (ICLR 2024), TimeLDM (2024), TSDiff (MDPI Eng. Proc. 2024), DDPM con wavelet (Tandfonline 2025, Quantitative Finance) consistentemente superan a TimeGAN en discriminative score, y en finanzas reproducen mejor fat tails y autocorrelación de volatilidad.
- **SeriesGAN** (arXiv 2410.21203): mejora directa sobre TimeGAN (~30% mejor discriminative, ~39% predictive).

**Recomendación arquitectónica para el TFE**:
- **Mantener TimeGAN multivariado como modelo principal** (alineado con el enunciado del TFE y existe un cuerpo de literatura comparativa amplio).
- **(Opcional, alto valor académico)** añadir como segundo punto de comparación **un baseline de bootstrap** (circular block bootstrap, muy usado en Karzanov et al. 2025 *"Regret-Optimized Portfolio Enhancement"*) — esto demuestra que el valor del Agente B no viene solo de "más datos" sino del modelado generativo de TimeGAN.
- **(Si el tiempo lo permite)** discutir o ejecutar un secondary check con TimeVAE o un DDPM ligero, demostrando conocimiento del estado del arte.

### 2.6 Cuántos escenarios sintéticos generar — ratio óptimo

No hay consenso fuerte en la literatura, pero pautas:

- Karzanov et al. 2025 (AXA) entrenan **20 agentes independientes** y combinan datos reales con sintéticos generados por circular block bootstrap; el ratio efectivo de aumentación es ~1×–3× el dataset original.
- ORNL 2024 reporta que la calidad no mejora significativamente más allá de 1×–2× del dataset original.
- Si reales tienen N secuencias, generar **K × N sintéticas con K ∈ {1, 2, 3}**.

**Recomendación**: ratio **real:sintético = 1:2** (es decir, **dataset Agente B = 1 parte real + 2 partes sintético**, que sumadas son 3× el dataset original). Esto está dentro del rango eficaz reportado en literatura y mantiene el peso de los datos reales como ancla. **Hacer ablation con ratios 1:1, 1:2, 1:3** y reportar curva de Sharpe en validación vs ratio.

---

## 3. Implementación de PPO para Portfolio Management

### 3.1 Stack — librería recomendada

**Decisión (recomendación firme): Stable-Baselines3 (PPO) + Gymnasium + entorno custom propio.**

Justificación:

- **FinRL (`AI4Finance-Foundation/FinRL`)**: tiene `env_portfolio_allocation` que casi calza, pero hay **issues abiertos crónicos** con compatibilidad SB3 (e.g. issue #239 sobre `logger.record`, issue #1005 sobre `DummyVecEnv`), su código es educativo, los tests son escasos y en los FinRL Contests 2023–2025 los propios autores admiten que las DRL libraries ortodoxas (FinRL, RLlib, SB3) son insuficientes para producción y promueven *parallel market environments* propios (arXiv 2504.02281). Para un TFE, **FinRL es útil como referencia conceptual** (ver tutoriales `FinRL_PortfolioAllocation_NeurIPS_2020.ipynb` para entender state/action/reward), pero **no como dependencia central**.
- **RLlib**: sobredimensionado, curva de aprendizaje alta, integración compleja con entornos custom.
- **Stable-Baselines3 (DLR-RM)**: la opción estándar de la comunidad, PyTorch nativo, soporta Gymnasium >=0.28, API estable, documentación excelente, integración trivial con Optuna/W&B/MLflow. **Versión recomendada: SB3 >= 2.3.0** (requiere Python 3.10+).

**Para LSTM en política**: usar **`sb3-contrib.RecurrentPPO`** (PPO LSTM). Documentación: https://sb3-contrib.readthedocs.io/. Importante: los autores de SB3 advierten que **frame stacking** (con `VecFrameStack`) suele ser una alternativa más simple y competitiva que LSTM (ver report W&B SB3). Para un mínimo viable, **empezar con MlpPolicy + frame stacking de N=10–20 días** y solo escalar a `RecurrentPPO` si el desempeño es insuficiente.

### 3.2 Diseño del entorno (Gymnasium)

**Estructura mínima recomendada:**

```python
class PortfolioEnv(gym.Env):
    """
    State: (window, n_assets * n_features_per_asset + n_macro_features + n_assets [pesos actuales])
    Action: vector continuo de shape (n_assets,) ∈ [-1,1]; se proyecta a simplex
    Reward: log-return diferencial - costes de transacción - penalización opcional
    """
    observation_space = Box(low=-inf, high=+inf, shape=(window, total_features), dtype=float32)
    action_space      = Box(low=-1.0, high=+1.0, shape=(n_assets,), dtype=float32)  # n=6 (sin cash) o 7 (con cash)
```

**Sobre la proyección al simplex (clave):**

Tres patrones se ven en literatura:

1. **Softmax post-acción** (FinRL clásico): `weights = softmax(action)`. Simple, diferenciable, garantiza pesos en [0,1] que suman 1. **Recomendado por simplicidad.**
2. **Action ∈ [0,1]^n + normalización por suma**: `weights = action / sum(action)`. Problema: gradiente patológico cuando todo el vector es ~0.
3. **Tanh + proyección sobre simplex**: aplicar tanh para `[-1,1]`, después proyectar; ver `arXiv 2509.14385` (Adaptive and Regime-Aware RL) y "OpenReview SAPPO" (sentiment-augmented PPO).

**Recomendación**: empezar con **softmax** sobre `n_assets + 1` (incluyendo cash como activo virtual con retorno = tasa libre de riesgo diaria, opcionalmente 0 si se simplifica). Esto da al agente la opción de salirse del mercado en regímenes adversos.

**Sobre la reward function** (las opciones más usadas en papers 2023–2026):

| Reward | Forma | Ventajas | Desventajas |
|---|---|---|---|
| Log-return puro | `r_t = log(V_t/V_{t-1}) - cost_t` | Simple, aditivo en tiempo | No incorpora riesgo explícitamente |
| **Differential Sharpe** (Sood et al., adoptada en Karzanov 2025) | DSR de Moody-Saffell | Premia retornos suaves, penaliza volatilidad | Más complejo, requiere estados auxiliares A_t, B_t |
| **Log-return - β·drawdown** | `r_t - β·max(0, DD_t - α)` | Penaliza drawdowns >α explícitamente; usado en Wu et al., baseline en Karzanov 2025 | Hay que tunear α y β |
| Sharpe diferencial sobre ventana móvil | `(μ_W - r_f) / σ_W` | Aproximación a Sharpe; estable | Sesgo según W |

**Recomendación para mínimo viable**: **log-return con coste de transacción explícito**. Ablation opcional: añadir penalización por drawdown.

### 3.3 Costes de transacción y slippage

Práctica estándar (medium 2024 PPO+LSTM, opastpublishers 2024, Karzanov 2025, arXiv 2509.14385):

```
turnover_t = 0.5 * sum(|w_t - w_{t-1}_drifted|)   # turnover bilateral / 2
cost_t = turnover_t * (transaction_cost_pct + slippage_pct)
```

Valores realistas para activos USA equity líquidos a frecuencia diaria:

- Comisión: 0.01%–0.10% (1–10 bps). En papers académicos suele usarse **10 bps (0.1%)** como conservador (FinRL default es 0.1%, opastpublishers usa 0.1%, OpenReview SAPPO 0.01% por unit turnover, Karzanov 0.10–0.20%).
- Slippage diario para activos líquidos: 1–5 bps. Suele incluirse dentro del coste total.
- **Recomendación**: usar **0.10%** (10 bps) total por unit turnover, equivalente al estándar conservador. **Hacer un análisis de sensibilidad** con 0.05% y 0.20% para mostrar robustez.

`w_{t-1}_drifted`: los pesos drift después del retorno del día. Hay que recalcularlos antes de aplicar la nueva acción (la mayoría de implementaciones FinRL ignoran esto y aplican costes sobre `|w_t - w_{t-1}|` directamente, lo cual sobreestima costes en mercados con retornos altos). Implementar correctamente.

### 3.4 Hiperparámetros típicos de PPO

Defaults razonables de SB3 para activos financieros (combinando defaults SB3, RL Zoo, observaciones de medium PPO+LSTM, GitHub issue #1746 SB3, Karzanov 2025):

```python
PPO(
    policy="MlpPolicy",      # o "MultiInputPolicy" si dict obs
    learning_rate=3e-4,       # default SB3; range típico 1e-4 a 5e-4
    n_steps=2048,             # length de rollout; con datos diarios y horizonte total ~2500 días, considerar 1024
    batch_size=64,            # SB3 default; con ent_coef alto y env financieros, 128–256 funciona mejor
    n_epochs=10,              # default SB3; reducir a 4–6 si se observa overfitting al rollout
    gamma=0.99,               # alto = horizonte largo; con frecuencia diaria, podría reducirse a 0.95
    gae_lambda=0.95,          # default
    clip_range=0.2,           # default
    clip_range_vf=None,       # opcional, suele dejarse None
    ent_coef=0.01,            # CRÍTICO: default SB3 = 0.0; subir a 0.01-0.05 para evitar concentración en un activo
    vf_coef=0.5,
    max_grad_norm=0.5,
    target_kl=None,           # alternativa a clip_range; útil si entrenamiento inestable
    verbose=1,
    seed=42,
)
```

**Total timesteps**: depende de la longitud del dataset. Con ~2000 días train y `n_steps=1024`, se necesitan ~500–1000 rollouts → **500 000 a 1 000 000 timesteps**. Curva de aprendizaje suele estabilizarse antes de 500k.

### 3.5 Diseño de la red actor-critic

**MLP estándar (recomendado para empezar):**

```python
policy_kwargs = dict(
    net_arch=dict(pi=[128, 128], vf=[128, 128]),
    activation_fn=nn.Tanh,    # tanh es más estable que ReLU para PPO con observaciones financieras
    ortho_init=True,
)
```

Para **frame stacking de ventana W** (alternativa preferida a LSTM por simplicidad), envolver el VecEnv:

```python
from stable_baselines3.common.vec_env import VecFrameStack
vec_env = VecFrameStack(vec_env, n_stack=10)
```

Para **LSTM** (`sb3_contrib.RecurrentPPO`):

```python
policy_kwargs = dict(
    lstm_hidden_size=128,
    n_lstm_layers=1,
    enable_critic_lstm=True,
    net_arch=dict(pi=[64], vf=[64]),
)
```

**CNN-feature-extractor** (MDPI 2024 *"Systematic Approach"* mostró que CNN con lookback supera a MLP). Para un mínimo viable es overengineering, pero **mencionar como ablation** futura.

### 3.6 Window/lookback en estado

Dos enfoques compatibles con el state space:

- **Stateless con frame stacking**: observación = matriz (W × features) aplanada o procesada por CNN/MLP. **W = 20–60 días** es lo más común. Recomendado: **W = 30**.
- **Stateful con LSTM**: observación = features del día actual; LSTM mantiene memoria interna.

### 3.7 Problemas comunes de entrenamiento

| Problema | Diagnóstico | Solución |
|---|---|---|
| Agente concentra todo en 1 activo | Cabeza de softmax saturada; entropía → 0 | Subir `ent_coef` a 0.05–0.1; reducir `clip_range` a 0.1; aumentar `n_steps` |
| Agente siempre en cash | Reward negativa por costes domina cuando trade-off no compensa | Reducir penalización por costes (verificar bps); inicializar pesos cerca de equiponderado; añadir reward de exploración |
| Resultados altamente variables entre seeds | Inestabilidad estructural de PPO | Aumentar `n_steps`; reducir `learning_rate`; promediar 5–10 seeds; reportar IQM (interquartile mean) |
| Overfitting al training set | Sharpe train >> Sharpe val | Reducir `n_epochs`; activar early stopping basado en validación; aumentar `ent_coef` |
| Loss explode | Reward sin escalar (e.g. raw retornos en bps × 10000) | Escalar reward (multiplicar por 100 o 252) o usar `VecNormalize(norm_reward=True)` |

---

## 4. Metodología experimental y evaluación

### 4.1 División train/validation/test

Para un TFE con dataset relativamente pequeño (2015–2025 ≈ 2520 días) y objetivo de **comparar dos agentes**, no de entrenar el mejor modelo posible:

**Esquema recomendado (parsimonioso pero riguroso):**

- **Train**: 2015-01-01 a 2021-12-31 (~70%, ~1760 días) — incluye COVID crash de marzo 2020 dentro del train
- **Validation**: 2022-01-01 a 2022-12-31 (~10%, ~250 días) — incluye sell-off de 2022 (rates shock)
- **Test (out-of-sample, INTOCABLE)**: 2023-01-01 a 2025-04-30 (~20%, ~580 días) — período moderno, distinto del train

**Alternativa más rigurosa (walk-forward) si hay tiempo**: ver Liu et al. *"DRL for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting"* (NeurIPS 2022 / OpenReview) y arXiv 2010.08497 *"AAMDRL"*. Tres ventanas walk-forward consecutivas, retraining en cada una. Para TFE puede ser excesivo; mencionarlo como future work.

**Discusión sobre incluir COVID en train vs test**: hay dos escuelas:
- **Incluir COVID en train** (recomendado para este TFE): permite que el agente **aprenda** de un shock real, lo cual es lo que justifica el uso de TimeGAN para aumentar con escenarios sintéticos. El argumento del trabajo es: "TimeGAN ayuda a generalizar a regímenes raros". Si COVID ya está en train, hay que evaluar la robustez en el test (que tendrá su propia turbulencia: rates 2023, IA bubble fears 2024).
- **Dejar COVID en test**: prueba más dura de generalización, pero datos sintéticos solos no podrían cubrir un evento como COVID si no estaba en train (TimeGAN no extrapola).

**Recomendación: COVID en train**, e identificar **sub-períodos de estrés en test** para reportar métricas condicionadas (drawdown durante regímenes turbulentos en test).

### 4.2 Evitar data leakage

**Reglas de oro (Liu et al. 2022, MITRE 2023, blockchain-council 2024):**

1. **TimeGAN se entrena ÚNICAMENTE con datos de TRAIN** (2015-2021). Nunca con val o test.
2. **Scalers (MinMax, RobustScaler, etc.) se ajustan solo en TRAIN.** Persistir el scaler a disco.
3. **Indicadores técnicos** se calculan a partir de **datos del pasado solamente**. Cuidado especial con indicadores que requieren ventanas largas: descartar los primeros N filas (donde N = max(periodo de cualquier indicador)) **después** de calcularlos para no usar nans rellenados.
4. **No usar datos de val/test para tuning de hiperparámetros del agente final**. Usar solo train+val para tuning; reportar test una sola vez.
5. **Backtesting causal**: asegurar que la decisión en `t` solo usa info disponible al cierre de `t-1` (o al cierre de `t` si la ejecución se hace en `t+1`). El precio de ejecución debe ser el de apertura del día siguiente o el cierre del día t (si se asume liquidación instantánea al cierre).
6. **Para sintéticos**: las secuencias generadas por TimeGAN no tienen "fecha" — el riesgo de leakage existe si el conjunto de entrenamiento del agente B mezcla aleatoriamente reales (con fecha) y sintéticas (sin fecha). Solución: durante entrenamiento del PPO, samplear aleatoriamente entre dataset real y sintético, pero al evaluar **siempre usar el test real cronológico**.

### 4.3 Backtesting riguroso

- **Look-ahead bias en indicadores**: usar siempre `df['indicator'].shift(1)` antes de pasar al estado, garantizando que la observación en t solo depende de info hasta t-1. O bien, calcular indicadores y luego shiftear todo el dataframe.
- **Survivorship bias**: para el universo de este TFE (S&P 500 índice + NASDAQ 100 índice + 4 stocks que han existido y crecido durante todo el periodo), el survivorship bias está **acotado** pero presente: AAPL/AMZN/NVDA son **ganadoras conocidas ex-post**. Acknowledge esto explícitamente en la discusión.
- **Costes de transacción ignorados**: el error #1 en backtesting de DRL. Ya discutido en §3.3.
- **Sobreajuste al backtest** (Probability of Backtest Overfitting): Liu et al. 2022 propone un test de hipótesis explícito; reportar al menos que no se ha hecho cherry-picking de hiperparámetros sobre el test.

### 4.4 Métricas de desempeño

**Set mínimo a reportar para TFE comparativo (papers 2023–2026 estándar):**

| Métrica | Fórmula / Definición | Por qué |
|---|---|---|
| **Annualized return** | `(1+r̄)^252 - 1` | Baseline simple |
| **Annualized volatility** | `std(r) * sqrt(252)` | Riesgo simple |
| **Sharpe ratio** | `(r̄ - r_f) / σ * sqrt(252)` | Risk-adjusted return; principal métrica del TFE |
| **Sortino ratio** | Como Sharpe pero solo desviación negativa | Distinguir volatilidad "buena" de "mala" |
| **Maximum Drawdown (MDD)** | `min((V_t - max_{s≤t} V_s) / max_{s≤t} V_s)` | Peor caída peak-to-trough |
| **Calmar ratio** | `Annualized return / |MDD|` | Retorno por unidad de drawdown |
| **CVaR / Expected Shortfall** (95%) | Media de retornos diarios en peor 5% | Risk tail |
| **Turnover** | `mean(sum_assets |Δw|) anualizado` | Coste implícito |
| **Win rate** | % de días con retorno positivo | Estabilidad |
| **Recovery time** | Días desde MDD hasta nuevo máximo | Resiliencia |

**Métricas adicionales útiles para tu comparativa A vs B específicamente:**

- **Sharpe en sub-períodos** (split del test en regímenes alta/baja volatilidad): muestra si TimeGAN ayuda más en momentos turbulentos.
- **Stress test métricas**: Sharpe y MDD condicionados a `VIX > 25` en test.

### 4.5 Pruebas de estrés

Identificar a priori los siguientes sub-períodos en el test (post-2023):

- **Rates shock 2023 (mar–oct)**: SVB crash, regional banks
- **AI rally + correcciones 2024**: alta vol idiosincrática en NVDA
- **Sell-off de Q1 2025** (si aplicable según fecha actual del experimento)

Reportar tabla con métricas por sub-período.

### 4.6 Tests estadísticos

**Pregunta clave del TFE: ¿la diferencia A vs B es significativa?**

Solución estándar (Colas et al. 2018 *"How Many Random Seeds?"* arXiv 1806.08295; Colas 2019 *"Hitchhiker's Guide"* arXiv 1904.06979):

1. **Múltiples seeds**: entrenar **mínimo 5, idealmente 10 seeds** independientes para A y B (Henderson 2018, Colas 2018 demuestran que <5 seeds no permiten conclusiones estadísticas robustas). Karzanov 2025 entrena 20 agentes, lo cual es el ideal.
2. **Bootstrap confidence interval test**: bootstrap de las muestras de Sharpe/MDD entre seeds; si el IC de la diferencia (B-A) no contiene 0, la diferencia es significativa. Implementación en `scipy.stats.bootstrap`.
3. **Welch's t-test** sobre Sharpes de seeds (asumiendo aproximada normalidad, revisar con Shapiro-Wilk).
4. **Diebold-Mariano test** sobre **series temporales de retornos diarios** del test: compara si los errores predictivos (en este caso, los retornos) son significativamente diferentes. Implementación en `arch.bootstrap` o manual; aplicable cuando se evalúan dos estrategias sobre el mismo periodo.
5. Para múltiples comparaciones (más de dos métricas/sub-períodos): aplicar **Bonferroni** o **Benjamini-Hochberg** para controlar FDR.
6. **AdaStop** (arXiv 2306.10882, 2023): métodos modernos adaptativos para test de DRL; para un TFE es overkill, mencionarlo.

**Mínimo recomendado para este TFE:** 5 seeds × 2 agentes = 10 entrenamientos completos. Reportar **media ± std** en métricas, además de bootstrap CI 95% de la diferencia y p-value de Welch's t-test.

### 4.7 Número de seeds

Colas et al. 2018 (arXiv 1806.08295) muestra que para detectar un effect size moderado en RL hacen falta ~10 seeds para reducir falsos positivos a niveles aceptables. Para TFE: **5 seeds como mínimo absoluto, 10 como objetivo**. Usar siempre los mismos seeds para A y B (e.g. 0, 1, 42, 123, 1337) por reproducibilidad.

---

## 5. Stack tecnológico (versiones 2024–2026)

### 5.1 Versión de Python

**Python 3.11** es el sweet spot para 2024-2026:
- 3.10: soportado por todas las libs pero ya quedando antiguo
- 3.11: ~25% más rápido que 3.10 en mucho ML; soportado por PyTorch, TF, SB3 >= 2.3, ydata-synthetic
- 3.12: soporte mejorando, algunos paquetes aún tienen issues con compilación
- 3.13: demasiado bleeding-edge

**Recomendación: Python 3.11.x**.

### 5.2 PyTorch vs TensorFlow

- **PyTorch 2.x** para todo el pipeline DRL (SB3 lo requiere), si TimeGAN se reimplementa o se usa `zwzhang123/TimeGAN-pytorch`.
- **TensorFlow 2.x** únicamente si se usa `ydata-synthetic` para TimeGAN. En este caso, aislar TF en un entorno virtual separado o subprocess para evitar conflictos de CUDA.

**Recomendación**: **stack 100% PyTorch** si se tiene tiempo de adaptar la implementación TimeGAN. Si no, **TF2 solo para TimeGAN aislado, PyTorch para PPO**.

### 5.3 Datos: yfinance vs alternativas

**yfinance (estado 2025)**:
- Sigue siendo la opción gratuita más usada para datos diarios académicos.
- **Problemas conocidos**: no oficial (scraping); rate limiting agresivo; ocasionalmente cambia formato y rompe bibliotecas (suele tener fixes en pocos días); IP bans posibles (TildAlice 2025).
- **Para un TFE**: descargar **una vez** todos los datos históricos, **persistir a disco** (Parquet), no volver a llamar yfinance en runtime. Evita problemas de rate limit y reproducibilidad.

**Alternativas si yfinance falla:**

- **Alpha Vantage**: free tier muy limitado (25 requests/día en 2025; antes era 500/día). Útil como fallback complementario, no como fuente primaria.
- **Stooq.com** (vía `pandas_datareader.data.DataReader('SPX', 'stooq')`): backup gratuito robusto.
- **Tiingo**: 1000 requests/día free; calidad alta.
- **Finnhub**: 60 req/min en free tier (TildAlice 2025).
- **EOD Historical Data (EODHD)**: muy bueno para histórico, requiere subscripción.
- **Polygon.io** Developer ($29-79/mo): si alguna vez se necesita producción.

**Recomendación para este TFE**: **yfinance como primario** (descarga única, cache local), **Stooq como fallback**. Documentar la fecha exacta de descarga para reproducibilidad.

### 5.4 Indicadores técnicos

| Librería | Estado 2024–2026 | Recomendación |
|---|---|---|
| **TA-Lib** (C wrapper) | Establecida desde hace años; instalación dolorosa (compilación C); muy rápida; 200+ indicadores | Usar si la instalación funciona; performance superior |
| **`pandas-ta`** (Python puro, +numba) | **Activa pero bajo riesgo** — la propia web pandas-ta.dev advierte: "Current levels are unsustainable and risks discontinuation"; ~150 indicadores | Más fácil de instalar; lentes para datasets grandes; **alternativa por defecto** |
| **`ta`** (Bukosabino) | Mantenida; menos indicadores que pandas-ta | Alternativa minimalista |

**Recomendación**: usar **pandas-ta** como librería primaria por simplicidad de instalación y suficiencia para los ~10 indicadores requeridos. Si se requiere TA-Lib (compatibilidad TradingView exacta), usar wheels precompilados (`pip install TA-Lib` con wheels disponibles para Python 3.11 desde 2024).

### 5.5 Stable-Baselines3 y Gymnasium

- **Stable-Baselines3 >= 2.3.0** (publicada en 2024) — requiere `gymnasium >= 0.28`, Python >= 3.10.
- Compatibilidad con `gym` antiguo (el de OpenAI clásico) **eliminada**; usar siempre `import gymnasium as gym`.
- **`sb3-contrib`** (mismo repo team): para `RecurrentPPO`. Versión emparejada con SB3.

### 5.6 FinRL — usar o no usar

**Veredicto: NO usar FinRL como dependencia central, SÍ usarlo como referencia.**

- El repo principal `AI4Finance-Foundation/FinRL` está activo pero con muchos issues abiertos sobre compatibilidad SB3 y errores en paths del config (issue #239, #1005).
- El código educativo (`FinRL_PortfolioAllocation_NeurIPS_2020.ipynb`) sirve como inspiración para diseñar tu propio entorno de portfolio.
- **FinRL Contests 2023-2025** han evolucionado hacia entornos paralelos custom propios, alejándose de SB3.
- Su `INDICATORS` por defecto (MACD, RSI, CCI, ADX, BBANDS) es exactamente el set core que recomendamos en §1.1 — confirma la convergencia.

### 5.7 Reproducibilidad y experiment tracking

**Stack recomendado:**

- **MLflow**: tracking local de experimentos (parámetros, métricas, artefactos). Suficiente para TFE. Alternativa: **W&B** (cloud, free para uso académico).
- **Hydra** (`hydra-core`): config management con YAML jerárquicos y override en CLI. Patrón muy adoptado en proyectos ML 2024+ (lightning-hydra-template, hydraflow). Permite ejecuciones tipo `python train.py model=ppo agent=A seed=42`.
- **OmegaConf**: dependencia de Hydra; permite resoluciones dinámicas.
- **DVC** (`dvc`): control de versiones de datos. Útil pero **opcional para TFE**.
- **Seeds**: fijar `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, y `gym.utils.seeding`. SB3 expone `seed` en el constructor; respetar también `vec_env.seed(seed)`.

### 5.8 Docker

**Recomendación**: crear un `Dockerfile` minimal con la imagen base `python:3.11-slim` o `pytorch/pytorch:2.x-cuda12.1-cudnn8-runtime` para reproducibilidad. Es buena práctica académica y se valora en la defensa del TFE. Imagen tipo `nvidia/cuda:12.1.1-runtime-ubuntu22.04` si se necesita GPU.

### 5.9 Cómputo

**Estimación realista:**

- **TimeGAN multivariado** (~9-12 dims, seq_len=24, 20k iter): **~1–4 horas en GPU T4 / Colab Free**. CPU: 12+ horas, factible pero tedioso.
- **PPO** (500k–1M timesteps, MLP, env financiero diario, 6 activos): **~1–2 horas en GPU**, **~3–6 horas en CPU 8-core**. PPO es CPU-bound en gran medida (rollouts del env), GPU acelera el net forward/backward pero no es 10× speedup.
- **5–10 seeds × 2 agentes** = 10–20 ejecuciones de PPO ⇒ **20–40 horas total** distribuible.

**Recomendación de hardware:**

- **Google Colab Free** (T4): sí es suficiente, pero con cuidado por timeouts (12h max por sesión; 90 min idle). Se necesitan checkpoints persistentes en Drive.
- **Google Colab Pro** ($10/mes): A100 o L4, sesiones más largas, mucho mejor experiencia.
- **Kaggle Notebooks** (T4 x2 free): alternativa muy buena, 30h/semana de GPU.
- **Local con GPU consumer (RTX 3060+)**: ideal si está disponible.

Para un TFE bien planificado: **Colab Pro** o **Kaggle** son suficientes.

---

## 6. Estructura del proyecto y buenas prácticas

### 6.1 Organización de directorios (recomendación)

Patrón inspirado en `lightning-hydra-template`, FinRL y Cookiecutter Data Science:

```
tfe-drl-portfolio/
├── README.md
├── pyproject.toml             # poetry o setup.cfg
├── requirements.txt
├── Dockerfile
├── .env.example               # API keys (alpha vantage, etc.)
├── .gitignore                 # excluir data/raw, models/, mlruns/
│
├── configs/                   # Hydra
│   ├── config.yaml            # main
│   ├── data/
│   │   ├── default.yaml       # tickers, fechas, indicadores
│   │   └── extended.yaml
│   ├── timegan/
│   │   ├── default.yaml       # seq_len=24, hidden_dim=24, etc.
│   │   └── tuned.yaml
│   ├── ppo/
│   │   ├── default.yaml
│   │   └── recurrent.yaml
│   ├── env/
│   │   ├── portfolio_default.yaml
│   │   └── portfolio_with_cash.yaml
│   └── experiment/
│       ├── agent_a.yaml       # PPO solo real
│       └── agent_b.yaml       # PPO real + sintético
│
├── data/
│   ├── raw/                   # CSV/Parquet originales de yfinance
│   ├── processed/             # con indicadores e imputaciones
│   ├── synthetic/             # outputs de TimeGAN (Parquet con seed encoded)
│   └── splits/                # train/val/test indices guardados
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── download.py        # yfinance + Stooq fallback
│   │   ├── features.py        # indicadores técnicos, normalización
│   │   ├── splits.py          # train/val/test cronológico
│   │   └── alignment.py       # calendarios bursátiles
│   ├── generative/
│   │   ├── timegan/
│   │   │   ├── model.py       # E, R, G, S, D
│   │   │   ├── train.py       # 3 fases: embedding → supervised → joint
│   │   │   ├── generate.py
│   │   │   └── metrics.py     # discriminative_score, predictive_score, stylized_facts
│   │   └── baselines/
│   │       └── block_bootstrap.py  # baseline conservador
│   ├── envs/
│   │   ├── portfolio_env.py   # Gymnasium env custom
│   │   ├── transaction_costs.py
│   │   └── rewards.py
│   ├── agents/
│   │   ├── ppo_agent.py       # wrapper SB3
│   │   └── callbacks.py       # logging custom
│   ├── eval/
│   │   ├── backtest.py
│   │   ├── metrics.py         # sharpe, sortino, mdd, calmar, cvar, turnover, etc.
│   │   ├── stat_tests.py      # bootstrap, welch t-test, Diebold-Mariano
│   │   └── plots.py           # equity curves, drawdown, distrib retornos
│   └── utils/
│       ├── seeding.py
│       └── logging_utils.py
│
├── scripts/
│   ├── 01_download_data.py
│   ├── 02_build_features.py
│   ├── 03_train_timegan.py
│   ├── 04_evaluate_synthetic.py
│   ├── 05_train_agent.py      # parametrizable A o B vía Hydra
│   ├── 06_backtest_agent.py
│   └── 07_compare_agents.py   # tests estadísticos finales
│
├── notebooks/                 # exploratorios, no para pipeline
│   ├── 01_eda.ipynb
│   ├── 02_synthetic_quality.ipynb
│   └── 03_results_analysis.ipynb
│
├── tests/                     # pytest
│   ├── test_features.py
│   ├── test_env.py            # tests del Gymnasium env (clave!)
│   ├── test_metrics.py
│   └── test_no_leakage.py     # tests específicos anti-leakage
│
└── outputs/                   # outputs Hydra (generados)
    ├── timegan_runs/
    ├── ppo_runs/
    └── reports/
```

### 6.2 Separación módulo generativo / módulo RL

**Acoplamiento por archivos** (no por imports). El módulo TimeGAN escribe a `data/synthetic/<run_id>/` archivos Parquet con la misma columna-schema que `data/processed/`. El módulo PPO los lee igual que los reales. Permite:
- Cambiar el modelo generativo sin tocar el agente.
- Re-correr solo PPO sin re-entrenar TimeGAN.
- Experimentar con bootstrap baseline reusando la pipeline.

### 6.3 Pipeline de datos

- **Storage format**: **Parquet** (mucho más rápido que CSV, tipos nativos, compresión). Para ~2500 días × 6 activos × ~30 features, el dataset cabe en <50 MB Parquet.
- **No usar HDF5** para datasets pequeños como este: complejidad innecesaria.
- **CSV solo para**: configs visuales, outputs human-readable de tabla de resultados.

### 6.4 Configuración

**Hydra + OmegaConf** con composición jerárquica. Permite:

```bash
# Train Agent A con config default
python scripts/05_train_agent.py experiment=agent_a seed=42

# Override desde CLI
python scripts/05_train_agent.py experiment=agent_b ppo.learning_rate=1e-4 seed=123

# Multirun (sweeps)
python scripts/05_train_agent.py -m experiment=agent_a,agent_b seed=0,1,42,123,1337
```

### 6.5 Logging y experiment tracking

Patrón estándar:

- **MLflow**: log de params, metrics por step (training reward, loss), metrics finales (Sharpe test, MDD test, etc.), artefactos (modelos, gráficos, tablas).
- **Tensorboard**: integrado en SB3 vía `tensorboard_log=` en el constructor PPO.
- **HydraFlow** (https://github.com/daizutabi/hydraflow): integración Hydra+MLflow tipada.

### 6.6 Tests unitarios mínimos críticos

Para un proyecto de DRL financiero el conjunto mínimo es:

1. **`test_features.py::test_no_lookahead`**: verifica que un indicador en t solo depende de datos en [0, t].
2. **`test_env.py::test_action_space`**: pesos siempre suman 1, son ≥0.
3. **`test_env.py::test_reward_consistency`**: log-return acumulado coincide con `(V_T - V_0) / V_0` ≈ `exp(sum(log_returns)) - 1`.
4. **`test_env.py::test_transaction_costs`**: costes son no-negativos; cero turnover ⇒ cero coste.
5. **`test_no_leakage.py::test_scaler_only_train`**: el scaler tiene `data_min_` calculado solo sobre fechas de train.
6. **`test_no_leakage.py::test_timegan_train_only`**: los datos pasados a `TimeGAN.fit` son siempre subset de train.
7. **`test_metrics.py::test_sharpe_ratio`**: contra valores conocidos (e.g. retornos constantes ⇒ Sharpe = inf si σ=0; retornos N(μ, σ) ⇒ Sharpe ≈ μ/σ × sqrt(252)).
8. **`test_env.py::test_gymnasium_compliance`**: usar `gymnasium.utils.env_checker.check_env` para validar la API.

Nota: estos tests son **tan importantes** como los resultados experimentales. Detectan los errores que invalidan TFEs.

---

## 7. Pitfalls y riesgos conocidos

### 7.1 Errores típicos que invalidan resultados de DRL en backtesting

Recopilación basada en Liu et al. (NeurIPS 2022), MITRE 2023, blockchain-council 2024, AAMDRL arXiv 2010.08497:

1. **Look-ahead bias en indicadores**: el más común. Resolver con `.shift(1)` consistente y tests unitarios.
2. **Mismatch entre acción y precio de ejecución**: agente decide en t basándose en cierre de t, pero ejecuta al cierre de t (ya pasado). Realista: ejecutar a cierre de t (asumiendo decisión a precio de cierre) o apertura de t+1.
3. **No incorporar costes**: ya discutido (§3.3).
4. **Re-entrenar mirando el test**: ajustar hiperparámetros del PPO viendo Sharpe en test → overfitting al test. Usar val para tuning.
5. **Cherry-picking de seed**: reportar solo el seed con mejores resultados. Reportar **siempre todos los seeds + estadística de población**.
6. **Survivorship bias** (ver 7.2).
7. **Comparación injusta**: A entrenado con 1M timesteps, B con 3M timesteps. Igualar **timesteps efectivos** o **número de épocas sobre datos únicos**.
8. **Backtest sobre train**: reportar resultados sobre train no aporta nada; siempre out-of-sample.

### 7.2 Survivorship bias

Para tu universo (S&P 500 índice, NASDAQ-100 índice, AAPL/AMZN/NFLX/NVDA):

- Los **índices** S&P 500 y NASDAQ-100 ya están sujetos a survivorship bias por construcción (rebalanceos eliminan fracasos), pero esto es estructural y aceptable.
- Los **stocks individuales** son ganadores ex-post conocidos. AAPL y NVDA especialmente han tenido rendimientos extraordinarios 2015-2025.
- **Cómo mitigar / acknowledge**:
  - Reconocerlo explícitamente en limitaciones del TFE.
  - **Comparar contra benchmarks pasivos**: equiponderado de los 6 activos, S&P 500 buy-and-hold, 60/40 stocks/bonds. Si el agente DRL no supera el equiponderado, no aporta valor.
  - Idealmente, validar la metodología en un universo "más justo" como un futuro experimento (e.g. Dow 30 con sobreviviente y deslistadas, Norgate Data).

### 7.3 Look-ahead en indicadores específicos

Casos sutiles a vigilar:

- **MACD en t** depende de EMA(12) y EMA(26) que sí son causales si se calculan secuencialmente. Pero algunos cálculos vectorizados pueden usar valores futuros si no se hace bien. Verificar con `pandas-ta` que ya lo hace correctamente.
- **Bollinger Bands**: media móvil + std móvil; ambos causales si se usa `.rolling(20).mean()` sin `.center=True`.
- **Normalización con z-score global**: usa media y std del dataset COMPLETO, lo cual incluye train+val+test ⇒ leakage. Usar **z-score con ventana móvil** o **z-score con stats solo de train**.

### 7.4 Costes de transacción ignorados — impacto

Karzanov et al. 2025 (AXA) muestra que un PPO sin scheduler de costes converge a estrategias de turnover alto con reward "ficticia". Cuando se aplican costes reales en evaluación, el alfa desaparece. **Siempre incluir costes en train Y en test.**

### 7.5 Asegurar que la mejora del Agente B es real

**Checklist anti-falsa-mejora:**

1. ✅ Agente A y B entrenados con **mismos hiperparámetros, misma arquitectura, mismos seeds**, diferenciándose **solo en el dataset de entrenamiento**.
2. ✅ Agente B con dataset de tamaño 3× ⇒ recibe 3× timesteps por época si se itera sobre todo el dataset; **igualar timesteps totales**, no épocas.
3. ✅ Comparar contra **baseline conservador (block bootstrap)**: si Agente B (TimeGAN) supera a Agente A (real) **pero también supera a Agente B' (real + bootstrap)**, entonces TimeGAN aporta valor más allá de "más datos".
4. ✅ Reportar **5–10 seeds**, bootstrap CI 95%, Welch's t-test.
5. ✅ Verificar que la mejora es **persistente en sub-períodos** del test, no concentrada en uno solo (potencial accidente).
6. ✅ Verificar que **TimeGAN pasa las métricas de calidad** del §2.4. Si TimeGAN genera basura, los datos sintéticos no pueden ayudar y cualquier "mejora" del Agente B es ruido.

### 7.6 ¿Cuándo la aumentación con sintéticos puede empeorar el desempeño?

Casos documentados (Hindawi DAuGAN 2021, ORNL 2024, ScienceDirect Pena 2022, ICAIF SDARL4T):

1. **TimeGAN entrenado con datos insuficientes**: si N_train < ~500-1000 secuencias, TimeGAN colapsa o genera artefactos. Para tu TFE con seq_len=24 y 1760 días train ⇒ ~1736 secuencias overlapping (rolling), debería bastar.
2. **Mode collapse**: sintéticos con poca diversidad ⇒ agente B se sobreajusta a un régimen estrecho.
3. **Distributional shift de sintéticos**: si los sintéticos no cubren el régimen del test, no ayudan a generalizar y pueden degradar (covariate shift).
4. **Ratio real:sintético muy alto** (e.g. 1:10): el agente aprende dinámicas sintéticas alejadas de las reales.
5. **Sintéticos sin stylized facts**: TimeGAN típico no captura bien fat tails. Si el agente B se entrena con retornos demasiado gaussianos, su risk management será inadecuado para la cola real.
6. **Costes de transacción inflados en sintéticos**: si los sintéticos tienen turnover artificial alto por ruido, el agente aprende a hacer trades excesivos.

**Mitigación**: monitorear las métricas de calidad de §2.4 antes de usar los sintéticos para entrenar PPO. Si discriminative_score > 0.3 o stylized facts no se reproducen, **no usar esos sintéticos**.

---

## 8. Recomendaciones finales accionables

Para construir el prompt de Claude Cowork, las decisiones arquitectónicas concretas a fijar son:

| Dimensión | Decisión recomendada |
|---|---|
| Activos | 6 (S&P 500, NASDAQ-100, AAPL, AMZN, NFLX, NVDA) + cash virtual con r_f≈0 |
| Periodo | 2015-01-01 → 2025-04-30 |
| Frecuencia | Diaria, calendario XNYS |
| Train/Val/Test | 70/10/20 cronológico (COVID en train, sub-períodos de estrés en test) |
| Features OHLCV | Returns + body/range + log_volume normalizado (no precios crudos) |
| Indicadores | MACD(12,26,9), RSI(14), BBANDS(20,2), CCI(14), ADX(14), EMA(10,30), ATR(14), OBV, RealizedVol(5,21) |
| Macro | log(VIX), ΔVIX, ΔTNX (bps), DXY log-return; replicados global |
| Normalización | MinMaxScaler solo para TimeGAN; RobustScaler con stats train para PPO state |
| Scaler fitting | Solo en train; persistir con joblib |
| TimeGAN | Multivariado sobre log-returns 6 activos + 3 macro = 9 dims; seq_len=24; PyTorch (zwzhang123 fork) o ydata-synthetic TF2 aislado |
| TimeGAN HP | hidden_dim=24, num_layers=3, lr=5e-4, batch=128, iter=20000 |
| Sintéticos | Ratio real:sintético = 1:2 (ablation 1:1, 1:3) |
| Calidad sintético | Discriminative score < 0.15, predictive score gap < 10%, stylized facts (fat tails, volatility clustering, ACF returns ≈ 0, leverage effect, cross-corr ✓) |
| RL Library | Stable-Baselines3 >= 2.3.0 (PPO MlpPolicy + VecFrameStack n=10) |
| Action space | `Box(-1, 1, (n_assets+1,))` con softmax post-hoc |
| Reward | log-return diferencial - costes 0.10% × turnover |
| State window | 30 días (frame stacking n=10–30) |
| PPO HP | lr=3e-4, n_steps=1024, batch=128, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip=0.2, **ent_coef=0.01** (clave), vf_coef=0.5 |
| Total timesteps | 1M por seed |
| Seeds | 5 mínimo (objetivo 10): {0, 1, 42, 123, 1337, 7, 31, 99, 256, 2024} |
| Métricas | Sharpe, Sortino, Calmar, MDD, CVaR-95%, Turnover, Win Rate, Recovery Time + sub-períodos estrés |
| Stat tests | Bootstrap CI 95% sobre Sharpe(B)−Sharpe(A), Welch t-test, Diebold-Mariano sobre series de retornos |
| Tracking | MLflow + Hydra |
| Reproducibilidad | Docker + seeds + config Hydra + Parquet versionado + scaler joblib |
| Python | 3.11 |
| Hardware | Colab Pro / Kaggle / GPU local; ~30-40h cómputo total |

### Estructura de comparación principal del TFE

```
                     | Agente A      | Agente B (TimeGAN)
Dataset entrenamiento|  Real (train) | Real (train) + 2× Sintético (TimeGAN solo train)
Mismos HP de PPO     |  ✓            | ✓
Mismos seeds         |  ✓            | ✓
Mismos timesteps     |  ✓            | ✓
Evaluación           |  Test real    | Test real
                     |  (mismo split)| (mismo split)
```

**Baseline adicional crítico**: Agente B' = PPO entrenado con Real (train) + 2× Bootstrap (circular block bootstrap, block_size=20). Si A < B' < B, TimeGAN aporta valor real más allá del bootstrap. Si B ≈ B', el aumento se debe simplemente a "más datos diversos", no a la modelización generativa de TimeGAN — resultado igualmente publicable y honesto.

### Repositorios de referencia clave para Claude Cowork

- `DLR-RM/stable-baselines3` — PPO base
- `Stable-Baselines-Team/stable-baselines3-contrib` — RecurrentPPO si se necesita LSTM
- `jsyoon0823/TimeGAN` — referencia canónica TimeGAN (TF1 original)
- `zwzhang123/TimeGAN-pytorch` — port PyTorch
- `ydataai/ydata-synthetic` (o fork actual `Data-Centric-AI-Community/fg-data-synthetic`) — TimeGAN high-level TF2
- `stefan-jansen/machine-learning-for-trading` cap. 21 — implementación financiera TimeGAN TF2
- `AI4Finance-Foundation/FinRL` — referencia conceptual env portfolio (no usar como dependencia)
- `flowersteam/rl-difference-testing` (Colas et al. 2018) — código para tests estadísticos
- `ashleve/lightning-hydra-template` — template de proyecto Hydra
- `daizutabi/hydraflow` — integración Hydra+MLflow

### Papers clave a citar en el TFE

- Yoon, Jarrett, van der Schaar (2019) *"Time-series Generative Adversarial Networks"*, NeurIPS
- Schulman et al. (2017) *"Proximal Policy Optimization Algorithms"*, arXiv 1707.06347
- Liu, X.-Y. et al. (2020) *"FinRL: A DRL Library for Automated Stock Trading"*, NeurIPS Deep RL Workshop
- Liu, X.-Y. et al. (2022) *"DRL for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting"*, ICAIF/AAAI'23 — metodología CPCV
- Karzanov et al. (2025) *"Regret-Optimized Portfolio Enhancement through DRL and Future Looking Rewards"*, arXiv 2502.02619 — uso de sintéticos vía bootstrap, transaction cost scheduler
- Wiese et al. (2020) *"Quant GANs: Deep Generation of Financial Time Series"*, *Quantitative Finance* 20(9)
- Colas, Sigaud, Oudeyer (2018) *"How Many Random Seeds? Statistical Power Analysis in Deep RL"*, arXiv 1806.08295
- Henderson et al. (2018) *"Deep Reinforcement Learning that Matters"*, AAAI
- Cont, R. (2001) *"Empirical properties of asset returns: stylized facts and statistical issues"*, *Quantitative Finance* (clásico para validación de stylized facts)
- arXiv 2511.17963 (2024) *"Hybrid LSTM and PPO Networks for Dynamic Portfolio Optimization"* — referencia reciente comparable
- MDPI *Algorithms* 17(12), 570 (2024) *"A Systematic Approach to Portfolio Optimization: Comparative Study of RL Agents"* — comparativa exhaustiva DQN/DDPG/PPO/SAC
- arXiv 2503.04143 (2025) *"MTS: A DRL Portfolio Management Framework with Time-Awareness and Short-Selling"*
- ACM ICAIF *"Synthetic Data Augmentation for DRL in Financial Trading"* (SDARL4T, 2022) — referencia directa para tu motivación

### Aspectos que distinguirán este TFE de un trabajo mediocre

1. **Tests estadísticos rigurosos** (no solo "B > A en mi seed favorito").
2. **Baseline de bootstrap** comparativo además del estándar B vs A.
3. **Validación de calidad de sintéticos** con stylized facts financieros, no solo discriminative score genérico.
4. **Análisis por sub-períodos de estrés** en el test, no solo aggregado.
5. **Tests unitarios anti-leakage** explícitamente verificados y documentados.
6. **Acknowledge explícito** del survivorship bias y otras limitaciones.
7. **Reproducibilidad total**: Docker, seeds, configs Hydra versionados.

Esta combinación es lo que separa un TFE técnico sólido de los muchos trabajos publicados con resultados infladas que no se reproducen.