# Plan de Implementación – Fase 2 (Semanas 7–15)

**TFE Máster IA UNIR** — *Evaluación de la Robustez en Inversiones mediante Aprendizaje por Refuerzo Profundo y Escenarios Generados en Mercados Financieros*

**Equipo**: Diegbys Mudarra y Andrés Pereira
**Director**: Eduardo Muñoz Lorenzo
**Tipología**: Comparativa de soluciones (Tipo 4 cronograma UNIR)
**Versión**: 1.0 – Plan didáctico + ejecutable para Claude Code

---

## Cómo leer este plan

Este documento organiza el desarrollo de la Fase 2 del TFE en **9 fases del ciclo de vida estándar de un proyecto de Machine Learning**, no por semanas del cronograma UNIR. Las semanas UNIR se mapean a las fases ML al final del documento (sección **Plan de ejecución sugerido**).

Cada fase tiene esta estructura fija:

1. **¿Qué es esta fase y por qué importa?** — explicación general aplicable a cualquier proyecto de ML, sin asumir conocimiento previo.
2. **¿Cómo se aplica a nuestro TFE?** — conexión específica con las decisiones del reporte y la memoria.
3. **Conceptos clave que vamos a usar** — definiciones cortas de términos ML y finanzas (solo la primera vez que aparecen).
4. **Tareas (todos)** — tabla con ID, descripción, archivo destino, criterio de aceptación, dependencias y esfuerzo.
5. **Riesgos / pitfalls específicos** — los más críticos del reporte aplicados a la fase.
6. **Mapeo al cronograma UNIR** — en qué semana(s) cae y qué entrega se ve afectada.

**Convenciones**:

- IDs de tarea: `F<fase>-T<n>` (p. ej. `F1-T1`, `F4-T7`).
- Esfuerzo: **S** = pocas horas (<4h), **M** = un día (4–8h), **L** = varios días (>8h, o dependencia de cómputo).
- "Owner" se deja sin asignar — el reparto formal entre Diegbys y Andrés (Tabla 1 de la memoria) se aplica al ejecutar.
- Cada tarea está pensada para convertirse en un prompt de Claude Code. El criterio de aceptación es el "definition of done" verificable.

---

## Mapa Fase ML → Semana UNIR (vista rápida)

| Fase ML | Nombre | Semanas UNIR | Entrega afectada |
|---|---|---|---|
| 1 | Adquisición de datos | Sem. 7 | Borrador Intermedio (Sem. 9) |
| 2 | EDA y entendimiento del dominio | Sem. 7 | Borrador Intermedio |
| 3 | Preprocesamiento e ingeniería de features | Sem. 7–8 | Borrador Intermedio |
| 4 | Modelado generativo (TimeGAN) | Sem. 8–9 | Borrador Intermedio |
| 5 | Diseño del entorno de trading (Gymnasium) | Sem. 8 | Borrador Intermedio |
| 6 | Entrenamiento PPO – Agentes A y B | Sem. 10–12 | Borrador Final |
| 7 | Backtesting y métricas | Sem. 12–13 | Borrador Final (Cap. 5–6) |
| 8 | Análisis estadístico comparativo | Sem. 13 | Borrador Final (Cap. 6) |
| 9 | Empaquetado, reproducibilidad y documentación | Sem. 14–15 | Borrador Final |

> El Borrador Intermedio (Sem. 9) no exige resultados finales: basta con tener **el pipeline de datos + TimeGAN funcionando con primeros sintéticos validados + entorno Gymnasium probado** y poder mostrar resultados preliminares de un primer entrenamiento de PPO. La comparativa rigurosa (5 seeds × 2 agentes + tests estadísticos) cae en el Borrador Final.

---

## Glosario inicial — términos que usaremos repetidamente

**Términos de Machine Learning** (se asumen conocidos a partir de aquí):

- **Hiperparámetro**: parámetro que se fija *antes* de entrenar (lr, batch size, número de capas). No se aprende de los datos.
- **Overfitting**: el modelo memoriza los datos de entrenamiento y falla en datos nuevos. El enemigo número uno en este TFE.
- **Data leakage**: información del conjunto de test "se filtra" al entrenamiento. Invalida resultados. Veremos formas concretas en cada fase.
- **Walk-forward**: validación cronológica donde se entrena con `[t0, t1]` y se valida con `[t1, t2]` sin mirar al futuro.
- **Mode collapse**: patología típica de GANs donde el generador produce siempre la misma "forma" de salida, perdiendo diversidad.
- **Seed**: semilla aleatoria. Fijarla permite reproducibilidad. En RL se reportan múltiples seeds para tener significancia estadística.

**Términos financieros** (se asumen conocidos a partir de aquí):

- **Log-return**: `r_t = log(P_t / P_{t-1})`. Aditivo en el tiempo y aproximadamente gaussiano. Lo usaremos en todo el pipeline.
- **Sharpe ratio**: `(r̄ - r_f) / σ × √252`. Retorno por unidad de riesgo. Métrica principal del TFE.
- **Sortino ratio**: como Sharpe pero solo penaliza la desviación negativa (volatilidad "mala").
- **Maximum Drawdown (MDD)**: peor caída desde un máximo histórico hasta el siguiente mínimo. Mide riesgo de pérdida real.
- **Calmar ratio**: `Annualized Return / |MDD|`. Retorno por unidad de drawdown.
- **CVaR (Expected Shortfall) 95%**: media de los retornos del 5% peor. Mide riesgo de cola.
- **Volatility clustering**: hechos estilizados según los cuales periodos de alta volatilidad tienden a agruparse.
- **Fat tails (colas pesadas)**: las distribuciones de retornos reales tienen kurtosis > 3 (más eventos extremos que una gaussiana).
- **Leverage effect**: correlación negativa entre retorno actual y volatilidad futura (cuando el mercado cae, la volatilidad sube).
- **Stylized facts**: propiedades empíricas universales de los retornos financieros (Cont 2001). TimeGAN debe reproducirlas.
- **Survivorship bias**: usar solo activos que "sobrevivieron" inflate los resultados. AAPL/NVDA son ganadoras ex-post; lo reconoceremos como limitación.
- **Turnover**: cuánto cambia la cartera entre `t-1` y `t`. Genera costes de transacción.
- **VIX / TNX / DXY**: VIX = índice de volatilidad implícita del S&P 500; TNX = yield del treasury 10Y; DXY = índice del dólar. Variables macro de contexto.

---

## Fase 1: Adquisición de datos

### ¿Qué es esta fase y por qué importa?

En cualquier proyecto de ML, la fase de adquisición consiste en obtener la materia prima del modelo: los datos. Aunque suene trivial, es la fase con mayor impacto a largo plazo, porque cualquier error introducido aquí (datos faltantes, fechas mal alineadas, columnas mezcladas) se propaga silenciosamente y solo aparece como "modelo raro" semanas más tarde. Una buena práctica universal es **descargar una sola vez, persistir en formato eficiente y nunca volver a llamar al origen externo en runtime**, garantizando reproducibilidad y aislándonos de cambios o caídas del proveedor.

### ¿Cómo se aplica a nuestro TFE?

Necesitamos descargar series diarias de **6 activos USA** (S&P 500, NASDAQ-100, AAPL, AMZN, NFLX, NVDA) más **3 variables macro** (VIX, ^TNX, DXY) en el período **2015-01-01 a 2025-04-30** desde **yfinance**. El reporte (§5.3) recomienda persistir todo en **Parquet local** y usar **Stooq como fallback**. Como nuestros activos cotizan todos en NYSE/NASDAQ, el calendario es común (XNYS); esto simplifica la alineación. La salida de esta fase es un dataset único, fechado, validado e inmutable que alimenta todas las fases siguientes.

### Conceptos clave que vamos a usar

- **Parquet**: formato columnar binario; mucho más rápido que CSV y conserva tipos nativos. Estándar en ML moderno.
- **Calendario bursátil (XNYS)**: conjunto canónico de días hábiles de la NYSE. Se obtiene con `pandas_market_calendars`.
- **OHLCV**: cinco columnas estándar de cada activo — Open, High, Low, Close, Volume.
- **Forward-fill**: rellenar un valor faltante con el último valor conocido. Aceptable para macro en huecos puntuales, **nunca** para precios de los activos en el universo de evaluación.

### Tareas

| ID | Descripción | Archivo destino | Criterio de aceptación | Dependencias | Esfuerzo |
|---|---|---|---|---|---|
| **F1-T1** | Crear estructura de directorios del proyecto siguiendo el patrón del reporte (§6.1): `src/`, `data/{raw,processed,synthetic,splits}/`, `configs/`, `scripts/`, `notebooks/`, `tests/`, `outputs/`. Inicializar repo Git, `.gitignore` excluyendo `data/raw/`, `models/`, `mlruns/`, `outputs/`. | `README.md`, `.gitignore`, estructura de carpetas | `git status` limpio; árbol de carpetas coincide con §6.1 del reporte; README con título, equipo, director y stack | — | S |
| **F1-T2** | Crear `pyproject.toml` (o `requirements.txt`) con stack pinneado: Python 3.11, `numpy`, `pandas`, `pyarrow`, `yfinance`, `pandas-datareader`, `pandas_market_calendars`, `pandas-ta`, `pyyaml`, `matplotlib`, `seaborn`. Otras libs se añaden en sus fases. | `pyproject.toml` o `requirements.txt` | `pip install -r requirements.txt` se completa sin error en venv limpio | F1-T1 | S |
| **F1-T3** | Implementar descarga primaria de los 6 activos vía yfinance entre 2015-01-01 y 2025-04-30, con retry/backoff y validación básica (sin fechas duplicadas, sin precios negativos, columnas OHLCV completas). Persistir cada activo como Parquet en `data/raw/equities/{ticker}.parquet`. | `src/data/download.py` (función `download_equities`) | Genera 6 ficheros Parquet; cada uno tiene ≥2500 filas; columnas {Open, High, Low, Close, Volume, Adj Close} sin NaN | F1-T2 | M |
| **F1-T4** | Implementar fallback con Stooq (`pandas_datareader.data.DataReader(ticker, 'stooq')`) que se activa si yfinance falla o devuelve dataframe vacío. Documentar en docstring que el fallback es solo para descarga inicial, no para producción. | `src/data/download.py` (función `download_with_fallback`) | Forzando fallo de yfinance, la función trae datos válidos de Stooq para los 6 activos | F1-T3 | S |
| **F1-T5** | Implementar descarga de las 3 variables macro: VIX (`^VIX`), Treasury 10Y (`^TNX`), DXY (`DX-Y.NYB`). Persistir en `data/raw/macro/{var}.parquet`. Misma ventana temporal y validación que activos. | `src/data/download.py` (función `download_macro`) | Genera 3 Parquet de macro con ≥2500 filas cada uno | F1-T3 | S |
| **F1-T6** | Implementar alineación al calendario XNYS usando `pandas_market_calendars`: obtener el universo canónico de fechas hábiles 2015-01-01 a 2025-04-30, hacer `reindex()` de todas las series. Para activos: eliminar filas si algún activo tiene NaN (registrar % eliminado, debe ser <0.5%). Para macro: aplicar forward-fill solo en huecos ≤3 días. | `src/data/alignment.py` (función `align_to_xnys`) | Dataset combinado tiene exactamente la longitud del calendario XNYS menos los días eliminados; logs muestran % eliminados por motivo | F1-T3, F1-T5 | M |
| **F1-T7** | Crear script CLI `scripts/01_download_data.py` que orquesta F1-T3, F1-T5 y F1-T6. Acepta arg `--force-redownload` (por defecto cachea). Loggea fecha de descarga (importante para reproducibilidad — anotarla en el README cuando se ejecute). | `scripts/01_download_data.py` | `python scripts/01_download_data.py` corre end-to-end y deja `data/processed/aligned.parquet` con 9 columnas de cierre + 5 OHLCV por activo + macro | F1-T6 | S |
| **F1-T8** | Crear test unitario que valide: (a) número de filas coincide con calendario XNYS aproximado; (b) no hay valores negativos en precios o volumen; (c) no hay NaN tras la alineación; (d) las fechas están ordenadas y son únicas. | `tests/test_data.py::test_aligned_dataset` | `pytest tests/test_data.py::test_aligned_dataset` pasa | F1-T7 | S |

### Riesgos / pitfalls específicos

1. **yfinance puede romperse sin aviso** (cambios de API o IP bans, §5.3). Mitigación: descarga única + cache local (Parquet); fallback a Stooq; documentar la fecha exacta de descarga.
2. **Survivorship bias**: el universo (S&P 500, NASDAQ-100, AAPL/AMZN/NFLX/NVDA) son ganadores ex-post. No se puede eliminar, pero hay que **acknowledge explícito en el Cap. 6** y compararnos contra benchmarks pasivos (equiponderado, S&P 500 buy-and-hold) en la Fase 7.
3. **Tickers ambiguos**: `^TNX` (Treasury 10Y) en yfinance representa el yield × 10. Hay que validar la unidad antes de usarlo (rango razonable: 0.5–5.0).
4. **Días no comunes entre activos y macro**: días en que el VIX cotiza pero hay un feriado parcial. Resolver con la alineación al XNYS (F1-T6).
5. **NVDA pre-splits**: yfinance ajusta automáticamente por splits; usar `Adj Close` para retornos consistentes.

### Mapeo al cronograma UNIR

- **Semana 7** (Desarrollo de la contribución – inicio).
- Entrega afectada: **Borrador Intermedio (Sem. 9)** — el dataset descargado y alineado debe estar fijo antes de empezar TimeGAN.
- En la memoria, esta fase nutre la sección **"Fase 1: Adquisición de datos"** del Cap. 5 (Desarrollo de la comparativa).

---

## Fase 2: Análisis exploratorio de datos (EDA) y entendimiento del dominio

### ¿Qué es esta fase y por qué importa?

EDA significa "mirar los datos antes de modelar". Sirve para entender la distribución, detectar problemas no obvios, y formar intuición antes de tomar decisiones costosas. En cualquier proyecto de ML, omitir EDA es la causa #1 de modelos que parecen funcionar pero fallan en producción. Para series temporales financieras hay un EDA **adicional** específico: verificar que los datos tienen los **stylized facts** esperados de retornos reales — colas pesadas, volatility clustering, ausencia de autocorrelación lineal, leverage effect. Estos serán nuestra vara de medir cuando evaluemos los datos sintéticos de TimeGAN.

### ¿Cómo se aplica a nuestro TFE?

Vamos a hacer EDA con dos objetivos concretos: (1) **diagnóstico**: detectar outliers, períodos anómalos, datos faltantes que se nos hayan pasado en la Fase 1; (2) **baseline para validar TimeGAN**: medir los stylized facts en los datos reales para tener un objetivo cuantitativo contra el que comparar los sintéticos. También identificaremos a priori los **sub-períodos de estrés** dentro del test (rates shock 2023, AI rally 2024, sell-off Q1 2025) que después usaremos en la evaluación condicional de la Fase 7.

### Conceptos clave que vamos a usar

- **ACF (Autocorrelation Function)**: correlación de la serie consigo misma con desfase de `k` períodos. Para retornos: ≈0 a partir de lag 1. Para `|retornos|`: decae lentamente (volatility clustering).
- **Kurtosis**: 4º momento estandarizado. Distribución normal = 3; retornos reales típicamente >3 (fat tails).
- **Régimen de mercado**: período con propiedades estadísticas estables (alta vol vs baja vol, alcista vs bajista). Suele identificarse por umbrales de VIX o detección automática.
- **Drawdown**: serie que mide cuánto está cayendo el activo desde su máximo previo. Útil para visualizar regímenes de estrés.

### Tareas

| ID | Descripción | Archivo destino | Criterio de aceptación | Dependencias | Esfuerzo |
|---|---|---|---|---|---|
| **F2-T1** | Notebook EDA con: serie temporal de cierres ajustados de los 6 activos (panel de 6 plots, eje Y log), tabla resumen de estadísticas básicas (count, mean, std, min, max, fechas inicio/fin), distribución de log-returns por activo (histograma + ajuste normal sobreimpuesto), tabla de % NaN por columna. | `notebooks/01_eda.ipynb` (sección 1) | Notebook ejecuta sin error; visualmente se aprecian COVID 2020 y subidas 2023-2024 | F1-T7 | M |
| **F2-T2** | Calcular y reportar **stylized facts en datos reales** por activo: (a) kurtosis y skew de log-returns; (b) test de normalidad Shapiro-Wilk; (c) ACF de retornos hasta lag 30; (d) ACF de `abs(retornos)` hasta lag 30; (e) leverage effect = corr(r_t, abs(r_{t+1..k})) para k=1..5. Guardar tabla CSV en `outputs/eda/stylized_facts_real.csv`. | `notebooks/01_eda.ipynb` (sección 2), `src/eval/stylized_facts.py` | Tabla CSV con 5 métricas × 6 activos; plots ACF guardados en `outputs/eda/figs/` | F2-T1 | M |
| **F2-T3** | Análisis de variables macro: serie temporal de VIX, ^TNX, DXY; correlaciones contemporáneas y con lag entre macro y log-returns medios de los 6 activos; identificar y graficar regímenes alta-vol (`VIX > 25`) vs baja-vol. | `notebooks/01_eda.ipynb` (sección 3) | Plots de las 3 macro + matriz de correlación cross 6×3; lista de períodos VIX>25 con duración y MDD del S&P 500 | F2-T1 | M |
| **F2-T4** | Identificar a priori **sub-períodos de estrés en el test** (2023-01-01 a 2025-04-30) según el reporte (§4.5): rates shock 2023 mar–oct, AI rally + correcciones 2024, sell-off Q1 2025 si aplica. Documentar fechas exactas en YAML para uso posterior. | `configs/eval/stress_subperiods.yaml`, `notebooks/01_eda.ipynb` (sección 4) | YAML con 3 períodos `[start, end, label]`; gráfico del S&P 500 en test con bandas sombreadas marcando los períodos | F2-T3 | S |
| **F2-T5** | Matriz de correlaciones cruzadas entre los 6 activos (correlación de Pearson sobre log-returns) en train vs test; reportar diferencia (¿la estructura cambia?). Heatmap. | `notebooks/01_eda.ipynb` (sección 5) | 2 heatmaps + tabla con norma Frobenius de la diferencia | F2-T1 | S |
| **F2-T6** | Documentar hallazgos del EDA en una nota de 1 página (`outputs/eda/eda_summary.md`) con bullet points: (a) calidad de datos OK / pendientes; (b) propiedades estadísticas observadas; (c) regímenes identificados; (d) implicaciones para modelado. Esta nota se incorpora literalmente al Cap. 5 de la memoria. | `outputs/eda/eda_summary.md` | Documento de 300–600 palabras estructurado en 4 secciones | F2-T2, F2-T3, F2-T4, F2-T5 | S |
| **F2-T7** | Test unitario sobre los stylized facts: dado un activo cualquiera del dataset real, validar que kurtosis > 3 y que ACF(abs(r), 1) > ACF(r, 1). Sirve también como check para sintéticos en Fase 4. | `tests/test_stylized_facts.py::test_real_data_has_fat_tails` | `pytest tests/test_stylized_facts.py` pasa con datos reales | F2-T2 | S |

### Riesgos / pitfalls específicos

1. **EDA sin propósito**: producir 50 gráficos sin extraer decisiones es perder tiempo. Cada figura tiene que responder una pregunta concreta.
2. **Conclusiones sobre el test**: técnicamente, mirar el test en EDA puede sesgar decisiones de hiperparámetros (forma sutil de leakage). Aceptable: estadísticas descriptivas. Inaceptable: ajustar features según comportamiento del test.
3. **Subestimar impacto de COVID en train**: el shock de 2020 distorsiona las estadísticas globales. Considerar reportar stylized facts excluyendo el período mar–jun 2020 como sanity check adicional.
4. **No documentar**: si el notebook se borra por accidente, lo aprendido se pierde. Por eso F2-T6 (resumen Markdown).

### Mapeo al cronograma UNIR

- **Semana 7** (paralela a Fase 1).
- Entrega afectada: **Borrador Intermedio (Sem. 9)** — los hallazgos van al Cap. 5 sección "Análisis exploratorio".
- El listado de sub-períodos de estrés (F2-T4) es input directo de la Fase 7 y va citado en el Cap. 4 (Planteamiento de la comparativa).

---

## Fase 3: Preprocesamiento e ingeniería de características (features)

### ¿Qué es esta fase y por qué importa?

El preprocesamiento transforma los datos crudos en el "estado" que el modelo va a consumir. La ingeniería de características (feature engineering) decide qué variables derivadas se calculan a partir de los datos crudos para que el modelo aprenda más fácilmente. En ML financiero esta fase es **especialmente delicada** por dos motivos: (1) los retornos no son estacionarios y hay que normalizar con cuidado para evitar **data leakage** (el pecado capital), y (2) las decisiones tomadas aquí condicionan tanto al modelo generativo (TimeGAN) como al de RL (PPO).

### ¿Cómo se aplica a nuestro TFE?

Aplicaremos las decisiones del reporte (§1.1–1.6) **al pie de la letra**: log-returns como retornos base, set de 9 indicadores técnicos seleccionados (MACD, RSI, BBANDS, CCI, ADX, EMA, ATR, OBV, RealizedVol), transformaciones estacionarias para macro, MinMaxScaler `[0,1]` ajustado **solo en train** para alimentar TimeGAN, y un RobustScaler separado para alimentar el estado del PPO. El split cronológico **70/10/20** (train: 2015–2021, val: 2022, test: 2023–2025) queda persistido para que todos los módulos posteriores lean los mismos índices. La salida es un dataset listo para consumo en `data/processed/`.

### Conceptos clave que vamos a usar

- **Indicador técnico**: feature derivada del precio/volumen (MACD, RSI, etc.). Resume información en una ventana móvil.
- **MinMaxScaler**: transformación lineal que mapea cada feature al rango `[0, 1]`. Requerida por TimeGAN (sigmoide en su recovery network).
- **RobustScaler**: como MinMax pero usa mediana/IQR en vez de min/max; insensible a outliers. Adecuado para el estado del PPO.
- **Walk-forward / split cronológico**: dividir por fechas, no aleatoriamente. El test es siempre posterior al train.
- **Lag / shift**: desplazar una serie. Crítico para garantizar que la observación en `t` no contiene info de `t+1` (look-ahead bias).
- **Anti-leakage**: ajustar todo (`scalers`, indicadores con ventana móvil) usando solo datos del pasado disponibles en cada momento.

### Tareas

| ID | Descripción | Archivo destino | Criterio de aceptación | Dependencias | Esfuerzo |
|---|---|---|---|---|---|
| **F3-T1** | Implementar cálculo de log-returns por activo: `r_t = log(Adj Close_t / Adj Close_{t-1})`. Reemplazar precios crudos en el dataframe principal por columnas derivadas: `log_ret`, `body_rel = (Close-Open)/Open`, `range_rel = (High-Low)/Close`, `log_volume = log(1+Volume)`. | `src/data/features.py` (función `compute_returns_and_ohlcv_features`) | Dataframe resultante no contiene precios crudos (excepto Adj Close para referencia); 4 columnas nuevas por activo; sin NaN excepto la primera fila | F1-T7 | S |
| **F3-T2** | Implementar el set de 9 indicadores técnicos del reporte (§1.1) usando `pandas-ta`: MACD(12,26,9), RSI(14), BBANDS(20,2) con upper/lower/width/%B, CCI(14), ADX(14), EMA(10), EMA(30), ATR(14), OBV. Cuidar que **todas las funciones sean causales** (no center=True). | `src/data/features.py` (función `compute_technical_indicators`) | Dataframe con ~9-12 columnas extra por activo; tras descartar las primeras `max_lookback` filas (≈30) no hay NaN; lista de columnas coincide con reporte §1.1 | F3-T1 | M |
| **F3-T3** | Añadir feature de volatilidad realizada por activo: `realized_vol_5d = std(log_ret, 5) * sqrt(252)` y `realized_vol_21d`. Incluir ratio `vol_5d / vol_21d` como indicador de régimen. | `src/data/features.py` (función `compute_realized_vol`) | 3 columnas extra por activo; test de causalidad pasa (ver F3-T11) | F3-T1 | S |
| **F3-T4** | Transformar las macro en features estacionarias (reporte §1.2): `log(VIX)`, `ΔVIX = VIX_t - VIX_{t-1}`, `EMA(VIX, 5)`, `EMA(VIX, 21)`, flag binario `VIX > 25`; `Δyield_TNX` en puntos básicos; `log_return_DXY`. Estas son **variables globales** (replicadas igual para todos los activos). | `src/data/features.py` (función `compute_macro_features`) | 7 columnas globales en el dataframe; ninguna en niveles crudos; test de estacionariedad ADF muestra rechazo de raíz unitaria para `ΔVIX`, `Δyield`, `log_return_DXY` | F3-T1 | M |
| **F3-T5** | Implementar split cronológico **70/10/20** (reporte §4.1): train 2015-01-01 a 2021-12-31, val 2022-01-01 a 2022-12-31, test 2023-01-01 a 2025-04-30. Persistir los **índices** (no los datos) en `data/splits/{train,val,test}_idx.parquet` para que cualquier módulo cargue el mismo split de forma idéntica. | `src/data/splits.py` (función `chronological_split`) | 3 ficheros Parquet con fechas; suma de longitudes = longitud total del dataset; sin solapamiento; train contiene COVID (mar–abr 2020) | F3-T1 | S |
| **F3-T6** | Implementar `MinMaxScaler` para TimeGAN: ajustar **solo en train** sobre las features que entrarán al generativo (log-returns de 6 activos + 3 macro derivadas = 9 dims según opción A del reporte §1.6). Persistir el scaler con `joblib.dump` en `models/scalers/timegan_minmax.joblib`. | `src/data/scalers.py` (función `fit_timegan_scaler`) | Scaler persistido; al aplicarlo a val y test, los valores quedan típicamente en `[0,1]` (con posibles outliers de val/test que excedan, lo cual es esperado y correcto) | F3-T5 | S |
| **F3-T7** | Implementar `RobustScaler` para el estado del PPO: ajustar **solo en train** sobre el vector completo de features (indicadores + macro + retornos) que va al estado del agente. Persistir scaler en `models/scalers/ppo_robust.joblib`. | `src/data/scalers.py` (función `fit_ppo_scaler`) | Scaler persistido; mediana de cada feature transformada ≈0 sobre train | F3-T5, F3-T2, F3-T3, F3-T4 | S |
| **F3-T8** | Test anti-leakage del scaler: cargar `timegan_minmax.joblib` y `ppo_robust.joblib`, recalcular sus stats sobre train y verificar que coinciden bit a bit con los persistidos; verificar que `data_min_` / `data_max_` (MinMax) no contienen valores que solo existen en val o test. | `tests/test_no_leakage.py::test_scaler_only_train` | `pytest tests/test_no_leakage.py::test_scaler_only_train` pasa | F3-T6, F3-T7 | S |
| **F3-T9** | Construir el **estado del PPO** como tensor `(window=30, n_features)` aplicando el RobustScaler ya ajustado. Función pura que recibe un dataframe + índice `t` y devuelve la observación. Documentar shape exacto. | `src/data/state_builder.py` (función `build_state`) | Para cualquier `t` en train ∪ val ∪ test (con `t ≥ window`), devuelve tensor de shape `(30, total_features)`; sin NaN; reproducible (misma entrada → misma salida) | F3-T7 | M |
| **F3-T10** | Construir el **dataset de entrada a TimeGAN**: ventanas deslizantes de longitud `seq_len=24` (reporte §1.6) sobre los datos de **train escalados con MinMax**, output shape `(N_seq, 24, 9)`. Persistir en `data/processed/timegan_train_sequences.parquet` (o .npy si Parquet no soporta el formato). | `src/data/sequence_builder.py` (función `build_timegan_sequences`) | Genera tensor de forma `(≈1700, 24, 9)`; valores en `[0,1]`; primera y última secuencia coinciden con cálculo manual | F3-T6 | M |
| **F3-T11** | Test de no look-ahead: para una fecha `t` aleatoria del test, verificar que todas las features (indicadores técnicos, realized vol, macro) calculadas en `t` solo dependen de datos en `[0, t]`. Implementar comparando contra recálculo truncado. | `tests/test_no_leakage.py::test_no_lookahead_in_features` | Test pasa para 100 fechas aleatorias | F3-T2, F3-T3, F3-T4 | M |
| **F3-T12** | Crear script CLI `scripts/02_build_features.py` que orquesta F3-T1 a F3-T10 y persiste todos los artefactos. Logueo de número de filas, columnas, shapes generados. | `scripts/02_build_features.py` | Una sola ejecución produce: dataset procesado, scalers, splits, secuencias TimeGAN. Idempotente | F3-T10 | S |

### Riesgos / pitfalls específicos

1. **Look-ahead bias en indicadores** (reporte §7.3, pitfall #1 en DRL financiero): si `pandas-ta` o nuestro código usa `.center=True` o ventanas que miran al futuro, todo el experimento queda invalidado. F3-T11 lo detecta automáticamente.
2. **Scaler ajustado en todo el dataset**: error muy común. La regla es **siempre `fit` solo en train, `transform` en val/test**.
3. **Caída de filas iniciales no documentada**: cada indicador con ventana W consume las primeras W-1 filas. Tras todos los indicadores, las primeras ~30 filas tendrán NaN: hay que descartarlas explícitamente y registrar cuántas (~ 1.5% del train). El split en F3-T5 debe hacerse **después** del descarte.
4. **Mezclar precios crudos con retornos**: el estado del PPO no debe contener precios absolutos (overfitting al nivel del activo). Solo retornos y features estacionarias.
5. **Sequence overlap en TimeGAN**: las secuencias de F3-T10 son ventanas deslizantes con stride=1 (overlapping). Esto es lo estándar pero genera correlación entre muestras; tenerlo presente al evaluar la métrica de discriminación.

### Mapeo al cronograma UNIR

- **Semana 7–8** (Desarrollo de la contribución).
- Entrega afectada: **Borrador Intermedio (Sem. 9)** — los splits, scalers y la receta de features deben estar fijos antes de entrenar TimeGAN.
- En la memoria, esta fase corresponde a la sección **"Fase 2: Preprocesamiento e ingeniería de características"** del Cap. 5.

---

## Fase 4: Modelado generativo (TimeGAN multivariado)

### ¿Qué es esta fase y por qué importa?

Un **modelo generativo** aprende la distribución conjunta de los datos para luego producir muestras nuevas que se "parezcan" a las reales. En tiempos recientes los modelos generativos profundos (GANs, VAEs, modelos de difusión) son la herramienta dominante para generar datos sintéticos. Para nuestro caso, el modelo generativo cumple una función concreta: **aumentar el conjunto de entrenamiento del Agente B** con escenarios plausibles que no aparecieron en la historia real, mejorando su robustez ante regímenes raros.

**TimeGAN** (Yoon et al., NeurIPS 2019) es la referencia estándar para series temporales. Combina cinco redes (Embedder, Recovery, Generator, Supervisor, Discriminator) entrenadas en **tres fases secuenciales** (embedding → supervised → joint). Se entrena en un espacio latente, no directamente sobre los datos crudos, lo que mejora la estabilidad.

### ¿Cómo se aplica a nuestro TFE?

Vamos a entrenar un **TimeGAN multivariado** sobre **9 dimensiones** (log-returns de los 6 activos + 3 macro derivadas) con `seq_len=24` (un mes hábil), siguiendo la **opción A** del reporte (§1.6) — los indicadores técnicos se recalcularán determinísticamente sobre los retornos sintéticos para no inflar la dimensionalidad. Stack: **PyTorch** (port de `zwzhang123/TimeGAN-pytorch` validado contra el oficial), porque SB3 también es PyTorch y queremos coherencia. Hiperparámetros del reporte (§2.2) tomados al pie de la letra. La salida son **K secuencias sintéticas de shape `(N_seq × K, 24, 9)`** con `K ∈ {1, 2, 3}` para ablación, y `K=2` como ratio principal (real:sintético = 1:2).

La validación de calidad (§2.4) es **crítica**: si TimeGAN no pasa los tests, el Agente B se entrena con basura y la comparativa pierde sentido.

### Conceptos clave que vamos a usar

- **GAN (Generative Adversarial Network)**: dos redes (Generator y Discriminator) que compiten; Generator aprende a engañar al Discriminator que distingue real vs sintético.
- **Embedding (E) y Recovery (R)**: redes auto-encoder que mapean datos al espacio latente y de vuelta. TimeGAN entrena en latente.
- **Supervisor (S)**: red que predice la siguiente representación latente dado el pasado; añade un loss supervisado que ayuda a la dinámica temporal.
- **Discriminative score**: métrica de calidad. Entrenar un clasificador real-vs-sintético; reportar `|accuracy - 0.5|`. <0.15 es bueno.
- **Predictive score (TSTR)**: Train on Synthetic, Test on Real. Mide si los sintéticos contienen dinámica predictiva equivalente a la real.
- **t-SNE / PCA**: reducción de dimensionalidad para visualizar si las muestras reales y sintéticas se solapan en 2D.
- **Stylized facts en sintéticos**: los mismos 4 que en Fase 2 (kurtosis, ACF retornos, ACF |retornos|, leverage) pero ahora aplicados a los sintéticos para validarlos.

### Tareas

| ID | Descripción | Archivo destino | Criterio de aceptación | Dependencias | Esfuerzo |
|---|---|---|---|---|---|
| **F4-T1** | Implementar las 5 redes de TimeGAN en PyTorch: Embedder (E), Recovery (R), Generator (G), Supervisor (S), Discriminator (D). Todas con módulo GRU (más estable que LSTM, reporte §2.3), `num_layers=3`, `hidden_dim=24`. Ver paper Yoon et al. 2019 figura 2. | `src/generative/timegan/model.py` | Las 5 clases instancian sin error; un forward pass con tensor `(B, 24, 9)` aleatorio devuelve shapes correctos | F1-T2 | M |
| **F4-T2** | Implementar el bucle de entrenamiento en **3 fases secuenciales** (reporte §2.3, OBLIGATORIO no opcional): (1) **Embedding**: entrenar E+R con loss de reconstrucción `L_R = MSE(R(E(x)), x)`; (2) **Supervised**: entrenar S sobre embeddings reales con loss `L_S = MSE(S(E(x)_t), E(x)_{t+1})`; (3) **Joint**: entrenar todo conjuntamente con `gamma·L_GAN + eta·L_R + L_S`, `gamma=1`, `eta=10`. Logs en **MLflow** (ver desviación I al final de la sección). | `src/generative/timegan/train.py` | Loop completo arranca y muestra losses descendentes en cada una de las 3 fases. Logs en MLflow | F4-T1 | L |
| **F4-T3** | Configurar hiperparámetros de TimeGAN en YAML/Hydra (reporte §2.2): `seq_len=24, hidden_dim=24, num_layers=3, batch_size=128, lr=5e-4, iterations=20000, gamma=1, eta=10, module='gru', noise_dim=32`. **No reformulearlos**. | `configs/timegan/default.yaml` | Config se carga con `OmegaConf.load()`; valores coinciden con reporte §2.2 | — | S |
| **F4-T4** | Implementar **early stopping** basado en **discriminative score** evaluado periódicamente (cada 1000 iter de la fase joint) sobre un mini-batch del val. Persistir el mejor checkpoint según ese criterio. Reporte §2.3: "reportar mejor modelo según validación discriminative score, no el último". | `src/generative/timegan/train.py` (callback) | Logs muestran discriminative score evaluado cada 1000 iter; mejor checkpoint se guarda en `models/timegan/best.pt` | F4-T2, F4-T8 | M |
| **F4-T5** | Implementar generación de muestras: dada una semilla aleatoria y un número `K`, producir `K × N_seq` secuencias sintéticas en el espacio escalado `[0,1]`, luego invertir el `MinMaxScaler` persistido para recuperar log-returns multivariados. | `src/generative/timegan/generate.py` (función `generate_synthetic`) | Para `K=2` produce ≈3400 secuencias de shape `(24, 9)`; valores tras inverse_transform están en rangos plausibles de log-returns (típicamente `[-0.1, 0.1]`) | F4-T2, F3-T6 | M |
| **F4-T6** | A partir de las secuencias de log-returns sintéticas (F4-T5), **reconstruir series tipo "OHLC sintético"** acumulando log-returns desde un precio inicial arbitrario (e.g. 100). Necesario para recalcular indicadores técnicos sobre sintéticos. Documentar que los precios son ficticios pero los retornos preservan las dinámicas relevantes. | `src/generative/timegan/reconstruct_prices.py` | Para cada secuencia sintética genera precios sintéticos coherentes (Close fila i+1 = Close i × exp(r_{i+1})); valores siempre positivos | F4-T5 | S |
| **F4-T7** | Recalcular el set completo de **indicadores técnicos sobre las series sintéticas** (mismas funciones de F3-T2/F3-T3) y producir el dataset sintético definitivo con la misma estructura de columnas que el real procesado. Persistir en `data/synthetic/run_<id>/synthetic_dataset.parquet`. | `src/generative/timegan/build_synthetic_dataset.py` | Dataset sintético con misma cantidad de columnas que el real procesado tras F3 | F4-T6, F3-T2 | M |
| **F4-T8** | Implementar **discriminative score**: entrenar un clasificador GRU 2 capas para distinguir real vs sintético sobre 80% de un set combinado, evaluar sobre 20%, reportar `abs(accuracy - 0.5)`. Repetir 3 veces y promediar. | `src/generative/timegan/metrics.py::discriminative_score` | Función devuelve un escalar en `[0, 0.5]`. Para sintéticos basura (e.g. ruido gaussiano) devuelve ≈0.5; para reales-vs-reales (control) devuelve ≈0 | F4-T7 | M |
| **F4-T9** | Implementar **predictive score (TSTR)**: entrenar un GRU forecaster que prediga el siguiente log-return dado el pasado, **entrenado sobre sintéticos** y evaluado sobre reales. Reportar MAE y compararlo con un baseline equivalente entrenado en reales (TRTR). Reporte §2.4. | `src/generative/timegan/metrics.py::predictive_score` | Devuelve `{mae_tstr, mae_trtr, gap}`. `gap < 10%` es bueno | F4-T7 | M |
| **F4-T10** | Visualización **t-SNE / PCA**: proyectar 1000 secuencias reales y 1000 sintéticas a 2D y plotear; deben solaparse visualmente. Guardar figura en `outputs/timegan/tsne.png`. | `src/generative/timegan/metrics.py::tsne_plot` | Función genera figura PNG con 2 nubes de puntos coloreadas | F4-T7 | S |
| **F4-T11** | Implementar **batería de stylized facts sobre sintéticos** (reporte §2.4 punto 4): kurtosis, skew, ACF retornos, ACF abs(retornos), leverage effect, matriz de correlaciones cruzadas (Frobenius distance vs real). Producir tabla `outputs/timegan/stylized_facts_compare.csv` con 6 métricas × 2 (real, sintético) × 6 activos. | `src/eval/stylized_facts.py::compare_real_vs_synthetic` (reusa F2-T2) | Tabla CSV; gráficos comparativos guardados | F4-T7, F2-T2 | M |
| **F4-T12** | Test anti-leakage de TimeGAN: validar que `data/processed/timegan_train_sequences.npy` (formato real, ver desviación II) contiene solo fechas en `train_idx`. Validar que el scaler usado fue `timegan_minmax.joblib` ajustado solo en train. | `tests/test_no_leakage.py::test_timegan_train_only` | Test pasa | F3-T8, F3-T10 | S |
| **F4-T13** | Crear script CLI `scripts/03_train_timegan.py` que orquesta entrenamiento + checkpoint + generación + métricas. Multirun con `seed ∈ {0, 42, 123}` para mostrar reproducibilidad de la calidad. | `scripts/03_train_timegan.py` | Una ejecución produce: checkpoint, sintéticos, tabla de métricas. `--seed` cambia los sintéticos pero no rompe métricas | F4-T4, F4-T7 | M |
| **F4-T14** | **Gate de calidad**: el script `04_evaluate_synthetic.py` lee la tabla de métricas de F4-T8/T9/T11 y aborta con error si: (a) discriminative_score > 0.30; (b) gap predictive > 25%; (c) ACF(abs(r), lag=1) sintético < 50% del real (volatility clustering ausente). Si pasa, marca los sintéticos como "aptos para Fase 6". | `scripts/04_evaluate_synthetic.py` | Si los criterios fallan, exit code ≠ 0 e informe de qué falló. Si pasan, escribe `data/synthetic/run_<id>/QUALITY_OK.flag` | F4-T13 | S |

### Riesgos / pitfalls específicos

1. **Saltarse la fase supervisada** (reporte §2.3): es la causa más documentada de fracaso de TimeGAN. Las 3 fases son obligatorias.
2. **Mode collapse**: si todos los sintéticos parecen iguales, la t-SNE será una nube apretada y la kurtosis cae. Mitigaciones (reporte §2.3): bajar lr, usar GRU, considerar WGAN-GP en una iteración futura.
3. **Discriminator domina**: D loss → 0, G loss → ∞. Si pasa, reducir updates de D por update de G; añadir noise a inputs de D.
4. **Sintéticos sin fat tails**: TimeGAN típico subestima la kurtosis (es conocido en literatura). El gate F4-T14 detecta el caso extremo; en el reporte del Cap. 6 se debe discutir como limitación incluso si se pasa el gate.
5. **Memorización**: TimeGAN puede aprender a reproducir secuencias casi-idénticas a las reales. Test sugerido (no obligatorio): para cada secuencia sintética, calcular distancia al vecino real más cercano; si la mediana es muy baja, alarma.
6. **Cómputo**: 20k iteraciones × seq_len=24 × 9 dims pueden tardar 1–4h en GPU T4 (Colab Free). Asegurar checkpointing intermedio para no perder horas si Colab desconecta.
7. **Compatibilidad PyTorch del fork**: validar antes de invertir tiempo. Si el fork tiene bugs, plan B: aislar `ydata-synthetic` (TF2) en venv separado y comunicar por archivos (reporte §2.1).

### Mapeo al cronograma UNIR

- **Semana 8–9** (parte central del Borrador Intermedio).
- Entrega afectada: **Borrador Intermedio (Sem. 9)** — para entregarlo, debe haber al menos una corrida exitosa de TimeGAN con métricas decentes; idealmente 1 corrida con gate aprobado.
- En la memoria, esta fase corresponde a la sección **"Fase 3: Implementación del modelo generativo TimeGAN"** del Cap. 5. Los resultados de F4-T8/9/10/11 son la base de la **subsección de validación de sintéticos** del Cap. 5 (y se vuelven a discutir en Cap. 6).

### Desviaciones documentadas vs ADR original

Durante la ejecución de F4 se tomaron 17 decisiones arquitectónicas no explícitas en el ADR/compass. Se documentan aquí para trazabilidad en la memoria. Todas están justificadas en el plan de implementación (`~/.claude/plans/en-base-a-este-synchronous-storm.md`).

**Correcciones de hechos del ADR original**:

- **I. Tracking**: el ADR menciona TensorBoard en F4-T2 y F4-T4. En la implementación se usa **MLflow** (file-based en `mlruns/`, un único experimento `timegan_phase4`, run por seed). Coherente con el comentario en `pyproject.toml` línea 12. Razón: comparativa entre runs más cómoda y artefactos integrados.
- **II. Formato de secuencias**: el ADR F4-T12 menciona `.parquet`; el formato real persistido en F3 es **`.npy`** (`data/processed/timegan_train_sequences.npy` shape `(1681, 24, 9)` float32). El test anti-leakage opera sobre `.npy`.

**Decisiones arquitectónicas (tabla)**:

| # | Decisión | Justificación |
|---|---|---|
| 0.1 | PyTorch en `[project].dependencies`, no en `dev` | Es runtime de `scripts/03_*.py` y `scripts/04_*.py` |
| 2.1 | Wrapper `TimeGAN(nn.Module)` agrupando las 5 redes + clases sueltas | `state_dict` único para `torch.save` + testeabilidad |
| 3.1 | Incluir `L_G_moments` y factor 100× en `L_G_supervised` en fase joint | Paper Yoon original (ecs. 6-7); sin ellos mode collapse casi seguro con N=1681 y seq_len=24. Documentar como "extensión vs ADR literal" |
| 3.2 | Holdout para early stopping = 10 % final cronológico (≈168 secs terminando en 2021-Q4) | Stride=1 ⇒ leakage trivial entre secuencias adyacentes si se randomiza |
| 3.3 | Iters embedding:supervised:joint = **1:1:1 → 10k+10k+10k = 30k total** (vs ADR §F4-T2 que fija solo joint=20k) | Acotar tiempo en M4 Pro (~3-6h overnight para 3 seeds). Compass §2.3 admite 10k-20k joint con early stopping. Modo smoke (2k+2k+5k, ~15 min) reservado para validación previa |
| 3.4 | Device auto-detect MPS → CPU fallback; `use_deterministic_algorithms(warn_only=True)` | M4 Pro tiene MPS estable para GRU; algunas ops MPS no son 100% deterministas |
| 4.1 | High/Low sintéticos vía `range_rel` gaussiano calibrado sobre TRAIN; `Open_t = Close_{t-1}` | TimeGAN no modela rango intra-día. Limitación documentada en summary |
| 4.2 | Volume sintético = `exp(N(μ_train, σ_train))` por ticker, **independiente del log_return** | OBV es acumulativo, robusto a este ruido. Limitación documentada |
| 4.3 | Buffer histórico de 60 días reales aleatorios de TRAIN para warmup EMA(60), descartado tras `build_features` | Alternativas peores (zero-padding inyecta indicadores falsos; descartar reduce N_synth) |
| 5.1 | Métricas (`discriminative_score`, `predictive_score`) operan en espacio **MinMax-escalado [0,1]**; stylized facts en log-returns originales | Replica paper Yoon §5; rangos uniformes mejoran convergencia del classifier/forecaster |
| 5.2 | `discriminative_score` **balancea las clases** a `min(n_real, n_synth)` por repeat. El gate usa `non_overlap_stride=24` sobre las reales (no-overlapping, sin leakage train/test); el early stopping usa `stride=1` sobre el holdout (pequeño) | El multirun inicial reveló un bug: con 71 reales (tras stride=24) vs 3362 sintéticas, el classifier colapsaba a la clase mayoritaria → `disc_score≈0.48` idéntico en los 3 seeds, sin relación con la calidad. Además congelaba el early stopping (paraba el joint a ~iter 1000/10000). Balancear las clases corrige ambos efectos. Test de regresión: `tests/test_timegan_metrics.py` |
| 7.1 | Test anti-leakage F4-T12 usa comparación **bit-a-bit** contra recálculo desde features_df + scaler (no MSE-search) | Más robusto y rápido; valida shape e identidad de contenido simultáneamente |
| 8.1 | Un único `outputs/timegan/timegan_summary.md` consolidado escrito por `scripts/04_*.py` (no por `03_*.py`) | Hasta no evaluar el gate no sabemos qué seeds son aptos |
| 8.2 | Hydra solo en `scripts/03_*.py` y `scripts/04_*.py`; `scripts/01_*.py` y `scripts/02_*.py` quedan con argparse | No reabrir trabajo de Fases 1/3 |
| 8.3 | MLflow file-based en `mlruns/`, 1 experimento `timegan_phase4`, run por seed | Sin servidor, simple, suficiente para 3 seeds |
| 9.1 | Override `timegan.device=cuda` en Kaggle vía Hydra CLI | Linux T4 no tiene MPS; misma codebase, distinto kernel |
| 9.2 | Reproducibilidad estricta MPS↔CUDA NO se garantiza | Kernels GRU difieren entre devices; documentado en summary |

**Apéndice: ejecución alternativa en Kaggle (T4 GPU)** — `notebooks/02_train_timegan_kaggle.ipynb` permite ejecutar el multirun en Kaggle gratuito (~1-2h vs 3-6h local). Empaqueta los artefactos como `.tar.gz` descargable para integrar en el repo local. Misma codebase, override `timegan.device=cuda` vía CLI. Los seeds 0/42/123 NO reproducen exactamente el run local (decisión 9.2): se reporta el run cuyo gate aprueba.

---

## Fase 5: Diseño del entorno de trading (Gymnasium)

### ¿Qué es esta fase y por qué importa?

En Reinforcement Learning, el "entorno" (`environment`) es el mundo simulado con el que el agente interactúa: define qué observa el agente (estado), qué acciones puede tomar, qué recompensa recibe y cómo evoluciona el sistema tras cada acción. Es la pieza que convierte un problema de negocio (gestión de cartera) en un MDP (Markov Decision Process) que un algoritmo como PPO puede optimizar.

**Diseñar bien el entorno es probablemente la decisión técnica con mayor impacto en el resultado** del proyecto: una mala definición de la recompensa, una acción mal proyectada al simplex de pesos, o un cálculo incorrecto de costes de transacción puede invalidar todo el experimento. La librería estándar es **Gymnasium** (sucesora de OpenAI Gym), que define una API mínima (`reset`, `step`, `observation_space`, `action_space`) que cualquier algoritmo de RL moderno entiende.

### ¿Cómo se aplica a nuestro TFE?

Construiremos un `PortfolioEnv` custom que sigue las decisiones del reporte (§3.2–3.3): **observación** = ventana de 30 días con features escaladas + pesos actuales; **acción** = vector continuo `[-1, 1]^{n_assets+1}` que vía **softmax** produce los pesos de cartera (incluyendo "cash" como activo virtual con retorno 0); **recompensa** = log-return diferencial menos costes de transacción (10 bps × turnover). Importante: los costes se aplican sobre `|w_t - w_{t-1}_drifted|` (los pesos drift después del retorno del día), no sobre `|w_t - w_{t-1}|` directo, para no sobreestimar costes (reporte §3.3). El entorno será **idéntico para Agente A y Agente B**; lo único que cambia entre ambos es el **dataset** que el entorno consume.

### Conceptos clave que vamos a usar

- **MDP (Markov Decision Process)**: marco matemático del RL. Cinco elementos: estados, acciones, transiciones, recompensa, factor de descuento.
- **Estado / observación**: lo que el agente "ve" en cada paso. En nuestro caso una matriz `(window=30, total_features)` aplanada.
- **Acción**: lo que el agente decide. Aquí, un vector continuo que tras softmax representa los pesos de cartera.
- **Recompensa (reward)**: señal escalar que el agente busca maximizar acumulativamente. Aquí, log-return diferencial neto de costes.
- **Factor de descuento (γ)**: pondera recompensas futuras. `γ=0.99` en nuestro caso.
- **Simplex de pesos**: subconjunto de vectores no negativos que suman 1 (las carteras válidas). El softmax garantiza estar en el simplex.
- **Turnover**: `0.5 × Σ |w_t - w_{t-1}_drifted|`, mide cambio de cartera. Genera coste.
- **Cash virtual**: activo extra con retorno 0 (o `r_f`); permite al agente "salirse" del mercado.

### Tareas

| ID | Descripción | Archivo destino | Criterio de aceptación | Dependencias | Esfuerzo |
|---|---|---|---|---|---|
| **F5-T1** | Implementar clase `PortfolioEnv(gym.Env)` con: `__init__(dataset, splits_idx, config)`, `reset()`, `step(action)`. Espacios: `observation_space = Box(-inf, inf, (window, n_features), float32)`, `action_space = Box(-1.0, 1.0, (n_assets+1,), float32)` (con cash). | `src/envs/portfolio_env.py` | Instanciable; `env.reset()` devuelve obs con shape correcto; `env.step(action)` devuelve `(obs, reward, terminated, truncated, info)` | F3-T9 | M |
| **F5-T2** | Implementar **proyección al simplex vía softmax** (reporte §3.2 patrón 1, recomendado por simplicidad): `weights = softmax(action)`. Garantiza pesos en `[0,1]` que suman 1. Documentar que el último componente es "cash". | `src/envs/portfolio_env.py::_action_to_weights` | Para 100 acciones aleatorias: pesos siempre en `[0,1]`, suma exacta 1 (con tolerancia 1e-6) | F5-T1 | S |
| **F5-T3** | Implementar **drift de pesos**: dado `w_{t-1}` y los retornos del día t, calcular `w_{t-1}_drifted = (w_{t-1} ⊙ (1+r_t)) / (w_{t-1} · (1+r_t))`. Aplicar antes de calcular costes (reporte §3.3). | `src/envs/portfolio_env.py::_drift_weights` | Para retornos cero, drift = identidad. Para retornos no cero, suma de drifted = 1 | F5-T2 | S |
| **F5-T4** | Implementar **cálculo de costes de transacción** (reporte §3.3): `turnover = 0.5 * sum(abs(w_t - w_{t-1}_drifted))`, `cost = turnover * 0.001` (10 bps). Hacer config-driven via YAML para poder hacer ablation con 5 bps y 20 bps. | `src/envs/transaction_costs.py` | `cost ≥ 0` siempre; `turnover=0 → cost=0`; valor por defecto 0.001 | F5-T3 | S |
| **F5-T5** | Implementar **función de recompensa**: `reward_t = log(V_t / V_{t-1}) - cost_t` donde `V_t = V_{t-1} * (1 + w_t · r_t) - V_{t-1} * cost_t`. Persistir tracking de V_t, costes acumulados y turnover en `info`. | `src/envs/rewards.py::log_return_minus_costs` | `reward` retorna float32; el wealth `V_T` final coincide con `V_0 * exp(sum(rewards + costs))` con tolerancia | F5-T4 | M |
| **F5-T6** | Configurar entorno via YAML/Hydra (reporte §3.2): `n_assets=6, window=30, transaction_cost_pct=0.001, slippage_pct=0, frame_stack_n=10, action_includes_cash=True, reward_type='log_return_minus_costs'`. **No reformular**. | `configs/env/portfolio_default.yaml` | Config carga; al instanciar `PortfolioEnv(config)` los valores se respetan | — | S |
| **F5-T7** | Validar el entorno con la utilidad oficial: `from gymnasium.utils.env_checker import check_env; check_env(env)`. Confirma que la API es correcta y compatible con SB3. | `tests/test_env.py::test_gymnasium_compliance` | `pytest tests/test_env.py::test_gymnasium_compliance` pasa sin warnings críticos | F5-T1, F5-T6 | S |
| **F5-T8** | Test de **action space**: para 100 acciones aleatorias, los pesos resultantes son no negativos y suman 1. | `tests/test_env.py::test_action_space` | Test pasa | F5-T2 | S |
| **F5-T9** | Test de **consistency de recompensa** (reporte §6.6 punto 3): un episodio completo con acciones fijas (e.g. equiponderado constante) debe satisfacer `wealth_T / wealth_0 ≈ exp(sum(rewards))` con tolerancia. | `tests/test_env.py::test_reward_consistency` | Test pasa con tolerancia 1e-3 | F5-T5 | M |
| **F5-T10** | Test de **costes de transacción**: cero turnover (acción = w_drifted) ⇒ coste cero; turnover=1 ⇒ coste = 0.001 (con la config default). | `tests/test_env.py::test_transaction_costs` | Test pasa | F5-T4 | S |

### Riesgos / pitfalls específicos

1. **Mismatch acción ↔ ejecución**: si el agente decide en `t` con info hasta `t-1` pero ejecuta al cierre de `t-1` (en el pasado), hay leakage. Convención: agente decide en `t` con info hasta cierre de `t-1`; ejecuta al cierre de `t`. **Documentar esto en docstring del env**.
2. **Costes de transacción ignorados** (reporte §7.4): el error #1 en backtesting de DRL. Confirmar que el coste se aplica en train Y en test (mismo entorno).
3. **Agente concentra todo en 1 activo**: síntoma típico cuando `ent_coef` es bajo. Se mitiga en Fase 6 subiendo entropía. Pero el env debe **permitir** la concentración (no clippearla artificialmente).
4. **Reward sin escalar**: log-returns diarios típicos están en `[-0.05, 0.05]`. PPO puede tener problemas con magnitudes tan pequeñas. Considerar `VecNormalize(norm_reward=True)` en Fase 6.
5. **Look-ahead en construcción del estado**: el estado en `t` solo debe contener observaciones de `[t-window, t-1]`. F3-T11 debe pasar antes de empezar a entrenar.
6. **Episodios incompletos**: definir claramente cuándo `terminated=True` (final del split) vs `truncated=True` (truncamiento por longitud). PPO trata ambos diferente.

### Mapeo al cronograma UNIR

- **Semana 8** (paralela a Fase 4 — Diegbys puede estar en TimeGAN mientras Andrés monta el entorno).
- Entrega afectada: **Borrador Intermedio (Sem. 9)** — el entorno con sus tests pasando es el "habilitador" del primer entrenamiento de PPO que se reporta como resultado preliminar.
- En la memoria, esta fase corresponde a la subsección **"Diseño del entorno de simulación"** dentro de la "Fase 4 / 5: Entrenamiento por refuerzo" del Cap. 5.

---

## Fase 6: Modelado por refuerzo (PPO) — entrenamiento de Agente A y Agente B

### ¿Qué es esta fase y por qué importa?

Aquí entrenamos los dos agentes que vamos a comparar — el **núcleo experimental del TFE**. **PPO (Proximal Policy Optimization)** es un algoritmo de policy gradient diseñado para evitar actualizaciones destructivas de la política mediante un *clipping* de la ratio de probabilidades. Es el estándar de facto en RL aplicado por su balance entre estabilidad, eficiencia muestral y simplicidad de implementación. Usaremos la implementación de **Stable-Baselines3 (SB3)**, que es PyTorch nativo, está bien testeada y se integra trivialmente con Gymnasium, MLflow y Hydra.

La regla de oro para que la comparativa A vs B sea válida es **idéntico todo excepto el dataset**: misma arquitectura, mismos hiperparámetros, mismos seeds, mismos timesteps totales (no épocas — esto es importante porque B tiene un dataset 3× más grande). Los seeds múltiples (mínimo 5) son **obligatorios** para hacer estadística (reporte §4.6, §4.7).

### ¿Cómo se aplica a nuestro TFE?

- **Agente A**: PPO entrenado solo sobre el dataset real de train (2015–2021).
- **Agente B**: PPO entrenado sobre dataset aumentado = real (1×) + sintético TimeGAN (2×), ratio 1:2 (reporte §2.6). El entorno samplea aleatoriamente entre real y sintético en cada episodio (estratificado para mantener la proporción).
- Stack: **SB3 ≥ 2.3.0**, `MlpPolicy` + `VecFrameStack(n_stack=10)` (reporte §3.5: alternativa más simple y competitiva que LSTM).
- Hiperparámetros del reporte (§3.4) literales: `lr=3e-4, n_steps=1024, batch_size=128, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip=0.2, ent_coef=0.01, vf_coef=0.5`. **`ent_coef=0.01` es CRÍTICO** para evitar concentración en un activo (reporte §3.7).
- **Total timesteps**: 1M por seed.
- **Seeds**: 5 mínimo `{0, 1, 42, 123, 1337}`, objetivo 10 `{0, 1, 42, 123, 1337, 7, 31, 99, 256, 2024}`.
- Tracking: **MLflow + Hydra** (decisión confirmada).

### Conceptos clave que vamos a usar

- **Policy gradient**: familia de algoritmos que optimizan directamente la política `π(a|s)` por gradiente ascendente sobre el retorno esperado.
- **Clipping (PPO)**: limitar cuánto puede cambiar la política en un paso para evitar inestabilidad.
- **Entropía (`ent_coef`)**: bonus que premia políticas estocásticas, fomenta exploración. Valor alto → más diversidad de acciones.
- **Frame stacking**: concatenar las últimas N observaciones como input. Evita usar LSTM, más simple.
- **VecEnv**: envoltorio de SB3 que paraleliza N entornos para acelerar el rollout.
- **Rollout**: recopilar `n_steps` transiciones por entorno antes de hacer una actualización de gradiente.
- **Callback**: hook que se ejecuta cada cierto número de pasos. Lo usamos para logging, checkpointing y evaluación periódica.

### Tareas

| ID | Descripción | Archivo destino | Criterio de aceptación | Dependencias | Esfuerzo |
|---|---|---|---|---|---|
| **F6-T1** | Configurar PPO en YAML/Hydra (reporte §3.4): `learning_rate=3e-4, n_steps=1024, batch_size=128, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, total_timesteps=1_000_000`. `policy_kwargs: net_arch.pi=[128,128], net_arch.vf=[128,128], activation_fn=tanh, ortho_init=True`. **No reformular**. | `configs/ppo/default.yaml` | Config carga vía Hydra; valores literales del reporte | — | S |
| **F6-T2** | Configurar experimentos de Hydra: `configs/experiment/agent_a.yaml` (dataset = solo real) y `configs/experiment/agent_b.yaml` (dataset = real + sintético 1:2). Cada uno hereda de `ppo/default` y `env/portfolio_default`. | `configs/experiment/agent_a.yaml`, `configs/experiment/agent_b.yaml` | `python scripts/05_train_agent.py experiment=agent_a -m seed=0,1,42,123,1337` arranca | F6-T1 | S |
| **F6-T3** | Implementar `make_env(dataset, splits_idx, env_config) -> VecEnv` que: (a) instancia `PortfolioEnv`, (b) lo envuelve en `DummyVecEnv` (1 env paralelo es suficiente para datos diarios; 4 si Colab Pro), (c) aplica `VecFrameStack(n_stack=10)`, (d) opcionalmente `VecNormalize(norm_reward=True, norm_obs=False)` (state ya viene escalado). | `src/agents/env_factory.py` | Devuelve VecEnv compatible con SB3. `env.observation_space.shape` refleja el frame stacking | F5-T1 | M |
| **F6-T4** | Implementar **dataset mixer para Agente B**: clase `MixedDataset` que en cada `reset()` del env elige con probabilidad 1/3 una secuencia real (de train) o con probabilidad 2/3 una sintética. Mantener hash de seeds para reproducibilidad. **Solo se usa en train**; val y test son siempre reales cronológicos (reporte §4.2 punto 6). | `src/data/mixed_dataset.py` | Distribución de muestras tras 10000 reset coincide con las proporciones declaradas (chi-cuadrado test); val/test del agente B son idénticos a los del A | F4-T7, F3-T5 | M |
| **F6-T5** | Implementar wrapper `train_ppo_agent(experiment_config) -> trained_model` con MLflow tracking: log de params (todos los HP de ppo y env), log de metrics por step (`ep_rew_mean`, `ep_len_mean`, `policy_loss`, `value_loss`, `entropy`), log de artefactos (modelo, scalers, config Hydra, plots). | `src/agents/ppo_trainer.py` | Una corrida deja entrada en `mlruns/` con todos los artefactos; `mlflow ui` los muestra | F6-T3 | M |
| **F6-T6** | Implementar **callback de evaluación periódica** sobre el conjunto de **val** (2022): cada 50k steps, correr 1 episodio sobre val (real, sin sintéticos), calcular Sharpe, MDD, retorno; loggear en MLflow. **Esta es la métrica de selección de mejor checkpoint**, no `ep_rew_mean` del rollout. | `src/agents/callbacks.py::ValidationEvalCallback` | Logs muestran `val_sharpe`, `val_mdd`, `val_return` cada 50k steps; el callback persiste el modelo con mejor Sharpe en val | F6-T5 | M |
| **F6-T7** | Implementar **fijación de seeds globales** (reporte §5.7): `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, `gym.utils.seeding`, además del `seed=` de SB3 y `vec_env.seed(seed)`. Función única `set_global_seed(seed)`. | `src/utils/seeding.py` | Dos corridas con mismo seed producen las mismas métricas en val (con tolerancia 1e-4) | — | S |
| **F6-T8** | Crear script CLI `scripts/05_train_agent.py` que con Hydra acepta `experiment=agent_a` o `experiment=agent_b` y `seed=N`. Soporta multirun (`-m seed=0,1,42,123,1337`). Persiste cada modelo en `outputs/ppo_runs/<exp>/<seed>/model.zip`. | `scripts/05_train_agent.py` | `python scripts/05_train_agent.py -m experiment=agent_a,agent_b seed=0,1,42,123,1337` lanza 10 corridas | F6-T5, F6-T6, F6-T7 | M |
| **F6-T9** | **Igualación de timesteps reales** (reporte §7.5 checklist): el Agente B ve un dataset 3× más grande pero debe consumir **los mismos `total_timesteps=1M`** que el A. El factor diferencial es la diversidad de muestras, no el cómputo. Documentar y validar contando los `rollout/timesteps` de MLflow. | (validación dentro de F6-T8) | Para cada (experiment, seed), `total_timesteps` final ≈ 1M ± 2% en MLflow | F6-T8 | S |
| **F6-T10** | Entrenar **Agente A** con 5 seeds `{0, 1, 42, 123, 1337}`. Cómputo estimado: 5 × ~1.5h ≈ 7.5h GPU (Colab/Kaggle). Persistir todos los checkpoints. | (ejecución de F6-T8) | 5 modelos `.zip` + 5 entradas MLflow + curvas de val Sharpe estables o ascendentes | F6-T8, F4-T14 (gate sintéticos no requerido para A) | L |
| **F6-T11** | Entrenar **Agente B** con los mismos 5 seeds. Cómputo similar. **Pre-requisito**: gate de calidad de sintéticos aprobado (F4-T14). | (ejecución de F6-T8) | 5 modelos `.zip` + 5 entradas MLflow | F6-T8, F4-T14, F6-T4 | L |
| **F6-T12** | Test de comparabilidad A vs B: extraer hiperparámetros, arquitectura y total_timesteps de las 10 corridas; verificar que coinciden bit a bit excepto por `experiment` y `dataset`. | `tests/test_experiments.py::test_a_vs_b_comparable` | Test pasa | F6-T10, F6-T11 | S |

### Riesgos / pitfalls específicos

1. **Cherry-picking de seed** (reporte §7.1 pitfall #5): tentación de reportar solo el seed con mejores resultados. **Reportar siempre los 5–10 seeds** con media ± std + IC bootstrap.
2. **Agente concentra todo en 1 activo** (reporte §3.7): si pasa, subir `ent_coef` a 0.05; reducir `clip_range` a 0.1; aumentar `n_steps`. Diagnóstico: monitorear entropía en MLflow — si cae a 0, mala señal.
3. **Agente siempre en cash**: reward dominada por costes. Reducir bps; inicializar pesos cerca de equiponderado.
4. **Overfitting a train**: Sharpe(train) >> Sharpe(val). Reducir `n_epochs`, activar early stopping en val (lo hace F6-T6).
5. **Tuning con info del test**: NUNCA usar test para tuning. Hyperparam search (si se hace) sobre val con seed único; reportar test una sola vez al final con los seeds completos. Reporte §4.2 regla 4.
6. **Mixing real/sintético no estratificado**: si el sampleador devuelve 90% sintéticos por casualidad, el Agente B aprende del régimen sintético y se desvía. F6-T4 con muestreo controlado lo evita.
7. **Cómputo desbordado**: 5 seeds × 2 agentes × 1.5h = 15h GPU. Si se usa Colab Free (12h/sesión), partir en sesiones; checkpointing de SB3 lo permite. Considerar Kaggle (30h/semana free).

### Mapeo al cronograma UNIR

- **Semana 10–12** (post Borrador Intermedio).
- Para el **Borrador Intermedio (Sem. 9)** basta con **una corrida exitosa de A y otra de B con un único seed** como demostración de viabilidad ("primeros resultados" del Cap. 5 según el cronograma).
- Las 5 seeds completas para análisis estadístico se hacen entre Sem. 10 y 12 y se reportan en el **Borrador Final (Sem. 15)**.
- En la memoria, esta fase corresponde a la sección **"Fase 5: Entrenamiento de los agentes A y B"** del Cap. 5.

---

## Fase 7: Backtesting y evaluación

### ¿Qué es esta fase y por qué importa?

**Backtesting** = simular cómo se habría comportado una estrategia sobre datos históricos no vistos en el entrenamiento. En ML "tradicional" sería el equivalente a evaluar el modelo en el conjunto de test, pero en finanzas hay matices: la simulación debe ser **causal** (no usar info del futuro), debe incluir **costes reales**, y las **métricas son específicas del dominio** (Sharpe, MDD, etc., no `accuracy`).

Esta fase produce el **dato bruto** que la Fase 8 procesará estadísticamente. La regla cardinal es **el test es intocable**: no se mira hasta el final, no se usa para ajustar nada, y se reporta una sola vez (reporte §4.2). La forma de implementar esto es claramente: un script `06_backtest_agent.py` que recibe un modelo y un split (`val` o `test`), corre el agente determinístico (sin exploración) sobre todo el período, y produce una serie de retornos diarios + métricas.

### ¿Cómo se aplica a nuestro TFE?

Vamos a evaluar **2 agentes × 5 seeds = 10 modelos** sobre el mismo test real (2023-01-01 a 2025-04-30). Para cada modelo producimos: (1) serie temporal de retornos diarios, (2) curva de equity (wealth), (3) curva de drawdown, (4) tabla de **9 métricas**: annualized return, annualized vol, Sharpe, Sortino, MDD, Calmar, CVaR-95%, turnover anualizado, win rate. Adicionalmente, calculamos las métricas **condicionadas a sub-períodos de estrés** (los identificados en F2-T4) para mostrar si TimeGAN ayuda más en regímenes adversos.

Comparamos también contra **3 baselines pasivos** (reporte §7.2) para acknowledge survivorship bias: (a) buy-and-hold equiponderado de los 6 activos, (b) buy-and-hold S&P 500, (c) cartera 60/40 stocks/bonds (aproximada con el TNX como proxy del bond return).

### Conceptos clave que vamos a usar

- **Backtest causal**: en `t`, decisión solo con info ≤ `t-1`. La construcción del estado en F3-T9 ya garantiza esto si se respeta el shift.
- **Determinístico vs estocástico**: en evaluación se usa la media de la política gaussiana (`deterministic=True` en SB3), no muestreo.
- **Equity curve**: `V_0 → V_T` acumulando retornos. Estándar para comparar estrategias.
- **Recovery time**: días desde que se toca un MDD hasta volver al máximo previo. Mide resiliencia.
- **Win rate**: % de días con retorno > 0.

### Tareas

| ID | Descripción | Archivo destino | Criterio de aceptación | Dependencias | Esfuerzo |
|---|---|---|---|---|---|
| **F7-T1** | Implementar `backtest(model, env, deterministic=True) -> pd.DataFrame` que corre un modelo SB3 sobre el env (val o test) y devuelve dataframe con columnas: `[date, action_weights, portfolio_return, cost, wealth, drawdown]`. | `src/eval/backtest.py::run_backtest` | Para cualquier modelo entrenado, devuelve dataframe sin NaN; longitud = `len(test_idx)` | F6-T10, F5-T1 | M |
| **F7-T2** | Implementar batería de **9 métricas** (reporte §4.4) en `src/eval/metrics.py`: `annualized_return, annualized_vol, sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio, cvar_95, annualized_turnover, win_rate, recovery_time`. Cada función pura, recibe serie de retornos diarios. | `src/eval/metrics.py` | Tests F7-T8 pasan; valores razonables sobre serie de prueba conocida | — | M |
| **F7-T3** | Implementar baselines pasivos: (a) `equal_weight_6assets()` rebalancea diario a 1/6 cada activo; (b) `buy_and_hold_spy()`; (c) `portfolio_60_40()` con 60% equiponderado equities + 40% bond proxy (TNX returns aproximados). Cada uno produce un dataframe igual al de F7-T1 con costes idénticos (10 bps × turnover). | `src/eval/baselines.py` | 3 funciones; los 3 dataframes corren sobre el mismo test sin error | F1-T7 | M |
| **F7-T4** | Implementar `evaluate_runs(experiment, seeds, split='test') -> pd.DataFrame`: itera sobre los modelos guardados, corre F7-T1 y F7-T2, agrega resultados en una tabla `(seed × métrica)` por agente. Persistir en `outputs/eval/<experiment>_<split>.csv`. | `src/eval/runner.py::evaluate_runs` | Genera CSV con 5 filas (seeds) × 9 columnas (métricas) | F7-T1, F7-T2 | M |
| **F7-T5** | Implementar **evaluación condicionada por sub-períodos** de estrés (F2-T4): filtrar la serie de retornos por las fechas del YAML y recalcular las métricas dentro de cada período. Tabla `outputs/eval/<experiment>_stress_periods.csv`. | `src/eval/runner.py::evaluate_stress_periods` | Genera CSV con `(seed × período × métrica)` | F7-T4 | M |
| **F7-T6** | Implementar **plots comparativos** (reporte §6 sección plots): (a) equity curves A vs B (5 seeds c/u, IQR sombreado), (b) drawdown curves, (c) distribución de retornos diarios (KDE), (d) bar chart de métricas con error bars. Funciones puras que reciben dataframes y devuelven `matplotlib.Figure`. | `src/eval/plots.py` | 4 funciones; figuras PNG generadas en `outputs/plots/` para uso directo en Cap. 5–6 | F7-T4 | M |
| **F7-T7** | Crear script CLI `scripts/06_backtest_agent.py`: para cada modelo en `outputs/ppo_runs/`, corre F7-T4 y F7-T5 sobre val y test, persiste todo. Incluye baselines de F7-T3. Idempotente. | `scripts/06_backtest_agent.py` | Una ejecución produce todos los CSVs y plots | F7-T4, F7-T5, F7-T6 | M |
| **F7-T8** | Tests unitarios de métricas (reporte §6.6 punto 7): (a) retornos constantes positivos → Sharpe → ∞ (con `σ=ε`); (b) `Returns ~ N(μ, σ)` simulados → Sharpe ≈ `μ/σ × √252` con tolerancia; (c) MDD de serie monótonamente creciente = 0; (d) MDD de `[100, 110, 80, 90]` = -27.27%. | `tests/test_metrics.py` | `pytest tests/test_metrics.py` pasa | F7-T2 | M |

### Riesgos / pitfalls específicos

1. **Backtesting sobre train** (reporte §7.1 pitfall #8): nunca reportar resultados de train como "evidencia" de robustez. Solo test out-of-sample.
2. **Look-ahead en el estado durante backtest**: si el wrapper de evaluación re-calcula features con info futura, todo se invalida. F3-T11 debe pasar antes.
3. **Comparación contra baseline pasivo desfavorable**: si el equiponderado supera ambos agentes, el TFE tiene un resultado válido pero "negativo" (decir abierta y honestamente). Reporte §7.2.
4. **Olvido de costes en baselines**: equal_weight rebalancea diario, lo que genera turnover y costes no triviales — aplicar la misma fórmula que en el agente.
5. **Métrica única**: reportar solo Sharpe es pobre. La defensa del TFE es mucho más fuerte con MDD, Calmar, CVaR y métricas en sub-períodos de estrés.
6. **Determinístico mal**: olvidarse de `deterministic=True` introduce ruido entre evaluaciones. Verificar.

### Mapeo al cronograma UNIR

- **Semana 12–13** (mejoras del Borrador Intermedio + finalización del desarrollo).
- Entrega afectada: **Borrador Final (Sem. 15)** — los CSV y plots de esta fase son la materia prima del **Cap. 5 (resultados de la comparativa)** y del **Cap. 6 (discusión)**.
- En el cronograma UNIR, la "discusión y análisis de resultados" es la actividad central de la Sem. 13.

---

## Fase 8: Análisis estadístico comparativo

### ¿Qué es esta fase y por qué importa?

En cualquier estudio comparativo "B parece mejor que A en mi corrida favorita" no es una conclusión, es ruido. La **diferencia entre A y B necesita un test estadístico** que cuantifique la probabilidad de que la diferencia observada sea atribuible al azar (los seeds) vs. a una mejora real. Esto es lo que separa un TFE técnicamente sólido de uno mediocre (reporte §4.6, §7.5, §8 "Aspectos que distinguirán este TFE...").

Tres herramientas clave: (1) **bootstrap CI sobre la diferencia** — método no paramétrico que da un intervalo de confianza; si el IC no contiene 0, la diferencia es significativa. (2) **Welch's t-test** — paramétrico, asume normalidad aproximada. (3) **Diebold-Mariano test** — específico para comparar dos series temporales de retornos (no requiere asumir independencia entre seeds para los retornos diarios).

### ¿Cómo se aplica a nuestro TFE?

Implementaremos **los tres tests** sobre las métricas principales (Sharpe, Sortino, Calmar, MDD) y sobre las series de retornos diarios. Reportaremos siempre **media ± std + IC bootstrap 95%** y los p-values. Para múltiples comparaciones (varias métricas y/o sub-períodos) aplicaremos **corrección de Bonferroni** o **Benjamini-Hochberg** para controlar la tasa de falsos descubrimientos. Adicionalmente, el reporte (§7.5) recomienda **verificar persistencia en sub-períodos**: la mejora del Agente B debe ser consistente, no concentrada en un único período del test.

### Conceptos clave que vamos a usar

- **Bootstrap**: muestreo con reemplazo de un dataset; permite estimar la distribución de un estadístico sin asumir forma.
- **Intervalo de confianza (IC) 95%**: rango que con 95% de probabilidad contiene el verdadero parámetro poblacional.
- **p-value**: probabilidad de observar un efecto al menos tan grande como el observado bajo la hipótesis nula. Convención clásica: <0.05 = significativo.
- **Welch's t-test**: variante del t-test que no asume varianzas iguales entre grupos.
- **Diebold-Mariano (DM)**: test pareado sobre series temporales; compara accuracy/performance media entre dos modelos.
- **Bonferroni / Benjamini-Hochberg**: ajustes para múltiples tests; reducen la tasa de falsos positivos.
- **IQM (Interquartile Mean)**: media excluyendo el 25% mejor y peor; robusto. Recomendado en RL moderno (rliable).

### Tareas

| ID | Descripción | Archivo destino | Criterio de aceptación | Dependencias | Esfuerzo |
|---|---|---|---|---|---|
| **F8-T1** | Implementar `bootstrap_ci_diff(samples_a, samples_b, stat='mean', n_resamples=10000, ci=0.95) -> (low, high, p_value)`: bootstrap de la diferencia `B - A` y devuelve IC y p-value (proporción de bootstraps con `diff ≤ 0`). Usar `scipy.stats.bootstrap`. | `src/eval/stat_tests.py::bootstrap_ci_diff` | Para 5 muestras conocidas con diferencia clara, IC excluye 0 y p<0.05 | F7-T4 | M |
| **F8-T2** | Implementar `welch_ttest(samples_a, samples_b) -> (t_stat, p_value)` con check de normalidad previo (Shapiro-Wilk) y reportar warning si la asunción se viola. | `src/eval/stat_tests.py::welch_ttest` | Devuelve los mismos resultados que `scipy.stats.ttest_ind(equal_var=False)` | — | S |
| **F8-T3** | Implementar `diebold_mariano(returns_a, returns_b, h=1) -> (dm_stat, p_value)` sobre series de retornos diarios pareados (mismo período test). Implementación clásica de Harvey-Leybourne-Newbold. Usar paquete `arch` o implementación manual. | `src/eval/stat_tests.py::diebold_mariano` | Para 2 series idénticas devuelve p_value→1; para series con diferencia clara, p<0.05 | F7-T1 | M |
| **F8-T4** | Implementar **corrección por múltiples comparaciones** (Bonferroni y Benjamini-Hochberg) sobre lista de p-values cuando se evalúan varias métricas/sub-períodos. Usar `statsmodels.stats.multitest`. | `src/eval/stat_tests.py::adjust_pvalues` | Para 5 p-values, devuelve dataframe con `[metric, p_raw, p_bonferroni, p_bh]` | F8-T1, F8-T2, F8-T3 | S |
| **F8-T5** | Implementar **tabla resumen comparativa** A vs B: para cada métrica `[Sharpe, Sortino, Calmar, MDD, CVaR95, Turnover]` reportar `mean_a ± std_a, mean_b ± std_b, diff (B-A), IC95 bootstrap, p_value, p_bh_adj`. Persistir como Markdown y CSV. | `src/eval/stat_tests.py::summary_table_a_vs_b` | Tabla generada con 6 filas y 7 columnas; archivo legible directamente en el Cap. 6 | F8-T1, F8-T4, F7-T4 | M |
| **F8-T6** | Implementar **análisis de persistencia en sub-períodos** (reporte §7.5 punto 5): para cada sub-período de estrés, calcular `Sharpe_B - Sharpe_A` y reportar si el signo es consistente entre todos los sub-períodos. Tabla y/o forest plot. | `src/eval/stat_tests.py::stress_period_persistence`, `outputs/eval/persistence_plot.png` | Tabla con 3 filas (sub-períodos) × 3 cols (Sharpe_A, Sharpe_B, diff); plot tipo forest | F7-T5 | M |
| **F8-T7** | Crear script CLI `scripts/07_compare_agents.py` que orquesta F8-T1 a F8-T6 y genera el **informe final estadístico** en `outputs/reports/comparison_report.md`. Este informe es input directo del Cap. 6. | `scripts/07_compare_agents.py` | Una ejecución produce el Markdown + CSV + plots requeridos para Cap. 6 | F8-T5, F8-T6 | M |

### Riesgos / pitfalls específicos

1. **5 seeds insuficientes para tests paramétricos**: con n=5 por grupo, Welch t-test tiene poca potencia. Por eso priorizamos bootstrap (no asume normalidad) y reportamos IC además de p-value.
2. **p-hacking**: si reportamos 20 p-values sin corrección, esperamos 1 "significativo" por puro azar. Aplicar siempre Bonferroni o BH (F8-T4).
3. **Diebold-Mariano sobre retornos correlacionados serialmente**: usar la varianza HAC (Newey-West) si hay autocorrelación detectable. La implementación clásica ya lo contempla.
4. **Conclusión inflada**: si IC95 contiene 0 (no significativo), no decir "B es mejor". Decir "no se observa diferencia significativa con n=5 seeds; future work con 10+ seeds".
5. **Olvidar reportar effect size**: además de p-value, reportar `Cohen's d` o el % de mejora relativa. Significancia estadística sin magnitud práctica es engañoso.
6. **Mezclar seeds entre A y B distintos**: usar **los mismos seeds** para A y B en las comparaciones pareadas. Garantizado por F6-T2 + F8-T7.

### Mapeo al cronograma UNIR

- **Semana 13** (Discusión y análisis de resultados — actividad explícita del cronograma).
- Entrega afectada: **Borrador Final (Sem. 15)** — el output de F8-T7 alimenta directamente el **Cap. 6**.
- Esta fase corresponde a la última subsección de **"Fase 6: Validación"** del Cap. 5 (formal) y el grueso del **Cap. 6** (interpretación).

---

## Fase 9: Empaquetado, reproducibilidad y documentación

### ¿Qué es esta fase y por qué importa?

La diferencia entre un experimento que se publica y uno que muere en un notebook ad-hoc está en **la reproducibilidad**. Un proyecto reproducible permite que: (a) cualquiera con el repo pueda re-ejecutar el experimento end-to-end y obtener los mismos números (con tolerancia); (b) el director del TFE pueda inspeccionar el código y los artefactos sin pedir explicaciones; (c) el equipo pueda iterar sin miedo de romper algo. Las técnicas estándar son: **fijar seeds**, **persistir scalers y configs**, **versionar dependencias** (`requirements.txt` pinneado), **Docker** para entorno hermético, y **un README ejecutable** que documente la receta paso a paso.

Esta fase también incluye la **redacción técnica del Cap. 5 y Cap. 6** de la memoria — convertir los CSVs y plots en prosa académica defendible.

### ¿Cómo se aplica a nuestro TFE?

Tendremos un repo Git completo con todo lo anterior + un **`Dockerfile`** que reproduce el entorno + un **README** con receta para correr todo el pipeline en orden + el **scaler de TimeGAN persistido en joblib** para que cualquiera pueda re-generar sintéticos con la misma normalización + **configs Hydra versionados** en Git. Como bonus de defensa, generaremos un **MLflow snapshot** que el director puede abrir localmente para inspeccionar los 10 runs.

Para la memoria, redactaremos los **Cap. 4 (Planteamiento de la comparativa), Cap. 5 (Desarrollo) y Cap. 6 (Discusión)** con la estructura típica que pide la plantilla UNIR.

### Conceptos clave que vamos a usar

- **Reproducibilidad**: un tercero puede obtener los mismos resultados ejecutando los mismos pasos.
- **Pipeline idempotente**: re-ejecutar el script no produce efectos colaterales (sobrescribe limpio o no hace nada si ya existe el output).
- **Imagen Docker**: paquete que contiene el entorno completo (OS + libs + código) congelado.
- **Snapshot de experimentos**: carpeta `mlruns/` con todos los runs; se comparte como zip.

### Tareas

| ID | Descripción | Archivo destino | Criterio de aceptación | Dependencias | Esfuerzo |
|---|---|---|---|---|---|
| **F9-T1** | Crear `Dockerfile` con base `python:3.11-slim` (CPU) o `pytorch/pytorch:2.x-cuda12.1-cudnn8-runtime` (GPU). Instala `requirements.txt`. Define `WORKDIR /workspace` y `ENTRYPOINT ["python"]`. | `Dockerfile`, `.dockerignore` | `docker build .` se completa; `docker run --rm <img> --version` imprime versiones | F1-T2 | M |
| **F9-T2** | README maestro: descripción del proyecto, instrucciones de instalación (venv y Docker), receta paso a paso para reproducir todo el experimento (`scripts/01_…` a `scripts/07_…`), descripción de configs Hydra, link a la memoria PDF/DOCX, equipo y director. | `README.md` | Un lector externo puede correr el pipeline end-to-end siguiendo solo el README | F1-T7…F8-T7 | M |
| **F9-T3** | Verificar **persistencia y versionado** de todos los artefactos críticos: `data/processed/aligned.parquet`, `models/scalers/*.joblib`, `models/timegan/best.pt`, `outputs/ppo_runs/**/model.zip`, `mlruns/`, todos los YAMLs de Hydra. Añadir hashes (SHA256) en un archivo `MANIFEST.csv`. | `MANIFEST.csv`, `scripts/utils/generate_manifest.py` | Manifest con hashes de todos los artefactos versionables; integrado en CI mínima si se quiere | F8-T7 | S |
| **F9-T4** | Empaquetar **MLflow snapshot** y **plots finales** en un zip listo para entregar. Incluye `mlruns/`, `outputs/plots/`, `outputs/eval/*.csv`, `outputs/reports/comparison_report.md`. | `outputs/release/tfe_v1.zip` (script `scripts/utils/package_release.py`) | Zip <500MB; tras descomprimir y `mlflow ui --backend-store-uri ./mlruns` se ven los runs | F8-T7 | S |
| **F9-T5** | Redactar **Cap. 4 (Planteamiento de la comparativa)** de la memoria: descripción formal de Agente A vs B, criterios de éxito (las 9 métricas), pruebas de estrés (sub-períodos), tests estadísticos previstos. Usar el reporte §8 como referencia para tabla resumen "estructura de comparación". | Sección Cap. 4 en la memoria (`Tesis_v2.docx` o documento aparte que se merge después) | Capítulo redactado, ~6-10 páginas, con figura de la arquitectura del pipeline | F2-T6, F8-T5 | L |
| **F9-T6** | Redactar **Cap. 5 (Desarrollo de la comparativa)** de la memoria: subsecciones Adquisición de datos, Preprocesamiento, TimeGAN (con métricas de calidad y figuras de F4-T8/T10/T11), Entorno, Entrenamiento PPO (curvas de learning, configs), primeros resultados. **Citar el reporte como apéndice técnico**. | Sección Cap. 5 en la memoria | Capítulo redactado, ~15-25 páginas, con todas las figuras generadas en F4 y F7 incrustadas | F4-T11, F6-T11, F7-T7 | L |
| **F9-T7** | Redactar **Cap. 6 (Discusión y análisis)** de la memoria: interpretación de la tabla F8-T5, discusión de persistencia en sub-períodos (F8-T6), comparación contra baselines pasivos, **acknowledge explícito de limitaciones** (survivorship bias, número de seeds, calidad de TimeGAN), implicaciones para el sector. | Sección Cap. 6 en la memoria | Capítulo redactado, ~8-15 páginas; cumple con plantilla UNIR | F8-T7, F9-T5, F9-T6 | L |
| **F9-T8** | Redactar **Cap. 7 (Conclusiones y trabajo futuro)** + **Resumen/Abstract** + **Estructura del trabajo** (parte del Cap. 1) según cronograma Sem. 14. Future work: bootstrap baseline (apéndice opcional), TimeVAE/Diffusion, walk-forward CPCV, RecurrentPPO. | Secciones finales de la memoria | Capítulos completados; resumen ES (150–300 palabras) y abstract EN (150–300) | F9-T7 | M |

### Riesgos / pitfalls específicos

1. **Documentar al final**: tentación común. La memoria pide redactar en paralelo (Sem. 9 borrador intermedio). F9-T5/T6/T7 pueden empezar antes de tener todos los resultados — la estructura y descripción del método ya pueden documentarse en Sem. 9.
2. **Sin requirements pinneados**: `pip install pandas` sin versión hace que un mes después la receta no reproduzca. Pinnear todo al `==X.Y.Z`.
3. **Olvidar el `Adj Close` ajuste**: si yfinance cambia algoritmo de ajuste por splits, los datos del año que viene son distintos. El cache local + manifest de hashes mitiga.
4. **Memoria que excede 100 páginas**: la guía UNIR sugiere 75 páginas mínimo (TFE grupal); evitar inflar con redundancias. Cada figura justifica su espacio.
5. **Falta de citas correctas**: usar formato APA consistente. Verificar las referencias del Cap. 2 ya redactadas y mantener el mismo estilo.
6. **Repo público con secrets**: nunca subir API keys (Alpha Vantage, etc.) — usar `.env` ignorado por git.

### Mapeo al cronograma UNIR

- **Semana 14**: Cap. 7 + Resumen + Abstract + Estructura (cronograma explícito).
- **Semana 15**: Finalización y entrega del Borrador Final.
- En paralelo, F9-T5/T6 se empiezan en Sem. 9 (Borrador Intermedio) y se refinan en Sem. 12 (mejoras post-feedback).

---

## Plan de ejecución sugerido (Fases ML × Semanas UNIR)

Esta sección mapea el plan ML al cronograma oficial de UNIR. La idea es que para cada **entrega oficial** quede claro qué fases ML deben estar completas y con qué nivel de profundidad.

### Vista general

| Semana UNIR | Actividad UNIR | Fases ML activas | Estado esperado al cierre de la semana |
|---|---|---|---|
| **Sem. 7** | Desarrollo de la contribución (inicio) | F1, F2, F3 | Dataset descargado + EDA documentado + features y splits listos |
| **Sem. 8** | Desarrollo de la contribución (continuación) | F3 (cierre), F4 (núcleo), F5 (núcleo) | TimeGAN entrenando, env Gymnasium con tests pasando |
| **Sem. 9** | **ENTREGA 2 — BORRADOR INTERMEDIO** | F4 (cierre), F5 (cierre), F6 (1 seed demo), F9-T5/T6 (draft Cap. 4-5) | Sintéticos validados + 1 corrida A y 1 corrida B como "primer resultado" + Cap. 4 y 5 borrador |
| **Sem. 10** | Finalización tareas pendientes | F6 (5 seeds A) | 5 modelos del Agente A entrenados |
| **Sem. 11** | Entrega borrador intermedio (revisión con director) | F6 (5 seeds B) | 5 modelos del Agente B entrenados |
| **Sem. 12** | Aplicación de mejoras al Borrador Intermedio | F6 (cierre), F7 (núcleo) | Backtest completo de los 10 modelos sobre val y test |
| **Sem. 13** | Finalización del desarrollo + Cap. 6 | F8 (núcleo y cierre) | Comparación estadística completa, figuras finales para Cap. 6 |
| **Sem. 14** | Conclusiones, futuras líneas, abstract | F9-T7/T8 | Cap. 6 redactado + Cap. 7 + Resumen + Abstract |
| **Sem. 15** | **ENTREGA 3 — BORRADOR FINAL** | F9-T1/T3/T4 (release) | Repo Docker + manifest + zip release + memoria final |

### Detalle por entrega

#### Borrador Intermedio (Sem. 9) — checklist mínimo

Tu director verá:

- **Cap. 4 redactado** (~6-10 pp) con planteamiento formal de la comparativa, criterios de éxito y métricas (output de F9-T5 incipiente).
- **Cap. 5 con primeros resultados**:
  - Subsección Adquisición y EDA (F1, F2 cerradas).
  - Subsección Preprocesamiento (F3 cerrada).
  - Subsección TimeGAN (F4 cerrada con métricas de calidad reportadas; al menos 1 corrida pasando el gate F4-T14).
  - Subsección Entorno (F5 cerrada con tests pasando).
  - Subsección Entrenamiento (1 seed de Agente A y 1 seed de Agente B como demo, F6-T8 ejecutado dos veces).
- **Cap. 6 borrador** con la primera comparación A vs B sobre 1 seed (sin estadística aún — eso queda para sem. 13).

> **No es necesario** tener los 10 entrenamientos completos en Sem. 9. La revisión del director sirve para detectar desviaciones en alcance/método antes de invertir las ~15h de cómputo restantes.

#### Borrador Final (Sem. 15) — checklist completo

- 5 seeds completos para A y 5 para B (10 corridas) con MLflow tracking.
- Comparación estadística completa (F8-T5, F8-T6) con IC bootstrap, Welch t-test y Diebold-Mariano.
- Comparación contra los 3 baselines pasivos.
- Cap. 4, 5, 6, 7, Resumen, Abstract redactados.
- Repo Docker, MANIFEST, zip release.
- (Opcional) Bootstrap baseline ejecutado — ver Apéndice.

### Ruta crítica (cosas que bloquean todo lo demás)

1. **F1-T7 (descarga + alineación)** bloquea TODO. Hacerlo Sem. 7 día 1.
2. **F4-T14 (gate sintéticos)** bloquea F6-T11 (Agente B). Si TimeGAN no pasa el gate, hay que iterar la Fase 4 antes de poder entrenar B.
3. **F5 (entorno)** y **F6-T3/T5 (wrappers PPO)** bloquean cualquier entrenamiento. Tener el env con `check_env` aprobado antes de Sem. 9.
4. **F6-T10/T11 (entrenamientos)** son los que consumen GPU. Planificar en bloques de ~2h y lanzar en Kaggle/Colab Pro entre Sem. 10 y 12.

### Reparto del trabajo (referencia, plan agnóstico de owner)

Según la **Tabla 1 de la memoria** (no asignada en el plan, pero útil como referencia):

- **Diegbys (Alumno 1)**: protagonismo en **Fase 4 (TimeGAN)**.
- **Andrés (Alumno 2)**: protagonismo en **Fase 5–6 (Entorno y PPO)**.
- **Ambos en colaboración**: Fases 1, 2, 3, 7, 8, 9 (datos, eval, estadística, escritura).

Quedan pendientes de decidir entre ustedes en función de carga semanal.

### Estimación de esfuerzo total

| Fase | Esfuerzo persona-días estimado | Cómputo GPU estimado |
|---|---|---|
| F1 Adquisición | 1.5 | — |
| F2 EDA | 1 | — |
| F3 Preproc | 2.5 | — |
| F4 TimeGAN | 4 (con ablation 3 seeds) | ~3-12h |
| F5 Entorno | 2 | — |
| F6 PPO 5×A + 5×B | 3 (código) + cómputo | ~15-20h |
| F7 Backtest | 2 | ~1h |
| F8 Stats | 2 | — |
| F9 Empaquetado + memoria | 6 (escritura) | — |
| **Total** | **~24 persona-días** | **~20-35h GPU** |

Dividido entre 2 personas y ~9 semanas (Sem. 7-15) → ~1.3 días/persona/semana de promedio. Compatible con dedicación parcial. Picos en Sem. 8 (TimeGAN) y Sem. 10-12 (entrenamientos PPO).

---

## Apéndice A — Tareas opcionales: Bootstrap baseline (Agente B')

> **Estado**: opcional. Decidir en Sem. 13 si hay margen.
>
> **Justificación**: el reporte (§7.5, §8) recomienda como baseline crítico un Agente B' entrenado con `Real (train) + 2× Bootstrap` (circular block bootstrap, block_size=20). Permite distinguir si la mejora del Agente B viene de **modelado generativo** (TimeGAN) o de **simplemente más datos diversos** (cualquier aumento). Si `B ≈ B'`, la conclusión honesta es "el aumento es lo importante, TimeGAN no añade más"; si `B > B'`, la modelización generativa aporta valor real. Ambos resultados son académicamente válidos y publicables.

| ID | Descripción | Archivo destino | Criterio de aceptación | Dependencias | Esfuerzo |
|---|---|---|---|---|---|
| **OPT-T1** | Implementar circular block bootstrap (`block_size=20`) sobre las secuencias de log-returns reales: muestrear bloques contiguos con reemplazo y concatenarlos para formar series sintéticas de la misma longitud. Producir 2× del dataset original. | `src/generative/baselines/block_bootstrap.py` | Genera secuencias con la misma forma que las de TimeGAN; preserva ACF de retornos; valor de bloque configurable | F3-T10 | M |
| **OPT-T2** | Crear configuración `configs/experiment/agent_b_prime.yaml` (clon de `agent_b.yaml` que apunta a las secuencias bootstrap en lugar de TimeGAN). Mismos hiperparámetros que A y B. | `configs/experiment/agent_b_prime.yaml` | Hereda correctamente de PPO default + env default | F6-T2 | S |
| **OPT-T3** | Entrenar Agente B' con los mismos 5 seeds. Cómputo similar a F6-T11 (~7h GPU). | (ejecución de F6-T8 con `experiment=agent_b_prime`) | 5 modelos `.zip` + 5 entradas MLflow | OPT-T1, OPT-T2 | L |
| **OPT-T4** | Backtest del Agente B' (reusa F7-T7 con `experiment=agent_b_prime`). | `outputs/eval/agent_b_prime_test.csv` | CSV con métricas | OPT-T3, F7-T7 | S |
| **OPT-T5** | **Comparación tripartita A vs B' vs B**: extender F8-T5 a 3 columnas. Tests estadísticos por pares: A-vs-B', A-vs-B, B'-vs-B. Reportar en `outputs/reports/comparison_report_extended.md`. | `scripts/07_compare_agents.py --extended` | Tabla comparativa con 3 agentes y todas las métricas; Cap. 6 con discusión adicional | OPT-T4, F8-T7 | M |

**Trade-off**: añadir el bootstrap baseline cuesta ~5-7h de cómputo extra y ~2-3 días persona, pero **eleva sustancialmente el rigor** del TFE (criterio "Aspectos que distinguirán este TFE de un trabajo mediocre", reporte §8).

---

## Apéndice B — Resumen final del plan

- **Tareas totales**: **86 obligatorias** + **5 opcionales (Apéndice A)** = **91 tareas**.
- **Distribución por fase**: F1=8, F2=7, F3=12, F4=14, F5=10, F6=12, F7=8, F8=7, F9=8, OPT=5.
- **Estructura**: cada fase con explicación didáctica, conceptos clave glosados la primera vez, tareas atómicas con ID, archivo destino, criterio de aceptación, dependencias y esfuerzo S/M/L.
- **Tracking stack**: Hydra + MLflow + Tensorboard.
- **Test stack**: pytest con tests anti-leakage explícitos (F1-T8, F3-T8, F3-T11, F4-T12, F5-T7..T10, F7-T8) — 9 tests críticos.
- **Cómputo total estimado**: 20-35h GPU (T4 / Colab Pro / Kaggle).
- **Esfuerzo persona-días**: ~24 entre dos personas, 9 semanas → carga sostenida con dos picos (Sem. 8 TimeGAN, Sem. 10-12 entrenamientos).
- **Ruta crítica**: F1-T7 → F3-T10 → F4-T14 → F6-T11 → F7-T7 → F8-T7 → F9-T4. Cualquier retraso aquí impacta directo en la entrega.
- **Mapeo a entregas UNIR**:
  - **Borrador Intermedio (Sem. 9)**: F1, F2, F3 cerradas; F4, F5 cerradas con primer resultado; F6 con 1 corrida demo de A y B.
  - **Borrador Final (Sem. 15)**: las 9 fases completas + Apéndice opcional ejecutado o documentado como future work.

---

## Apéndice C — Progress Tracker (checkboxes)

Marcá `[x]` (sustituyendo el espacio dentro de los corchetes por una `x`) cuando una tarea esté completa. Esta sección es el "tablero" del proyecto: una vista única para ver de un vistazo qué falta.

> Tip: en VS Code / Cursor / GitHub puedes hacer click directo en el checkbox; en otros editores Markdown basta con cambiar `[ ]` por `[x]`.

### Hitos macro

- [ ] **Hito 1 — Borrador Intermedio (Sem. 9)** entregado
- [ ] **Hito 2 — Borrador Final (Sem. 15)** entregado
- [ ] **Hito 3 — Predepósito** preparado
- [ ] **Hito 4 — Defensa** preparada

### Fase 1 — Adquisición de datos `[x] FASE COMPLETA`

- [x] **F1-T1** — Estructura de directorios + repo Git inicializado
- [x] **F1-T2** — Stack pinneado en `pyproject.toml` / `requirements.txt`
- [x] **F1-T3** — Descarga yfinance 6 activos → Parquet en `data/raw/equities/`
- [x] **F1-T4** — Fallback Stooq operativo
- [x] **F1-T5** — Descarga macro (VIX, ^TNX, DXY) → Parquet
- [x] **F1-T6** — Alineación al calendario XNYS
- [x] **F1-T7** — Script CLI `01_download_data.py` end-to-end
- [x] **F1-T8** — Test `test_aligned_dataset` pasa

### Fase 2 — EDA y entendimiento del dominio `[x] FASE COMPLETA`

- [x] **F2-T1** — Notebook con series, estadísticas, distribuciones, NaN
- [x] **F2-T2** — Stylized facts en datos reales (CSV + figs)
- [x] **F2-T3** — Análisis macro y regímenes (VIX, TNX, DXY)
- [x] **F2-T4** — Sub-períodos de estrés documentados en YAML
- [x] **F2-T5** — Matriz de correlaciones train vs test
- [x] **F2-T6** — Resumen EDA `eda_summary.md`
- [x] **F2-T7** — Test `test_real_data_has_fat_tails` pasa

### Fase 3 — Preprocesamiento e ingeniería de features `[x] FASE COMPLETA`

- [x] **F3-T1** — Log-returns + body/range/log_volume
- [x] **F3-T2** — Set de indicadores técnicos (`ta` Bukosabino — pandas-ta exige Py>=3.12); +EMA(60)
- [x] **F3-T3** — Volatilidad realizada (5d, 21d, ratio)
- [x] **F3-T4** — Macro features estacionarias (+ ADF informativo)
- [x] **F3-T5** — Split cronológico 70/10/20 persistido (1704/251/583)
- [x] **F3-T6** — MinMaxScaler TimeGAN ajustado solo en train (9 cols)
- [x] **F3-T7** — RobustScaler PPO ajustado solo en train (139 cols)
- [x] **F3-T8** — Test `test_scaler_only_train` pasa
- [x] **F3-T9** — `build_state` función pura
- [x] **F3-T10** — Secuencias TimeGAN `(1681, 24, 9)` persistidas en `.npy`
- [x] **F3-T11** — Test `test_no_lookahead_in_features` pasa (10 fechas test aleatorias)
- [x] **F3-T12** — Script CLI `02_build_features.py` + `outputs/preproc/preproc_summary.md`

### Fase 4 — Modelado generativo (TimeGAN) `[ ] FASE COMPLETA`

- [ ] **F4-T1** — Redes E/R/G/S/D en PyTorch
- [ ] **F4-T2** — Loop entrenamiento en 3 fases (embedding/supervised/joint)
- [ ] **F4-T3** — Config Hydra hiperparámetros TimeGAN
- [ ] **F4-T4** — Early stopping basado en discriminative score
- [ ] **F4-T5** — Función `generate_synthetic`
- [ ] **F4-T6** — Reconstrucción de precios sintéticos
- [ ] **F4-T7** — Dataset sintético con indicadores recalculados
- [ ] **F4-T8** — Métrica discriminative score implementada
- [ ] **F4-T9** — Métrica predictive score (TSTR) implementada
- [ ] **F4-T10** — Visualización t-SNE / PCA
- [ ] **F4-T11** — Batería de stylized facts sintéticos
- [ ] **F4-T12** — Test `test_timegan_train_only` pasa
- [ ] **F4-T13** — Script `03_train_timegan.py` con multirun
- [ ] **F4-T14** — Gate de calidad superado (`QUALITY_OK.flag`)

### Fase 5 — Diseño del entorno de trading (Gymnasium) `[ ] FASE COMPLETA`

- [ ] **F5-T1** — Clase `PortfolioEnv` con API Gymnasium
- [ ] **F5-T2** — Proyección al simplex via softmax
- [ ] **F5-T3** — Drift de pesos correcto
- [ ] **F5-T4** — Cálculo de costes de transacción
- [ ] **F5-T5** — Función de recompensa
- [ ] **F5-T6** — Config Hydra del entorno
- [ ] **F5-T7** — Test compliance Gymnasium (`check_env`) pasa
- [ ] **F5-T8** — Test `test_action_space` pasa
- [ ] **F5-T9** — Test `test_reward_consistency` pasa
- [ ] **F5-T10** — Test `test_transaction_costs` pasa

### Fase 6 — Entrenamiento PPO (Agente A y Agente B) `[ ] FASE COMPLETA`

- [ ] **F6-T1** — Config Hydra hiperparámetros PPO
- [ ] **F6-T2** — Configs experimentos `agent_a` y `agent_b`
- [ ] **F6-T3** — Factory `make_env` con `VecFrameStack`
- [ ] **F6-T4** — `MixedDataset` real+sintético operativo
- [ ] **F6-T5** — Wrapper `train_ppo_agent` con MLflow
- [ ] **F6-T6** — Callback evaluación periódica sobre val
- [ ] **F6-T7** — `set_global_seed` reproducible
- [ ] **F6-T8** — Script `05_train_agent.py` con Hydra multirun
- [ ] **F6-T9** — Igualación de timesteps verificada
- [ ] **F6-T10** — **Agente A**: 5 seeds entrenados (modelos persistidos)
- [ ] **F6-T11** — **Agente B**: 5 seeds entrenados (modelos persistidos)
- [ ] **F6-T12** — Test `test_a_vs_b_comparable` pasa

### Fase 7 — Backtesting y evaluación `[ ] FASE COMPLETA`

- [ ] **F7-T1** — Función `run_backtest` causal
- [ ] **F7-T2** — Batería de 9 métricas implementada
- [ ] **F7-T3** — Baselines pasivos (equiponderado, S&P 500, 60/40)
- [ ] **F7-T4** — `evaluate_runs` produce CSV por agente
- [ ] **F7-T5** — Evaluación condicionada por sub-períodos de estrés
- [ ] **F7-T6** — Plots comparativos generados
- [ ] **F7-T7** — Script `06_backtest_agent.py`
- [ ] **F7-T8** — Tests `test_metrics.py` pasan

### Fase 8 — Análisis estadístico comparativo `[ ] FASE COMPLETA`

- [ ] **F8-T1** — `bootstrap_ci_diff` implementado
- [ ] **F8-T2** — `welch_ttest` con check de normalidad
- [ ] **F8-T3** — `diebold_mariano` sobre series pareadas
- [ ] **F8-T4** — Corrección Bonferroni / Benjamini-Hochberg
- [ ] **F8-T5** — Tabla resumen A vs B (Markdown + CSV)
- [ ] **F8-T6** — Análisis de persistencia en sub-períodos
- [ ] **F8-T7** — Script `07_compare_agents.py` + `comparison_report.md`

### Fase 9 — Empaquetado, reproducibilidad y documentación `[ ] FASE COMPLETA`

- [ ] **F9-T1** — Dockerfile + `.dockerignore`
- [ ] **F9-T2** — README maestro reproducible end-to-end
- [ ] **F9-T3** — `MANIFEST.csv` con hashes SHA256
- [ ] **F9-T4** — Zip release con MLflow snapshot
- [ ] **F9-T5** — Cap. 4 (Planteamiento de la comparativa) redactado
- [ ] **F9-T6** — Cap. 5 (Desarrollo) redactado
- [ ] **F9-T7** — Cap. 6 (Discusión y análisis) redactado
- [ ] **F9-T8** — Cap. 7 + Resumen + Abstract redactados

### Apéndice A — Bootstrap baseline (Agente B') *opcional*

- [ ] **OPT-T1** — Circular block bootstrap implementado
- [ ] **OPT-T2** — Config `agent_b_prime.yaml`
- [ ] **OPT-T3** — Agente B' entrenado (5 seeds)
- [ ] **OPT-T4** — Backtest del Agente B'
- [ ] **OPT-T5** — Comparación tripartita A vs B' vs B documentada

### Suite anti-leakage (vista cruzada — referencia)

Estas tareas son los tests que protegen la validez del experimento. Cuando todas estén marcadas, el TFE está blindado contra los errores típicos de backtesting de DRL (reporte §7.1):

- [x] **F1-T8** — Test dataset alineado
- [x] **F2-T7** — Test stylized facts datos reales
- [x] **F3-T8** — Test scaler solo en train
- [x] **F3-T11** — Test no look-ahead en features
- [ ] **F4-T12** — Test TimeGAN solo train
- [ ] **F5-T7** — Test compliance Gymnasium
- [ ] **F5-T8** — Test action space
- [ ] **F5-T9** — Test reward consistency
- [ ] **F5-T10** — Test transaction costs
- [ ] **F6-T12** — Test A vs B comparables
- [ ] **F7-T8** — Tests métricas contra valores conocidos

---

*Fin del plan. Cualquier cambio se versiona en Git junto al código del repo.*
