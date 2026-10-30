# Contrato de datos entre ML y Dashboard (capa Gold)

**Versión:** 1.1 · **Fecha:** 2026-10-30 · **Responsable ML:** Hikari · **Consumidor:** Frontend (Streamlit)
**Decisiones relacionadas:** DEC-12, DEC-14, DEC-41, DEC-42, DEC-43, DEC-47, DEC-48, DEC-49

El dashboard **solo lee ficheros Parquet de `data/gold/`**. No carga modelos, no lee Bronze ni Silver y no accede a fuentes externas (NF2, apartado 5.6). Este documento fija qué garantiza Gold y qué debe hacer el dashboard con cada caso.

---

## 1. Ficheros

| Fichero | Contenido | Filas esperadas |
|---|---|---|
| `data/gold/predicciones.parquet` | Una predicción vigente por municipio y fecha | Hasta 313 por fecha |
| `data/gold/explicaciones.parquet` | Las 5 variables con más peso por predicción (SHAP) | 5 por predicción |

Ambos se regeneran con `python -m src.ml.inferir`. Tras cada regeneración se comprueban con `verificar_salidas.py`.

### `predicciones.parquet`

| Columna | Tipo | Descripción |
|---|---|---|
| `id_prediccion` | texto | `<id_municipio>_<fecha_objetivo>_<version_modelo>`, p. ej. `32031_2026-10-19_v1`. Clave primaria |
| `id_municipio` | texto | Código INE de 5 cifras. **Es texto**: no convertir a número (se pierden los ceros iniciales) |
| `fecha_objetivo` | fecha | Día D+1 al que se refiere la predicción. Convertir con `pd.to_datetime(...).dt.normalize()` |
| `version_modelo` | texto | `v1` (modelo principal, con FWI) o `v1-respaldo` (modelo sin FWI) |
| `probabilidad` | decimal 0–1 | Índice de riesgo del modelo (ver 4.3) |
| `nivel_riesgo` | texto | `bajo`, `medio` o `alto` |
| `fwi` | decimal | FWI del día D, solo como referencia. **Es nulo en las filas de respaldo** |
| `fecha_insert` | timestamp UTC | Cuándo se generó la fila |

### `explicaciones.parquet`

| Columna | Tipo | Descripción |
|---|---|---|
| `id_prediccion` | texto | Clave foránea a `predicciones` |
| `variable` | texto | Nombre de la variable (ver tabla de etiquetas en 5.2) |
| `valor_shap` | decimal | Contribución en **log-odds**. Positivo = sube el riesgo; negativo = lo baja |

**Compatibilidad:** el dashboard debe ignorar las columnas que no conozca y no depender del orden de las columnas. Se pueden añadir columnas (ver sección 7) sin avisar de ruptura.

---

## 2. Qué garantiza Gold

1. **Una sola predicción vigente por (`id_municipio`, `fecha_objetivo`).** Si llega el FWI de unos días predichos con el respaldo, `v1` sustituye a `v1-respaldo` (DEC-48).
2. `id_prediccion` es único, y toda predicción tiene exactamente 5 explicaciones (y al revés: no hay explicaciones huérfanas).
3. `probabilidad` está en [0, 1] y no tiene nulos. `nivel_riesgo` solo toma `bajo`, `medio` o `alto`, coherente con los umbrales guardados del modelo que generó la fila.
4. Las filas `v1` tienen `fwi`; las filas `v1-respaldo` no lo tienen y sus explicaciones no usan variables del FWI.
5. **No se predice fuera del dominio del modelo.** Si faltan datos, simplemente no hay fila (ver 3).

**Red de seguridad recomendada** si, aun así, apareciera un duplicado: gana `v1` sobre `v1-respaldo`, y entre iguales la `fecha_insert` más reciente.

```python
import pandas as pd

pred = pd.read_parquet("data/gold/predicciones.parquet")
pred["fecha_objetivo"] = pd.to_datetime(pred["fecha_objetivo"]).dt.normalize()
pred["_resp"] = pred["version_modelo"].str.endswith("-respaldo")
pred = (pred.sort_values(["id_municipio", "fecha_objetivo", "_resp", "fecha_insert"],
                         ascending=[True, True, True, False])
            .drop_duplicates(["id_municipio", "fecha_objetivo"])
            .drop(columns="_resp"))
```

---

## 3. Casos límite y qué mostrar

| Situación | Cómo se detecta | Qué mostrar |
|---|---|---|
| Municipio sin fila en la fecha | No hay fila para ese `id_municipio` y `fecha_objetivo` | **«Sin dato»**: categoría propia, distinta de «bajo». Gris con patrón o icono (no solo color). Texto: «Sin datos suficientes ese día» |
| Predicción con el modelo de respaldo | `version_modelo` termina en `-respaldo` | Etiqueta visible **«Estimación sin FWI»**. FWI como «no disponible». No incluirla en comparaciones modelo frente a FWI |
| Fecha sin ninguna fila | Filtro vacío | «No hay predicciones para esa fecha». Ofrecer la última fecha disponible |
| Fecha posterior a la última disponible | `fecha > pred["fecha_objetivo"].max()` | Igual que el caso anterior |
| Día con riesgo generalizado | ≥ 90 % de los municipios de la fecha en `medio` o `alto` | Comportamiento normal en episodios de sequía (DEC-49). Mostrar un aviso informativo tipo «Episodio de riesgo elevado en toda la región», no un error |
| Predicción sin explicaciones | Filtro vacío en `explicaciones` | Ocultar el panel SHAP con un mensaje; no debería ocurrir |

**Cobertura real hoy:** 313 municipios hasta el 19/10/2026 y 288 los días 20, 21 y 22 (25 municipios sin ninguna variable meteorológica). La cobertura mínima aceptada es del 80 %.

### Mapa con todos los municipios

El mapa debe pintar los 313 municipios (HU-10). Parte del catálogo completo y cruza con las predicciones del día:

```python
dia = pred[pred["fecha_objetivo"] == fecha]
mapa = catalogo.merge(dia, on="id_municipio", how="left")      # catalogo: 313 municipios
mapa["nivel_mostrado"] = mapa["nivel_riesgo"].fillna("sin dato")
```

---

## 4. Cómo presentar el riesgo

### 4.1 Niveles

Usa siempre `nivel_riesgo`. **No recalcules niveles a partir de la probabilidad**: cada modelo tiene sus propios umbrales.

| Nivel | Principal (`v1`) | Respaldo (`v1-respaldo`) | Significado operativo |
|---|---|---|---|
| medio o superior | p ≥ 0,0273 | p ≥ 0,0220 | Recall 0,80 en validación 2022; 0,73 en el test de 2023 (intervalo 0,62-0,81) |
| alto | p ≥ 0,0449 | p ≥ 0,0241 | Recall 0,50 en validación 2022 (no medido en el test) |

Estos umbrales son los de DEC-41 y **pueden cambiar** con DEC-49. Por eso el dashboard no debe fijarlos en el código.

**Texto sugerido** para la leyenda: «*Medio o alto* significa *no descartar*, no *habrá un incendio*. Entre 1 de cada 16 y 1 de cada 50 alarmas coincide con un incendio, según el año». (Precision de 2,0 % en el test de 2023 y de 2,3 % a 6,1 % en el walk-forward 2016-2022; depende de la tasa base del año).

**Accesibilidad (NF2, apartado 9):** el nivel debe leerse sin depender del color. Usa siempre el texto del nivel junto al color, y paleta apta para daltonismo. Propuesta (Okabe-Ito, la misma familia que ya usan los gráficos SHAP): bajo `#0072B2`, medio `#F0E442`, alto `#D55E00`, sin dato gris `#999999`.

### 4.2 Respaldo

Las filas `v1-respaldo` tienen **menos capacidad de discriminar** (equivalen al FWI, DEC-47) y sus niveles cambian con diferencias de 0,001 en la probabilidad. Deben distinguirse visualmente de las del modelo principal (por ejemplo, borde punteado) y llevar siempre la etiqueta «Estimación sin FWI».

### 4.3 Probabilidad

- Mostrarla como **«índice de riesgo»** (por ejemplo, «Índice de riesgo: 4,3 %»), **siempre junto al nivel**.
- No presentarla como probabilidad exacta: en validación 2022 el modelo sobrestimó la tasa real ~1,5 veces, y la tasa base cambia de un año a otro.
- No comparar índices entre años ni entre `v1` y `v1-respaldo`.

### 4.4 Prioridad relativa dentro del día (complementaria, provisional)

Sirve para ordenar municipios de un mismo día, sobre todo en días de riesgo generalizado. Se calcula en el dashboard, por fecha y por versión:

```python
dia = dia.assign(prioridad_dia = 1 - dia.groupby("version_modelo")["probabilidad"].rank(pct=True))
# 0 = el municipio de mayor riesgo del día; 0,10 = dentro del 10 % superior
```

Mostrarla como «Prioridad relativa hoy: top 12 %». **No usarla para definir el nivel.** En días saturados aporta señal modesta (lift 1,4–1,7 en validación). Su inclusión depende del cierre de DEC-49.

### 4.5 Fecha por defecto y frescura

- La fecha por defecto es `pred["fecha_objetivo"].max()`, **no la fecha de hoy**: los datos llegan con retraso y la última fecha puede ir varios días por detrás.
- Mostrar siempre «Última fecha con predicción: dd/mm/aaaa».
- Una predicción para `fecha_objetivo` usa información hasta el día anterior; **no es una previsión meteorológica**.

---

## 5. Explicaciones SHAP

### 5.1 Qué contienen y qué no

- Solo las **5 variables con mayor |SHAP|** de cada predicción, en log-odds.
- **No incluyen el valor de la variable** (no se puede mostrar «humedad = 38 %») **ni el valor base**, así que la suma de barras no reconstruye la probabilidad. Esto se ampliará (sección 7).
- Las variables del FWI (`ffmc`, `dmc`, `dc`, `isi`, `bui`, `fwi`) están correlacionadas: SHAP reparte el efecto entre ellas. Conviene mostrarlas como «Condiciones de sequía (índices FWI)» o aclarar que no son causas independientes.

### 5.2 Etiquetas legibles

| `variable` | Etiqueta sugerida |
|---|---|
| `fwi` | Índice meteorológico de incendio (FWI) |
| `ffmc` | Humedad del combustible fino (FFMC) |
| `dmc` | Humedad de la capa orgánica superficial (DMC) |
| `dc` | Sequía de capas profundas (DC) |
| `isi` | Potencial de propagación inicial (ISI) |
| `bui` | Disponibilidad de combustible (BUI) |
| `temperatura` | Temperatura |
| `humedad_relativa` | Humedad relativa |
| `precipitacion` | Precipitación del día |
| `racha_dias_secos` | Días consecutivos sin lluvia |
| `precip_acum_7d` | Precipitación acumulada en 7 días |
| `mes` | Época del año |

### 5.3 Gráfico

`src/explainability/plots.py` ya genera la figura a partir de las tablas de Gold, sin cargar el modelo:

```python
from src.explainability.plots import fig_local

expl = pd.read_parquet("data/gold/explicaciones.parquet",
                       filters=[("id_prediccion", "==", id_pred)])
fig = fig_local(expl, id_pred, "log-odds")
fig.update_layout(title=f"{nombre} · {fecha:%d/%m/%Y}")     # el título por defecto no incluye municipio ni fecha
```

La figura ya cumple la accesibilidad: el sentido de cada barra se indica también con texto («+0,40 (sube el riesgo)»). En el dashboard, sustituir el eje «log-odds» por un texto no técnico, p. ej. «Contribución al índice de riesgo».

### 5.4 Importancia global

`shap_global.csv` / `shap_global.html` es un **fichero estático** generado con `inferir --ejemplo-dir`, calculado sobre la validación 2022. No está en Gold y no cambia con cada predicción. Su ubicación está por acordar (sección 8).

---

## 6. Rendimiento (HU-09: menos de 5 s)

- Cachear la lectura (`st.cache_data`) y leer solo lo necesario: `columns=[...]` y `filters=[("fecha_objetivo", "==", fecha)]` en `predicciones`, y filtrar `explicaciones` por `id_prediccion`.
- `explicaciones.parquet` tiene unos 5 registros por predicción (más de 1,5 millones de filas con el histórico actual): no cargarlo entero para consultar un municipio.
- Estas recomendaciones no se han medido; si no se llega a los 5 s, avisar a ML.

---

## 7. Cambios previstos (aditivos)

| Cambio | Origen | Efecto en el dashboard |
|---|---|---|
| Columnas `valor_variable` y `valor_base` en `explicaciones.parquet` | Pendiente de DEC-43 | Permitirá mostrar el valor real de cada variable y reconstruir la probabilidad. No rompe nada |
| Nuevos umbrales o forma de definir los niveles | DEC-49 | Cambian los valores de `nivel_riesgo`, no las columnas. No fijar umbrales en el código |
| `version_modelo` distinta de `v1` (p. ej. `v2`) tras reentrenar | HU-11 | No suponer que solo existe `v1`; tratar como respaldo todo lo que termine en `-respaldo` |
| Columna indicando el modelo de respaldo | Posible | Hoy se deduce del sufijo de `version_modelo` |

---

## 8. Pendiente de acordar con Frontend

1. **Nombres y geometría de municipios.** Gold no los incluye (sí `id_municipio`). Hay que decidir de dónde salen (catálogo INE/CNIG, GeoJSON en el repositorio) para el mapa y el buscador por nombre.
2. **Ubicación de los ficheros de evaluación** (`informe_final.csv`, `informe_walkforward.csv`): se proponen en `docs/resultados/`. Su esquema y cómo leerlos están en la sección 11.
3. **Ubicación de la importancia global** (`shap_global.csv` / `.html`).
4. **Política de niveles y prioridad relativa** (DEC-49), que decide si hay que mostrar `prioridad_dia`.
5. **Aviso de episodio de riesgo generalizado:** umbral del 90 % y redacción.

---

## 9. Textos obligatorios

- **Pie permanente:** «Herramienta experimental de apoyo. No sustituye a los sistemas oficiales de predicción ni de gestión de emergencias. Las estimaciones son probabilísticas y no garantizan que vaya a producirse un incendio».
- **Fuentes (licencias):** «Fuente: AEMET. Origen de los datos: Ministerio para la Transición Ecológica y el Reto Demográfico. Copernicus/EFFIS. Fuente: INE. © Instituto Geográfico Nacional». Los organismos no respaldan el proyecto (NF2, apartado 8).
- **Evitar la estigmatización** de municipios: mostrar siempre probabilidad, nivel y explicación, nunca solo un ranking.

---

## 10. Criterios de aceptación

| Historia | Comprobación |
|---|---|
| HU-09 | Buscar un municipio por nombre y una fecha muestra nivel (con texto), índice de riesgo, FWI y variables en menos de 5 s |
| HU-10 | El mapa colorea los 313 municipios con leyenda apta para daltonismo; los municipios sin fila aparecen como «Sin dato»; hay tabla equivalente |
| Respaldo | Las filas `v1-respaldo` llevan la etiqueta «Estimación sin FWI» y FWI «no disponible» |
| Casos límite | Fecha sin datos, fecha futura y predicción sin explicaciones no producen errores |
| Smoke test | El dashboard arranca y lee `predicciones.parquet` (NF2, apartado 12) |

---

## 11. Resultados de evaluación (panel «modelo frente a FWI»)

`predicciones` solo trae el FWI del día D. La comparación del modelo con el baseline (HU-07) sale de dos ficheros estáticos que genera ML, **fuera de Gold** (ubicación propuesta: `docs/resultados/`; por confirmar). No cambian con cada predicción.

### 11.1 `informe_final.csv`: evaluación por escenario

| Columna | Descripción |
|---|---|
| `escenario` | `A-test` (municipios vistos, año 2023), `B` (municipios no vistos, 2009-2021), `C-test` (no vistos y 2023), `B-todos-los-anios` (como B, incluyendo 2001-2008) |
| `variante` | `con_fwi-sin_viento-desde2009` o `sin_fwi-sin_viento-desde2009` |
| `grupo` | `global` o la provincia |
| `sistema` | `modelo` o `baseline_fwi`. Comparar siempre las dos filas con el mismo escenario, variante y grupo |
| `n`, `n_pos`, `prevalencia` | Filas evaluadas, días con incendio y su proporción |
| `auc_pr`, `auc_pr_lo`, `auc_pr_hi` | AUC-PR con su intervalo (bootstrap por bloques de 7 días) |
| `recall`, `recall_lo`, `recall_hi`, `precision` | Con el umbral de la columna `umbral` |
| `umbral` | Del modelo: **probabilidad**. Del baseline: **valor del FWI** (4,22). No mezclar escalas |
| `dif_auc_pr`, `dif_lo`, `dif_hi`, `p_mejora` | Diferencia pareada modelo − FWI. **Solo en la fila `modelo` con `grupo = global`**. `p_mejora` no es un p-valor (proporción de remuestreos en que el modelo supera al FWI; a confirmar con ML) |

`informe_walkforward.csv` repite lo esencial por año (2016-2022): `anio`, `variante`, `sistema`, `n`, `n_pos`, `prevalencia`, `auc_pr` (con intervalo), `recall`, `precision` y la diferencia frente al FWI.

### 11.2 Resumen de resultados (solo contexto: leerlo del fichero, no fijarlo en el código)

| Escenario | Prevalencia | AUC-PR modelo `con_fwi` | AUC-PR FWI | Diferencia (intervalo) |
|---|---|---|---|---|
| A-test | 0,75 % | 0,0276 | 0,0193 | +0,0084 (+0,0011 a +0,0195) |
| B | 1,81 % | 0,0935 | 0,0692 | +0,0243 (+0,0147 a +0,0345) |
| C-test | 0,74 % | 0,0381 | 0,0212 | +0,0169 (+0,0028 a +0,0369) |

El modelo `sin_fwi` (el que genera las filas `v1-respaldo`) **no mejora** al FWI en ningún escenario global (A-test +0,0027, B +0,0002, C-test +0,0007; los intervalos incluyen el cero).

### 11.3 Cómo presentarlo

1. **Mostrar siempre la prevalencia y el azar** junto al AUC-PR: un AUC-PR de 0,03 parece bajo pero equivale a 3,7 veces el azar (0,75 % en 2023). Sin ese contexto se malinterpreta.
2. **Mostrar los intervalos.** Las provincias tienen pocos incendios en 2023 y sus intervalos son muy amplios; no resaltar diferencias entre provincias sin ellos.
3. **No ocultar A Coruña.** El modelo no supera al FWI allí (con `con_fwi`: AUC-PR 0,0161 frente a 0,0150 en A-test, 0,0656 frente a 0,0668 en B y 0,0181 frente a 0,0236 en C-test, con menor Recall). Mostrar el desglose por provincia con una nota («el modelo no mejora al FWI en esta provincia»); lo pide la mitigación de sesgos geográficos de NF2.
4. **Una caída o empate no es un error:** se publica sea cual sea su signo (DEC-47).
5. El panel debe indicar que son resultados de evaluación sobre años pasados, no de las predicciones actuales.
