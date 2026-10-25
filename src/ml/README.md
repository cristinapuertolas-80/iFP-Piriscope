# `src/ml` · Machine Learning (entrenamiento, evaluación e inferencia)

Estima la **probabilidad de que un municipio registre al menos un parte de incendio el día D+1**, a partir de la información disponible hasta el día D (apartado 5.4 del diseño técnico, HU-06 a HU-08). Lee el dataset de entrenamiento de la capa Gold, compara el modelo con el baseline FWI y escribe las predicciones en Gold. Las explicaciones SHAP están en [`src/explainability`](../explainability/README.md).

> **Estado:** implementado y probado con un dataset de juguete (`tests/toy_data.py`, solo para tests). Todavía **no** lee la capa Gold real ni está conectado a `run_pipeline.py`. Con datos de juguete el modelo no aprende nada: los tests comprueban la mecánica, no la calidad.

## Módulos

| Módulo | Qué hace |
|---|---|
| `config.py` | Contrato de datos (Anexo A): grupos de variables, variantes del modelo, columnas prohibidas, constantes de rutas y `validate_schema` |
| `splits.py` | Filas de entrenamiento y de los escenarios A, B y C; comprobación de integridad de las particiones |
| `train.py` | Entrenamiento de XGBoost y Random Forest con pesos de clase; guardado y carga de modelos |
| `evaluate.py` | AUC-PR, Recall y Precision; umbral por Recall objetivo; bootstrap; tabla modelo vs baseline FWI |
| `risk.py` | Umbrales de los niveles bajo, medio y alto, fijados sobre validación |
| `predict.py` | Inferencia batch D+1 con el esquema de `Prediccion`; escritura de `predicciones.parquet` |
| `reservados.py` | Selección reproducible de municipios reservados por bloques de estación, estratificada por provincia y frecuencia de incendios (`python -m src.ml.reservados`) |

## Contrato de entrada (tabla de entrenamiento de Gold)

`config.validate_schema(df)` exige estas columnas. Los nombres y valores salen del Anexo A; si Gold los cambia, se modifican las constantes de `config.py`.

| Grupo | Columnas |
|---|---|
| Identificación | `id_municipio`, `provincia`, `fecha` |
| Partición | `reservado` (0/1) y `particion` (`entrenamiento`, `validacion` o `test`, **sin tilde**) |
| Objetivo | `incendio_d1` (0/1) |
| Meteorología | `temperatura`, `humedad_relativa`, `viento`, `precipitacion` |
| Derivadas | `racha_dias_secos`, `precip_acum_7d`, `mes` (entero 1-12) |
| FWI | `fwi`, `ffmc`, `dmc`, `dc`, `isi`, `bui` |

Debe haber una fila por municipio y día. El nombre del fichero de la tabla de entrenamiento lo fija la capa Gold (pendiente).

## Uso

Los módulos se usan desde Python (aún no hay línea de comandos):

```python
import pandas as pd
from src.ml import config, splits, train, evaluate, risk, predict
from src.explainability import shap_explain

df = pd.read_parquet("data/gold/<tabla_entrenamiento>.parquet")   # nombre por definir en Gold
config.validate_schema(df)

# 1. Entrenar (solo reservado = 0 y particion = entrenamiento) y fijar niveles de riesgo
bundle = train.train_model(df, algorithm="xgboost", variant="con_fwi", version="v1")
val = train.score_dataframe(bundle, splits.select_scenario(df, "A", split="validacion"))
thr = risk.fit_risk_thresholds(val[config.TARGET], val["proba"], bundle.model_id)
train.save_model(bundle)       # models/v1_xgboost_con_fwi.joblib (+ .json)
risk.save_thresholds(thr)      # models/v1_xgboost_con_fwi_umbrales.json

# 2. Comparar con el baseline FWI (HU-07)
umbrales = evaluate.thresholds_from_validation(val)
test = train.score_dataframe(bundle, splits.select_scenario(df, "A", split="test"))
informe = evaluate.compare_model_vs_baseline(test, umbrales)   # global y por provincia, con IC

# 3. Predicción batch D+1 y explicaciones SHAP
pred = predict.build_predictions(df, bundle, thr)               # último día disponible
predict.write_predictions(pred)                                 # data/gold/predicciones.parquet
shap_explain.write_explanations(shap_explain.build_explanations(pred, df, bundle))
```

Para entrenar las tres variantes de una vez: `train.train_all_variants(df, "xgboost")`.

## Variantes del modelo

| Variante | Variables | Para qué |
|---|---|---|
| `con_fwi` | Meteorología, derivadas, FWI y sus componentes | Modelo completo |
| `sin_fwi` | Meteorología y derivadas | La comparación con el baseline no es circular (HU-07) |
| `solo_meteo` | Las cuatro variables meteorológicas | Mide cuánto aportan las derivadas |

Nunca entran al modelo: identificadores, coordenadas, `provincia`, `dist_estacion_km`, datos conocidos a posteriori (`causa`, `fecha_extincion`, `superficie_forestal_ha`) y columnas de trazabilidad (`config.FORBIDDEN_FEATURES`; un test lo comprueba).

## Escenarios de evaluación

| Escenario | Filtro (`splits.select_scenario`) | Pregunta |
|---|---|---|
| A. Temporal | `reservado = 0` y `particion` en validación o test | ¿Funciona como sistema operativo? |
| B. Geográfico | `reservado = 1` y `particion = entrenamiento` | ¿Generaliza a municipios no vistos? |
| C. Temporal y geográfico | `reservado = 1` y `particion` en validación o test | Caso más exigente |

El modelo y el baseline se comparan **sobre exactamente las mismas filas**. El Recall se mide con un umbral fijado en validación, nunca en el conjunto evaluado, y ambos sistemas usan el mismo Recall objetivo, así que en test la comparación se lee en Precision.

## Niveles de riesgo

Los umbrales se fijan en validación (municipios no reservados) por Recall objetivo: **medio** = Recall ≥ 0,8 y **alto** = Recall ≥ 0,5. Son valores **provisionales** hasta el EDA. Se guardan junto al modelo y el código rechaza aplicar los umbrales de otro modelo.

## Salidas

| Fichero | Contenido |
|---|---|
| `data/gold/predicciones.parquet` | `id_prediccion`, `id_municipio`, `fecha_objetivo`, `version_modelo`, `probabilidad`, `nivel_riesgo`, `fwi` (copia del día D), `fecha_insert` |
| `models/<versión>_<algoritmo>_<variante>.joblib` | Modelo serializado. Junto a él, un `.json` con versión, semilla, variables, parámetros, filas y fecha, y un `_umbrales.json` |

Las escrituras sobrescriben las filas con el mismo `id_prediccion` y conservan el resto, así que reprocesar un periodo no duplica datos. En producción se predicen todos los municipios, reservados incluidos.

## Decisiones

Registradas en [docs/decisiones.md](../../docs/decisiones.md), DEC-37 a DEC-45 (contrato de entrada, variantes, entrenamiento, comparación con el baseline, niveles de riesgo, inferencia, SHAP y persistencia de modelos).

## Limitaciones y pendientes

- Sin datos reales de Gold: no hay todavía métricas del modelo ni hiperparámetros ajustados (AUC-PR sobre validación).
- Los umbrales de riesgo (0,8 y 0,5) son provisionales hasta el EDA.
- El bootstrap remuestrea filas y ignora la correlación entre días y municipios vecinos: los intervalos son algo optimistas, sobre todo en B y C.
- Silver informa de un 56 % de nulos en viento. XGBoost y Random Forest admiten nulos, pero la política de imputación se decide tras el EDA.
- Los `.joblib` son pickles: solo cargar ficheros generados por el propio proyecto. Pendiente decidir si los modelos se alojan en Hugging Face.
- Integración en `run_pipeline.py` pendiente.

## Tests

```bash
pytest tests/test_config.py tests/test_splits.py tests/test_evaluate.py tests/test_train.py tests/test_risk.py tests/test_predict.py tests/test_shap.py
```

Usan datos aleatorios con el esquema de Gold (`tests/toy_data.py`, fixture `toy_gold` en `tests/conftest.py`) y no necesitan red ni datos reales. Comprueban, entre otras cosas, que ninguna variante usa variables prohibidas, que el entrenamiento no toca filas reservadas ni de test, que modelo y baseline se evalúan sobre las mismas filas y que SHAP es aditivo.
