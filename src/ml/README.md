# `src/ml` · Machine Learning (entrenamiento, evaluación e inferencia)

Estima la **probabilidad de que un municipio registre al menos un parte de incendio el día D+1**, a partir de la información disponible hasta el día D (apartado 5.4 del diseño técnico, HU-06 a HU-08). Lee el dataset de entrenamiento de la capa Gold, compara el modelo con el baseline FWI y escribe las predicciones en Gold. Las explicaciones SHAP están en [`src/explainability`](../explainability/README.md).

> **Estado:** el entrenamiento y la evaluación están implementados y **ejecutados sobre el dataset real de Gold** (resultados en [`docs/resultados`](../../docs/resultados/README.md)). La **inferencia y SHAP con el modelo real** todavía no se han ejecutado, y falta la integración en `run_pipeline.py` (HU-11).

## Resultado

El modelo con FWI (`con_fwi`) **mejora al FWI** en los tres escenarios de test (A, B y C) y en 6 de 7 años de la validación walk-forward, con una ventaja **modesta**: en 2023 su AUC-PR es 0,028, unas 3,7 veces el de un modelo al azar (el FWI, 2,6 veces). El modelo sin FWI (`sin_fwi`) equivale al FWI. Con un único año de test, las conclusiones son consistentes pero no definitivas. Tablas, limitaciones y cómo reproducirlo: [`docs/resultados/README.md`](../../docs/resultados/README.md).

## Módulos

| Módulo | Qué hace |
|---|---|
| `config.py` | Contrato de datos (Anexo A): grupos de variables, variantes del modelo, columnas prohibidas, constantes de rutas y `validate_schema` |
| `splits.py` | Filas de entrenamiento (con recorte opcional de años) y de los escenarios A, B y C; comprobación de integridad de las particiones |
| `train.py` | Entrenamiento de XGBoost y Random Forest, con recorte de años (`anio_desde`) y exclusión de variables (`excluir`); guardado y carga de modelos |
| `evaluate.py` | AUC-PR, Recall y Precision; umbral por Recall objetivo; bootstrap por filas o por bloques; diferencia de AUC-PR pareada; tabla modelo vs baseline FWI |
| `risk.py` | Umbrales de los niveles bajo, medio y alto, fijados sobre validación |
| `predict.py` | Inferencia batch D+1 con el esquema de `Prediccion`; escritura de `predicciones.parquet` |
| `reservados.py` | Selección reproducible de municipios reservados por bloques de estación, estratificada por provincia y frecuencia de incendios |
| `experimento.py` | Entrena las variantes, compara con el FWI y escribe el informe (fases de desarrollo y final) |
| `walkforward.py` | Validación walk-forward: repite la comparación con el FWI en varios años |
| `inferir.py` | Inferencia batch con el modelo guardado, predicciones y SHAP |

## Contrato de entrada (tabla de entrenamiento de Gold)

El dataset lo genera `python -m src.features` ([`src/features`](../features/README.md)) en `data/gold/dataset_entrenamiento.parquet`. `config.validate_schema(df)` exige estas columnas; los nombres y valores salen del Anexo A, y si Gold los cambia se modifican las constantes de `config.py`.

| Grupo | Columnas |
|---|---|
| Identificación | `id_municipio`, `provincia`, `fecha` |
| Partición | `reservado` (0/1) y `particion` (`entrenamiento`, `validacion` o `test`, **sin tilde**) |
| Objetivo | `incendio_d1` (0/1) |
| Meteorología | `temperatura`, `humedad_relativa`, `viento`, `precipitacion` |
| Derivadas | `racha_dias_secos`, `precip_acum_7d`, `mes` (entero 1-12) |
| FWI | `fwi`, `ffmc`, `dmc`, `dc`, `isi`, `bui` |

Debe haber una fila por municipio y día. Las columnas adicionales (como `dist_estacion_km`) se ignoran.

## Configuración elegida (DEC-47)

| Decisión | Valor |
|---|---|
| Algoritmo | XGBoost, hiperparámetros por defecto, sin ponderar la clase positiva (`scale_pos_weight=1`) |
| Periodo de entrenamiento | Desde 2009 (`anio_desde`) |
| Variables excluidas | `viento` (`excluir`) |
| Variantes finales | `con_fwi` y `sin_fwi` |
| Intervalos de confianza | Bootstrap por bloques de 7 días |

El nombre de cada modelo recoge estas opciones, p. ej. `con_fwi-sin_viento-desde2009`.

## Uso

### Línea de comandos

Desde la raíz del repositorio, con el dataset de Gold generado:

```bash
# Desarrollo: solo valida en 2022 con municipios no reservados (nunca toca test ni reservados)
python -m src.ml.experimento --variantes con_fwi sin_fwi --anio-desde 2009 --excluir viento

# Validación walk-forward: ¿gana el modelo al FWI en varios años? (2016-2022)
python -m src.ml.walkforward --anio-desde 2009 --excluir viento --desde-fold 2016

# Informe final: A (test), B y C. Se ejecuta UNA vez, con la configuración ya decidida
python -m src.ml.experimento --final --variantes con_fwi sin_fwi --anio-desde 2009 --excluir viento --guardar --version v1

# Selección de municipios reservados (la lista ya está congelada en docs/, DEC-45)
python -m src.ml.reservados
```

Opciones útiles de `experimento.py` y `walkforward.py`: `--param CLAVE=VALOR` (hiperparámetros, p. ej. `max_depth=3`), `--bloque-dias` (bloques del bootstrap, `0` = filas sueltas), `--bootstrap` (remuestras), `--salida` (CSV del informe) y `--models-dir`. Terminan con código 2 y un mensaje claro si falta el dataset o no cumple el contrato. `reservados.py` se niega a sobrescribir una lista existente salvo con `--forzar`.

### Desde Python

```python
import pandas as pd
from src.ml import config, splits, train, evaluate, risk, predict
from src.explainability import shap_explain

df = pd.read_parquet(config.DATASET_ENTRENAMIENTO)
config.validate_schema(df)

# 1. Entrenar (solo reservado = 0 y particion = entrenamiento) y fijar niveles de riesgo
bundle = train.train_model(df, "xgboost", "con_fwi", version="v1", anio_desde=2009, excluir=["viento"])
val = train.score_dataframe(bundle, splits.select_scenario(df, "A", split="validacion"))
thr = risk.fit_risk_thresholds(val[config.TARGET], val["proba"], bundle.model_id)
train.save_model(bundle)       # models/v1_xgboost_con_fwi-sin_viento-desde2009.joblib (+ .json)
risk.save_thresholds(thr)      # models/..._umbrales.json

# 2. Comparar con el baseline FWI (HU-07)
umbrales = evaluate.thresholds_from_validation(val)
test = train.score_dataframe(bundle, splits.select_scenario(df, "A", split="test"))
informe = evaluate.compare_model_vs_baseline(test, umbrales)   # global y por provincia, con IC

# 3. Predicción batch D+1 y explicaciones SHAP (necesita también los días recientes sin etiqueta)
todo = pd.read_parquet(config.GOLD_DIR / "dataset_municipio_dia.parquet")
pred = predict.build_predictions(todo, bundle, thr)             # último día disponible
predict.write_predictions(pred)                                 # data/gold/predicciones.parquet
shap_explain.write_explanations(shap_explain.build_explanations(pred, todo, bundle))
```

Para entrenar las tres variantes de una vez: `train.train_all_variants(df, "xgboost", anio_desde=2009, excluir=["viento"])`.

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

El modelo y el baseline se comparan **sobre exactamente las mismas filas**. El Recall se mide con un umbral fijado en validación, nunca en el conjunto evaluado, y ambos sistemas usan el mismo Recall objetivo, así que en test la comparación se lee en Precision y, sobre todo, en AUC-PR.

En el informe final, B se recorta a los **años de entrenamiento del modelo** (con `--anio-desde 2009`, usa 2009-2021) y se conserva `B-todos-los-anios` (sin recortar) para dejar a la vista ambas versiones.

## Evaluación y protocolo

- **Dos fases** para proteger el test: *desarrollo* (solo la validación de municipios no reservados) y *final* (A en test, B y C), que se ejecuta una vez con la configuración fijada.
- **Intervalos por bloques:** el bootstrap remuestrea bloques de 7 días seguidos, porque los incendios se agrupan en olas y remuestrear filas sueltas da intervalos demasiado estrechos. La diferencia de AUC-PR frente al FWI se calcula con bootstrap *pareado* (columnas `dif_auc_pr`, `dif_lo`, `dif_hi` y `p_mejora` del informe).
- **Walk-forward:** para cada año Y entrena solo con años anteriores, usa Y−1 para la parada temprana y los umbrales, y evalúa en Y. Solo usa municipios no reservados y se niega a llegar al año de test.

## Niveles de riesgo

Los umbrales se fijan en validación (municipios no reservados) por Recall objetivo: **medio** = Recall ≥ 0,8 y **alto** = Recall ≥ 0,5. Son valores **provisionales**: en test, el Recall real del 80 % fijado en 2022 fue de 0,73 (A), 0,81 (B) y 0,76 (C), porque la prevalencia baja de un año a otro. Se guardan junto al modelo y el código rechaza aplicar los umbrales de otro modelo.

## Salidas

| Fichero | Contenido |
|---|---|
| `data/gold/predicciones.parquet` | `id_prediccion`, `id_municipio`, `fecha_objetivo`, `version_modelo`, `probabilidad`, `nivel_riesgo`, `fwi` (copia del día D), `fecha_insert` |
| `data/gold/informe_desarrollo.csv`, `informe_final.csv`, `informe_walkforward.csv` | Informes de evaluación. Los finales se publican en `docs/resultados/` |
| `models/<versión>_<algoritmo>_<variante>.joblib` | Modelo serializado. Junto a él, un `.json` con versión, semilla, variables, parámetros, año inicial y variables excluidas, y un `_umbrales.json` |

Las escrituras de predicciones sobrescriben las filas con el mismo `id_prediccion` y conservan el resto, así que reprocesar un periodo no duplica datos. En producción se predicen todos los municipios, reservados incluidos.

## Decisiones

Registradas en [docs/decisiones.md](../../docs/decisiones.md), DEC-37 a DEC-47 (contrato de entrada, variantes, entrenamiento, comparación con el baseline, niveles de riesgo, inferencia, SHAP, persistencia de modelos, municipios reservados, dataset de Gold y configuración del modelo).

## Limitaciones y pendientes

- **Un único año de test (2023):** las conclusiones sobre la mejora frente al FWI son consistentes pero no definitivas. En A, el límite inferior del intervalo está casi en cero.
- **Rendimiento absoluto modesto** y **A Coruña no mejora al FWI**.
- **Probabilidades sin calibrar para 2023:** el modelo se entrena con una prevalencia del ~2 % y la de 2023 fue del 0,75 %, así que tiende a sobrestimarlas. Antes de mostrarlas como probabilidad hay que calibrarlas o presentarlas como índice de riesgo.
- **Etiquetas de 2022 y 2023:** pendiente contrastar con las estadísticas oficiales de EGIF que sean definitivas (DEC-31).
- Los umbrales de riesgo (0,8 y 0,5) son provisionales y el Recall objetivo no se reproduce con exactitud de un año a otro.
- Los `.joblib` son pickles: solo cargar ficheros generados por el propio proyecto, y no se suben al repositorio. Pendiente decidir dónde alojar los modelos finales (DEC-44).
- Pendiente: integración en `run_pipeline.py` (HU-11, que pasará `anio_desde=2009` y `excluir=["viento"]`).

## Tests

```bash
pytest tests/test_config.py tests/test_splits.py tests/test_evaluate.py tests/test_train.py tests/test_risk.py \
       tests/test_predict.py tests/test_reservados.py tests/test_experimento.py tests/test_walkforward.py
```

Usan datos aleatorios con el esquema de Gold (`tests/toy_data.py`, fixture `toy_gold` en `tests/conftest.py`) y no necesitan red ni datos reales. Comprueban, entre otras cosas, que ninguna variante usa variables prohibidas, que el entrenamiento no toca filas reservadas ni de test, que modelo y baseline se evalúan sobre las mismas filas, que el bootstrap por bloques da intervalos más anchos con datos agrupados, que el walk-forward solo usa el pasado y no llega al año de test, y que el escenario B se recorta al periodo de entrenamiento.