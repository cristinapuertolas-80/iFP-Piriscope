# `src/ml` · Machine Learning (entrenamiento, evaluación e inferencia)

Estima la **probabilidad de que un municipio registre al menos un parte de incendio el día D+1**, a partir de la información disponible hasta el día D (apartado 5.4 del diseño técnico, HU-06 a HU-08). Lee el dataset de entrenamiento de la capa Gold, compara el modelo con el baseline FWI y escribe las predicciones en Gold. Las explicaciones SHAP están en [`src/explainability`](../explainability/README.md).

> **Estado:** el entrenamiento, la evaluación, la inferencia batch y SHAP están implementados y **ejecutados sobre el dataset real de Gold** (resultados en [`docs/resultados`](../../docs/resultados/README.md)), y se lanzan desde `run_pipeline.py --etapas ml` (HU-11). Pendiente: decidir la política de niveles de riesgo (DEC-49) y comparar el FWI intermedio con el consolidado.

## Resultado

El modelo con FWI (`con_fwi`) **mejora al FWI** en los tres escenarios de test (A, B y C) y en 6 de 7 años de la validación walk-forward, con una ventaja **modesta**: en 2023 su AUC-PR es 0,028, unas 3,7 veces el de un modelo al azar (el FWI, 2,6 veces). En el walk-forward 2016-2022, a igualdad de alarmas captura entre un 14 % y un 19 % más incendios que el FWI. El modelo sin FWI (`sin_fwi`) equivale al FWI. Con un único año de test, las conclusiones son consistentes pero no definitivas. Tablas, limitaciones y cómo reproducirlo: [`docs/resultados/README.md`](../../docs/resultados/README.md).

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
| `walkforward.py` | Validación walk-forward: repite la comparación con el FWI en varios años. Con `--guardar-predicciones` guarda las probabilidades y los umbrales de cada pliegue |
| `inferir.py` | Inferencia batch con el modelo guardado: dominio del modelo, modelo de respaldo, predicciones, explicaciones SHAP, avisos de calidad y figuras de ejemplo |
| `verificar_salidas.py` | Comprueba `predicciones.parquet` y `explicaciones.parquet` (unicidad, cobertura, niveles frente a umbrales, reconstrucción de la probabilidad con SHAP) |

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

### Inferencia en producción

La forma recomendada de predecir es `inferir.py` (o `run_pipeline.py --etapas ml`), no `predict.build_predictions` a mano:

```bash
# Último día disponible
python -m src.ml.inferir --modelo v1_xgboost_con_fwi-sin_viento-desde2009 \
                         --modelo-respaldo v1_xgboost_sin_fwi-sin_viento-desde2009

# Un rango de días D, empezando las salidas de cero y con figuras SHAP de ejemplo
python -m src.ml.inferir --modelo ... --modelo-respaldo ... --desde 2024-01-02 --hasta 2026-10-21 \
                         --reemplazar --ejemplo-dir docs/resultados

# Comprobar las salidas
python src/ml/verificar_salidas.py --pred data/gold/predicciones.parquet --expl data/gold/explicaciones.parquet \
    --dataset data/gold/dataset_municipio_dia.parquet --models-dir models \
    --modelo <principal> --respaldo <respaldo>
```

- **Dominio del modelo (DEC-48):** si una variable que nunca faltó al entrenar (el FWI desde 2009) falta en un día, el modelo devuelve valores arbitrarios. Esas filas **no se predicen con el principal**: se predicen con `--modelo-respaldo` (modelo sin FWI, `version_modelo = <v>-respaldo`) o se omiten. Las filas sin ninguna variable meteorológica con dato tampoco se predicen (`--min-variables`).
- **Una predicción vigente por municipio y fecha:** al escribir, las nuevas sustituyen a las existentes del mismo municipio y `fecha_objetivo` (aunque tengan otra versión) y a sus explicaciones; así, cuando llega el FWI, `v1` reemplaza a `v1-respaldo`. La escritura es atómica. `--reemplazar` descarta todo lo anterior.
- **Avisos** por stderr: cobertura baja (`--min-cobertura`), fechas con casi todos los municipios en «medio» o «alto» (`--aviso-alarmas`), muchos nulos meteorológicos en el último día y retraso respecto a hoy (`--max-retraso`).
- **Figuras de ejemplo** (`--ejemplo-dir`): se generan después de escribir las salidas; si fallan, el código de salida es 2 pero las predicciones ya están escritas.

### Análisis auxiliares

Scripts independientes (solo pandas y numpy) para estudiar las salidas y los niveles de riesgo, en `scripts/`:

| Script | Para qué |
|---|---|
| `analisis_walkforward.py` | Políticas de niveles (Recall objetivo o presupuesto de alarmas), alarmas por mes, días saturados y calibración con las predicciones del walk-forward (DEC-49) |
| `analisis_niveles.py` | Lo mismo sobre la validación de 2022 con el modelo guardado (`python -m src.ml.analisis_niveles`) |
| `diagnostico_alarmas.py` | Diagnostica las fechas con casi todos los municipios en «medio» o «alto» |
| `comparar_fwi.py` | Compara el FWI intermedio con el consolidado para un mismo mes |

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

Los umbrales se fijan en validación (municipios no reservados) por Recall objetivo: **medio** = Recall ≥ 0,8 y **alto** = Recall ≥ 0,5. Son valores **provisionales** (DEC-41, DEC-49):

- En test, el Recall real del 80 % fijado en 2022 fue de 0,73 (A), 0,81 (B) y 0,76 (C). En el walk-forward 2016-2022, con el umbral de cada año fijado con el año anterior, el Recall va de 0,70 a 0,90 y las alarmas con «medio» del 21 al 39 % de los municipio-días; la Precision, del 2 % al 6 %.
- Un presupuesto de alarmas con umbral absoluto tampoco fija el volumen (el del 20 % marcó entre el 13 y el 28 % según el año), por lo que **se propone mantener el Recall objetivo** (DEC-49, pendiente de confirmar).
- En episodios de sequía casi todos los municipios superan el umbral «medio». No es un fallo de datos: el modelo refleja el riesgo general y los niveles absolutos pierden capacidad de priorizar. El percentil del municipio dentro de su día aporta algo de señal dentro de esos días (lift del top 10 % de 1,5 a 2,1 en los siete años), y se propone como dato complementario.
- La probabilidad sobrestima la tasa real entre 1,5 y 2,4 veces según el año, así que se presenta como **índice de riesgo**.

Los umbrales se reajustan con el último año completo en cada reentrenamiento. Se guardan junto al modelo y el código rechaza aplicar los umbrales de otro modelo.

## Salidas

| Fichero | Contenido |
|---|---|
| `data/gold/predicciones.parquet` | `id_prediccion`, `id_municipio`, `fecha_objetivo`, `version_modelo` (`v1` o `v1-respaldo`), `probabilidad`, `nivel_riesgo`, `fwi` (copia del día D; nulo en el respaldo), `fecha_insert` |
| `data/gold/explicaciones.parquet` | `id_prediccion`, `variable`, `valor_shap`, `valor_variable`, `valor_base`, `shap_resto`: las 5 variables de mayor \|SHAP\| por predicción (DEC-50). Ver [`src/explainability`](../explainability/README.md) |
| `data/gold/informe_desarrollo.csv`, `informe_final.csv`, `informe_walkforward.csv` | Informes de evaluación. Los finales se publican en `docs/resultados/` |
| `models/<versión>_<algoritmo>_<variante>.joblib` | Modelo serializado. Junto a él, un `.json` con versión, semilla, variables, parámetros, año inicial y variables excluidas, y un `_umbrales.json` |

Al escribir, las predicciones nuevas sustituyen a las existentes del mismo municipio y `fecha_objetivo` (y a sus explicaciones) y se conserva el resto, así que reprocesar un periodo no duplica datos. En producción se predicen todos los municipios, reservados incluidos. El contrato con el dashboard está en [`docs/contrato_dashboard_gold.md`](../../docs/contrato_dashboard_gold.md).

## Decisiones

Registradas en [docs/decisiones.md](../../docs/decisiones.md), DEC-37 a DEC-50 (contrato de entrada, variantes, entrenamiento, comparación con el baseline, niveles de riesgo, inferencia, SHAP, persistencia de modelos, municipios reservados, dataset de Gold, configuración del modelo, inferencia real con modelo de respaldo, niveles de riesgo y SHAP ampliado).

## Limitaciones y pendientes

- **Un único año de test (2023):** las conclusiones sobre la mejora frente al FWI son consistentes pero no definitivas (en A, el límite inferior del intervalo está casi en cero). El walk-forward 2016-2022 las respalda: gana en 6 de 7 años.
- **Rendimiento absoluto modesto:** Precision del 2 % al 6 % con Recall ≈ 0,8, según el año. «Medio» o «alto» significa «no descartar», no «habrá un incendio». **A Coruña no mejora al FWI.**
- **El modelo de respaldo no mejora al FWI** (`sin_fwi` equivale al FWI): las predicciones de los últimos días, sin FWI, no aportan más que el propio índice, y sus niveles discriminan poco.
- **Desfase de los datos:** el FWI y la meteorología llegan con retraso de varios días, así que la última fecha objetivo va por detrás de hoy. No es una previsión a varios días.
- **FWI intermedio:** los meses aún no consolidados usan el dataset intermedio, y el modelo se entrenó con el consolidado. No se ha comparado el efecto sobre las predicciones recientes (`scripts/comparar_fwi.py`).
- **Probabilidad sin calibrar:** sobrestima entre 1,5 y 2,4 veces según el año y una calibración fijada en un año no se mantiene; se presenta como índice de riesgo.
- **Niveles de riesgo:** política pendiente de confirmar (DEC-49). Los umbrales son provisionales y el Recall objetivo no se reproduce con exactitud de un año a otro.
- **Etiquetas de 2022 y 2023:** pendiente contrastar con las estadísticas oficiales de EGIF que sean definitivas (DEC-31).
- Los `.joblib` son pickles: solo cargar ficheros generados por el propio proyecto, y no se suben al repositorio. Pendiente subir los modelos finales a Hugging Face (DEC-44).

## Tests

```bash
pytest tests/test_config.py tests/test_splits.py tests/test_evaluate.py tests/test_train.py tests/test_risk.py \
       tests/test_predict.py tests/test_reservados.py tests/test_experimento.py tests/test_walkforward.py tests/test_inferir.py
```

Usan datos aleatorios con el esquema de Gold (`tests/toy_data.py`, fixture `toy_gold` en `tests/conftest.py`) y no necesitan red ni datos reales. Comprueban, entre otras cosas, que ninguna variante usa variables prohibidas, que el entrenamiento no toca filas reservadas ni de test, que modelo y baseline se evalúan sobre las mismas filas, que el bootstrap por bloques da intervalos más anchos con datos agrupados, que el walk-forward solo usa el pasado y no llega al año de test, y que el escenario B se recorta al periodo de entrenamiento. `test_inferir.py` cubre la inferencia: último día por defecto, rangos, dominio del modelo y respaldo, filas sin información y figuras de ejemplo.