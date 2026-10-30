# Decisiones

Registro de las decisiones técnicas y de diseño de **Piriscope**. Cada entrada recoge el contexto, la decisión adoptada, las alternativas consideradas y sus consecuencias, para que cualquier miembro del equipo (o quien evalúe el proyecto) pueda entender por qué el sistema es como es.

Las decisiones iniciales proceden del documento de diseño técnico de la fase NF2 (*Documentación y diseño técnico*, Grupo 6, entrega del 14/10/2026). Las nuevas decisiones se añaden al final siguiendo la [plantilla](#plantilla-para-nuevas-decisiones) y se incorporan mediante Pull Request, tal como indica [CONTRIBUTING.md](../CONTRIBUTING.md).

**Estados posibles:** `Aceptada` · `Pendiente` (se cerrará tras el EDA u otra fase) · `Sustituida por DEC-XX` · `Descartada`

## Índice

| ID | Decisión | Ámbito | Estado |
|----|----------|--------|--------|
| [DEC-01](#dec-01-galicia-como-región-piloto) | Galicia como región piloto | Alcance | Aceptada |
| [DEC-02](#dec-02-definición-de-riesgo-y-horizonte-d1) | Definición de riesgo y horizonte D+1 | Alcance | Aceptada |
| [DEC-03](#dec-03-unidad-de-análisis-municipio--día-y-construcción-de-negativos) | Unidad de análisis municipio + día | Datos | Aceptada |
| [DEC-04](#dec-04-tres-niveles-de-riesgo-con-umbrales-fijados-en-validación) | Tres niveles de riesgo | Modelo | Pendiente (umbrales) |
| [DEC-05](#dec-05-egif-miteco-como-única-fuente-de-eventos-de-incendio) | EGIF-MITECO como fuente de eventos | Datos | Aceptada |
| [DEC-06](#dec-06-fwi-de-effiscopernicus-como-variable-y-como-baseline) | FWI como variable y como baseline | Datos / Evaluación | Aceptada |
| [DEC-07](#dec-07-asignación-de-incendios-a-municipios-por-código-ine) | Asignación de incendios a municipios por código INE | Datos | Aceptada |
| [DEC-08](#dec-08-meteorología-de-la-estación-aemet-más-cercana) | Meteorología de la estación AEMET más cercana | Datos | Aceptada |
| [DEC-09](#dec-09-fwi-muestreado-en-el-centroide-municipal) | FWI muestreado en el centroide municipal | Datos | Aceptada |
| [DEC-10](#dec-10-arquitectura-medallón-bronze--silver--gold-sobre-parquet) | Arquitectura medallón sobre Parquet | Arquitectura | Aceptada |
| [DEC-11](#dec-11-hugging-face-datasets-como-almacén-de-datos-e-ingesta-automática) | Hugging Face Datasets como almacén de datos | Arquitectura | Aceptada |
| [DEC-12](#dec-12-aplicación-monolítica-en-streamlit-con-predicciones-precalculadas) | Monolito Streamlit con predicciones precalculadas | Arquitectura | Aceptada |
| [DEC-13](#dec-13-módulos-desacoplados-por-ficheros-y-orquestador-único) | Módulos desacoplados y orquestador único | Arquitectura | Aceptada |
| [DEC-14](#dec-14-identificadores-y-claves-lógicas) | Identificadores y claves lógicas | Modelo de datos | Aceptada |
| [DEC-15](#dec-15-trazabilidad-mediante-fecha_insert-y-version_modelo) | Trazabilidad con `fecha_insert` y `version_modelo` | Modelo de datos | Aceptada |
| [DEC-16](#dec-16-python--pandas-pyspark-solo-si-se-justifica) | Python + pandas | Tecnología | Aceptada |
| [DEC-17](#dec-17-xgboost-yo-random-forest-como-modelos) | XGBoost y/o Random Forest | Tecnología / Modelo | Aceptada |
| [DEC-18](#dec-18-shap-para-la-explicabilidad) | SHAP para la explicabilidad | Tecnología | Aceptada |
| [DEC-19](#dec-19-geopandas--plotly-para-lo-geoespacial-y-los-mapas) | GeoPandas + Plotly | Tecnología | Aceptada |
| [DEC-20](#dec-20-pytest--github-actions) | pytest + GitHub Actions | Calidad | Aceptada |
| [DEC-21](#dec-21-partición-temporal-congelada) | Partición temporal congelada | Validación | Aceptada (años pendientes) |
| [DEC-22](#dec-22-partición-geográfica-por-bloques-de-municipios) | Partición geográfica por bloques | Validación | Aceptada |
| [DEC-23](#dec-23-tres-escenarios-de-evaluación-a-b-c) | Escenarios de evaluación A, B y C | Validación | Aceptada |
| [DEC-24](#dec-24-métricas-y-tratamiento-del-desbalanceo) | Métricas y desbalanceo | Validación | Aceptada |
| [DEC-25](#dec-25-variables-excluidas-del-modelo) | Variables excluidas del modelo | Modelo | Aceptada |
| [DEC-26](#dec-26-flujo-git-basado-en-pull-requests) | Flujo Git basado en Pull Requests | Proceso | Aceptada |
| [DEC-27](#dec-27-datos-personales-y-licencias) | Datos personales y licencias | Ética / Legal | Aceptada |
| [DEC-28](#dec-28-criterios-de-accesibilidad-del-dashboard) | Accesibilidad del dashboard | Ética / UX | Aceptada |
| [DEC-29](#dec-29-control-de-alcance-y-punto-de-control-s5s6) | Control de alcance y punto de control | Proceso | Aceptada |
| [DEC-30](#dec-30-diseño-de-la-ingesta-en-bronze) | Diseño de la ingesta en Bronze | Datos / Arquitectura | Aceptada |
| [DEC-31](#dec-31-periodo-y-frecuencias-de-ingesta) | Periodo y frecuencias de ingesta | Datos / Proceso | Aceptada (periodo pendiente del EDA) |
| [DEC-32](#dec-32-diseño-de-la-capa-silver) | Diseño de la capa Silver | Datos / Arquitectura | Aceptada |
| [DEC-33](#dec-33-meteorología-en-silver-estación-fija-y-huecos-nulos) | Meteorología en Silver: estación fija y huecos nulos | Datos | Aceptada |
| [DEC-34](#dec-34-fwi-en-silver-celda-de-tierra-más-cercana-y-prioridad-del-consolidado) | FWI en Silver: celda de tierra más cercana | Datos | Aceptada |
| [DEC-35](#dec-35-asignación-del-código-ine-a-los-partes-de-egif) | Asignación del código INE a los partes de EGIF | Datos | Aceptada |
| [DEC-36](#dec-36-controles-de-calidad-y-validación-de-silver) | Controles de calidad y validación de Silver | Calidad | Aceptada |
| [DEC-37](#dec-37-contrato-de-entrada-del-módulo-ml) | Contrato de entrada del módulo ML | Modelo de datos / Modelo | Aceptada |
| [DEC-38](#dec-38-variantes-del-modelo-y-variables-permitidas) | Variantes del modelo y variables permitidas | Modelo | Aceptada |
| [DEC-39](#dec-39-protocolo-de-entrenamiento) | Protocolo de entrenamiento | Modelo | Sustituida en parte por DEC-47 |
| [DEC-40](#dec-40-comparación-con-el-baseline-fwi) | Comparación con el baseline FWI | Evaluación | Sustituida en parte por DEC-47 |
| [DEC-41](#dec-41-niveles-de-riesgo-por-recall-objetivo) | Niveles de riesgo por Recall objetivo | Modelo | Pendiente (valores provisionales) |
| [DEC-42](#dec-42-inferencia-batch-d1-y-salidas-de-gold) | Inferencia batch D+1 y salidas de Gold | Arquitectura / Modelo | Aceptada |
| [DEC-43](#dec-43-explicabilidad-shap-y-gráficos-desde-gold) | Explicabilidad SHAP y gráficos desde Gold | Modelo / UX | Aceptada |
| [DEC-44](#dec-44-persistencia-y-versionado-de-modelos) | Persistencia y versionado de modelos | Arquitectura | Aceptada (alojamiento pendiente) |
| [DEC-45](#dec-45-selección-de-municipios-reservados) | Selección de municipios reservados | Validación / Modelo | Aceptada |
| [DEC-46](#dec-46-dataset-de-gold-variables-derivadas-y-etiqueta) | Dataset de Gold: variables derivadas y etiqueta | Pipeline de datos / Modelo | Aceptada |
| [DEC-47](#dec-47-configuración-del-modelo-y-protocolo-de-evaluación) | Configuración del modelo y protocolo de evaluación | Modelo / Evaluación | Aceptada |
| [DEC-48](#dec-48-inferencia-real-dominio-del-modelo-modelo-de-respaldo-y-filas-sin-predicción) | Inferencia real: dominio del modelo, modelo de respaldo y filas sin predicción | Modelo / Arquitectura | Aceptada |
| [DEC-49](#dec-49-niveles-de-riesgo-saturación-en-episodios-de-sequía-y-presentación) | Niveles de riesgo: saturación en sequía y presentación | Modelo / UX | Pendiente |

---

## Alcance y definición del problema

### DEC-01. Galicia como región piloto

- **Fecha:** NF2 · **Estado:** Aceptada
- **Contexto:** construir un histórico homogéneo de incendios, meteorología y FWI para toda España es inviable en el plazo del proyecto. Galicia concentra una parte relevante de los incendios forestales del país y tiene un número de municipios manejable.
- **Decisión:** la primera versión cubre únicamente los **313 municipios de Galicia** (A Coruña, Lugo, Ourense y Pontevedra).
- **Alternativas descartadas:** cobertura nacional (excede el alcance); una sola provincia (pocos eventos y sin posibilidad de evaluar diferencias entre provincias).
- **Consecuencias:** el criterio de éxito mínimo es generar predicción diaria para al menos el **80 %** de los municipios gallegos. La ampliación a otras regiones queda fuera de alcance.

### DEC-02. Definición de riesgo y horizonte D+1

- **Fecha:** NF2 · **Estado:** Aceptada
- **Contexto:** "riesgo de incendio" es ambiguo; hace falta una variable objetivo medible con las fuentes disponibles.
- **Decisión:** el riesgo es la **probabilidad de que un municipio registre al menos un parte de incendio en EGIF el día D+1**, usando solo la información disponible hasta el día D (meteorología observada, FWI y variables derivadas). Variable objetivo: `incendio_d1`.
- **Alternativas descartadas:** usar previsión meteorológica a varios días vista (añade incertidumbre y otra fuente); predecir superficie quemada o propagación (fuera de alcance).
- **Consecuencias:** el sistema no detecta incendios activos ni predice su propagación. Todas las features deben respetar el corte temporal en D (ver DEC-25 y el test de fuga en DEC-20).

### DEC-04. Tres niveles de riesgo con umbrales fijados en validación

- **Fecha:** NF2 · **Estado:** Pendiente (umbrales)
- **Contexto:** una probabilidad aislada es difícil de interpretar para el público objetivo (técnicos de protección civil, gestores forestales).
- **Decisión:** la probabilidad se categoriza en **bajo, medio y alto**. Los umbrales se fijarán sobre el **conjunto de validación** después del EDA, nunca sobre el de test.
- **Consecuencias:** el dashboard muestra siempre el nivel junto con la probabilidad numérica. **Pendiente:** registrar aquí los umbrales elegidos y el criterio usado.

---

## Datos

### DEC-03. Unidad de análisis municipio + día y construcción de negativos

- **Fecha:** NF2 · **Estado:** Aceptada
- **Contexto:** EGIF solo registra incendios ocurridos; no existen observaciones negativas explícitas.
- **Decisión:** la unidad de análisis es **municipio + día**. Un día sin parte registrado se etiqueta como `incendio_d1 = 0`. **No se generan datos sintéticos.**
- **Consecuencias:** el volumen estimado es de ~1,7 M filas (15 años) a ~3 M (25 años), con un **desbalanceo de clases marcado** (ver DEC-24). Se asume el riesgo de subregistro de incendios pequeños en zonas remotas, que se documentará como limitación; en el EDA se revisará que la ausencia de datos no se interprete como ausencia de incendio en periodos con cobertura dudosa.

### DEC-05. EGIF-MITECO como única fuente de eventos de incendio

- **Fecha:** NF2 · **Estado:** Aceptada
- **Contexto:** se valoraron EGIF-MITECO y EFFIS como fuentes del histórico de incendios.
- **Decisión:** **EGIF-MITECO** es la fuente de eventos individuales (fecha de detección, municipio, coordenadas UTM, causa, superficie). Se trata como **carga histórica estática** por campañas anuales.
- **Alternativas descartadas:** EFFIS como fuente de eventos, porque su acceso público no ofrece el histórico individual necesario. EFFIS solo se usa para contrastar información.
- **Consecuencias:** hay que unificar ficheros fragmentados por provincia y periodo con un script reproducible que compruebe duplicados y recuentos frente a las estadísticas oficiales. Las campañas más recientes pueden ser provisionales.

### DEC-06. FWI de EFFIS/Copernicus como variable y como baseline

- **Fecha:** NF2 · **Estado:** Aceptada
- **Contexto:** el proyecto quiere demostrar si un modelo supervisado mejora el índice de referencia actual.
- **Decisión:** el **FWI** y sus componentes (FFMC, DMC, DC, ISI, BUI) se obtienen de EFFIS/Copernicus y se usan **como variables del modelo y como baseline** de comparación. El modelo se evaluará **también sin FWI ni sus componentes**, para que la comparación con el baseline no sea circular.
- **Consecuencias:** al menos dos variantes del modelo (con y sin FWI). Si el histórico de FWI no cubre el periodo común, se ajustará el periodo de entrenamiento o se calculará un baseline aproximado, documentándolo.

### DEC-07. Asignación de incendios a municipios por código INE

- **Fecha:** NF2 · **Estado:** Aceptada
- **Contexto:** los partes de EGIF identifican el municipio por nombre, con inconsistencias y cambios históricos (fusiones, cambios de denominación).
- **Decisión:** cada parte se asigna a un municipio mediante su **código INE**, a partir del nombre y **validando con las coordenadas UTM** del parte. Se mantiene una **tabla de equivalencias** (`docs/tabla_equivalencias.csv`) basada en las modificaciones anuales del INE, tomando como referencia el catálogo vigente.
- **Consecuencias:** se genera un informe de partes no asignados y se valida manualmente una muestra. Un municipio que no pueda reconciliarse de forma fiable **se excluye y se documenta**.

### DEC-08. Meteorología de la estación AEMET más cercana

- **Fecha:** NF2 · **Estado:** Aceptada
- **Contexto:** AEMET publica datos por estación, no por municipio, y la densidad de estaciones es desigual.
- **Decisión:** cada municipio se asocia a la **estación AEMET más cercana a su centroide**, registrando `dist_estacion_km` como **indicador de calidad**.
- **Alternativas descartadas:** interpolación espacial entre estaciones (más compleja; se podría reconsiderar si la distancia resulta problemática en el EDA).
- **Consecuencias:** municipios alejados de su estación tendrán datos menos representativos; se analizará en el EDA. El histórico de AEMET se descarga una sola vez, por lotes, con reintentos y caché en Bronze. La API key se configura en el codespace y nunca se versiona.

### DEC-09. FWI muestreado en el centroide municipal

- **Fecha:** NF2 · **Estado:** Aceptada
- **Contexto:** el FWI se distribuye como datos en rejilla, no por municipio.
- **Decisión:** el FWI y sus componentes se **muestrean en el centroide** de cada municipio y se alinean por fecha.
- **Alternativas descartadas:** media zonal sobre el polígono municipal (más precisa, pero más costosa; posible mejora futura).
- **Consecuencias:** los centroides (`lat_centroide`, `lon_centroide`) se obtienen de INE/CNIG y sirven también para asignar estaciones y dibujar el mapa.

---

## Arquitectura

### DEC-10. Arquitectura medallón Bronze / Silver / Gold sobre Parquet

- **Fecha:** NF2 · **Estado:** Aceptada
- **Contexto:** hace falta separar datos crudos, datos limpios y dataset final para garantizar trazabilidad y reproducibilidad, sin añadir infraestructura innecesaria.
- **Decisión:** pipeline en tres capas implementadas como **directorios de ficheros**, sin motor de base de datos:
  - **Bronze:** datos crudos tal cual se descargan, archivados por fuente y fecha de ingesta, sin modificar.
  - **Silver:** limpieza, normalización, tipado, conversión a **Parquet**, cruce espacio-temporal y controles de calidad (duplicados, cobertura, valores fuera de rango).
  - **Gold:** variables derivadas (`racha_dias_secos`, `precip_acum_7d`, `mes`…), dataset de entrenamiento, predicciones diarias y explicaciones SHAP.
- **Alternativas descartadas:** SQLite o PostgreSQL + PostGIS, que quedan como ampliación opcional porque el volumen previsto no las justifica para el MVP.
- **Consecuencias:** la unicidad y las claves foráneas no las garantiza el formato, sino los **tests del pipeline** (ver DEC-14).

### DEC-11. Hugging Face Datasets como almacén de datos e ingesta automática

- **Fecha:** NF2 · **Estado:** Aceptada
- **Contexto:** los datos son demasiado voluminosos para versionarlos en GitHub, pero deben ser compartidos por todo el equipo y accesibles desde la ingesta automática.
- **Decisión:** el **código** vive en GitHub y los **datos** en un dataset de **Hugging Face** (`cpuertolas/piriscope`), con `raw/` por fuente y marca de tiempo, y `data/bronze`, `data/silver`, `data/gold` y `models/` en Parquet. La ingesta automática se ejecuta con un **workflow programado (cron) de GitHub Actions** que sube los ficheros originales; el análisis los lee con pandas.
- **Seguridad:** se usa un **token *fine-grained* con permiso de escritura solo sobre el dataset**, guardado como `HF_TOKEN` en los *Secrets* del repositorio.
- **Consecuencias:** las carpetas `data/` del repositorio no se versionan (`.gitignore`, solo `.gitkeep`). El README documentará cómo obtener los datos y reproducir el pipeline.

### DEC-12. Aplicación monolítica en Streamlit con predicciones precalculadas

- **Fecha:** NF2 · **Estado:** Aceptada
- **Contexto:** el procesamiento es batch (una predicción diaria) y el equipo es pequeño; no hace falta responder a consultas en tiempo real.
- **Decisión:** **arquitectura monolítica basada en Streamlit**, con frontend y backend en un único componente. Las predicciones y las explicaciones SHAP **se calculan en el proceso batch** y se escriben en Gold; Streamlit **solo lee la capa Gold**, sin cargar ni ejecutar el modelo y sin acceder a fuentes externas.
- **Alternativas descartadas:** Flask + React o una API separada de inferencia, por añadir complejidad innecesaria para el MVP.
- **Consecuencias:** el dashboard es ligero y fácil de desplegar; cualquier cambio en las predicciones requiere volver a ejecutar el pipeline.

### DEC-13. Módulos desacoplados por ficheros y orquestador único

- **Fecha:** NF2 · **Estado:** Aceptada
- **Contexto:** aunque el despliegue sea monolítico, la lógica no debe residir en la interfaz.
- **Decisión:** el código de `src/` se organiza en módulos (`ingest`, `features`, `ml`, `explainability`, `dashboard`) que **se comunican únicamente mediante ficheros Parquet** en `data/` y el modelo serializado en `models/`. Un único orquestador, `run_pipeline.py`, **parametrizable por rango de fechas**, encadena ingesta → Silver → Gold → entrenamiento/inferencia.
- **Consecuencias:** cada módulo se puede probar por separado. No se prevén colas de mensajería, autenticación ni servicios externos más allá de las descargas de datos. El pipeline debe poder reejecutarse con otro rango temporal sin tocar el código.

---

## Modelo de datos

### DEC-14. Identificadores y claves lógicas

- **Fecha:** NF2 · **Estado:** Aceptada
- **Decisión:**
  - `id_municipio`: código INE vigente de **5 cifras guardado como texto** (conserva los ceros iniciales).
  - `id_obs_meteo` e `id_indicador_fwi`: derivados de `id_municipio + fecha` (p. ej. `15001_2020-05-14`).
  - `id_prediccion`: derivado de `id_municipio + fecha_objetivo + version_modelo` (p. ej. `15001_2026-10-05_v1`), para que convivan predicciones de distintas versiones del modelo.
  - `id_evento`: campaña + número de parte de EGIF; si el número se repite entre provincias se añade la provincia (se comprobará en el EDA).
  - `ExplicacionPrediccion`: clave compuesta `id_prediccion + variable`.
- **Consecuencias:** las relaciones son lógicas; la unicidad se garantiza con **tests de duplicados** en el pipeline.

### DEC-15. Trazabilidad mediante `fecha_insert` y `version_modelo`

- **Fecha:** NF2 · **Estado:** Aceptada
- **Decisión:** todas las entidades de Silver y Gold incluyen la columna técnica **`fecha_insert`** (marca de tiempo UTC). Las predicciones incluyen además **`version_modelo`**. Cada ejecución **sobrescribe** los datos del periodo reprocesado.
- **Alternativas descartadas:** conservar versiones históricas de cada fila (duplica volumen sin aportar al MVP).
- **Consecuencias:** se puede saber cuándo se generó un valor anómalo, pero no se conserva su versión anterior. Bronze mantiene siempre el dato original.

---

## Tecnologías

### DEC-16. Python + pandas (PySpark solo si se justifica)

- **Fecha:** NF2 · **Estado:** Aceptada
- **Decisión:** **Python** como lenguaje y **pandas** para el procesamiento de datos.
- **Alternativas descartadas:** R (menor integración con el resto del stack); PySpark, que requiere infraestructura distribuida no justificada para unos pocos millones de filas.
- **Consecuencias:** PySpark queda como ampliación opcional **si el volumen o los tiempos de ejecución lo justifican**.

### DEC-17. XGBoost y/o Random Forest como modelos

- **Fecha:** NF2 · **Estado:** Aceptada
- **Decisión:** modelos de árboles (**XGBoost y/o Random Forest**), adecuados para datos tabulares, relaciones no lineales y compatibles con SHAP.
- **Alternativas descartadas:** redes neuronales densas, por complejidad no justificada con el volumen previsto.
- **Consecuencias:** el entrenamiento es *offline*; el modelo se serializa versionado en `models/` y solo lo usa el módulo de inferencia. **Pendiente:** registrar el modelo finalmente elegido y sus hiperparámetros.

### DEC-18. SHAP para la explicabilidad

- **Fecha:** NF2 · **Estado:** Aceptada
- **Decisión:** **SHAP** para explicar cada predicción. Los valores se calculan en el proceso batch y se guardan en Gold (`ExplicacionPrediccion`).
- **Alternativas descartadas:** LIME, para mantener una metodología homogénea y eficiente con modelos de árboles.

### DEC-19. GeoPandas + Plotly para lo geoespacial y los mapas

- **Fecha:** NF2 · **Estado:** Aceptada
- **Decisión:** **GeoPandas** para asociar estaciones y partes a municipios y **Plotly** para el mapa coroplético dentro de Streamlit, sin servidor de mapas.
- **Alternativas descartadas:** Folium y PostGIS, por mayor complejidad para el MVP.

### DEC-20. pytest + GitHub Actions

- **Fecha:** NF2 · **Estado:** Aceptada
- **Decisión:** pruebas con **pytest**, ejecutadas automáticamente en cada Pull Request mediante **GitHub Actions**.
- **Cobertura mínima prevista:** generación de features (incluido un test de **ausencia de información posterior a la fecha de predicción**), transformación y normalización, cruce espacial y temporal (asignación de código INE y de estación), generación de la variable objetivo y un *smoke test* de arranque del dashboard leyendo `predicciones.parquet`.
- **Alternativas descartadas:** solo pruebas manuales, por menor trazabilidad.

---

## Validación y evaluación del modelo

### DEC-21. Partición temporal congelada

- **Fecha:** NF2 · **Estado:** Aceptada (años pendientes)
- **Contexto:** una división aleatoria filtraría información del futuro al entrenamiento.
- **Decisión:** años anteriores para entrenamiento y posteriores para validación y test; **el último año completo disponible es el test final**. La partición se fija **durante el EDA, antes de crear features y ajustar hiperparámetros**, y no se modifica. Se materializa en la columna `particion`.
- **Pendiente:** registrar aquí la división concreta de años cuando se conozca el periodo común de AEMET, EGIF y FWI.

### DEC-22. Partición geográfica por bloques de municipios

- **Fecha:** NF2 · **Estado:** Aceptada
- **Contexto:** hay que comprobar que el modelo generaliza a municipios no vistos y no memoriza el histórico de cada uno.
- **Decisión:** se reserva un subconjunto de municipios que **nunca se usa en entrenamiento** (`reservado = 1`). La selección:
  - se hace **por bloques de municipios próximos** (comarca o estación AEMET asignada), porque los vecinos comparten clima y reservar municipios sueltos daría resultados optimistas;
  - se **estratifica por provincia y por frecuencia histórica de incendios**;
  - se fija con una **semilla** y se guarda en `docs/municipios_reservados.csv`, sin modificarse durante el proyecto.

### DEC-23. Tres escenarios de evaluación (A, B, C)

- **Fecha:** NF2 · **Estado:** Aceptada
- **Decisión:** modelo y baseline FWI se evalúan **sobre exactamente los mismos datos** en tres escenarios:

  | Escenario | Filtro | Pregunta |
  |-----------|--------|----------|
  | A. Temporal | `reservado = 0` y `particion` ∈ {validación, test} | ¿Funciona como sistema operativo? |
  | B. Geográfico | `reservado = 1` y `particion = entrenamiento` | ¿Generaliza a otros municipios? |
  | C. Temporal y geográfico | `reservado = 1` y `particion` ∈ {validación, test} | Caso más exigente |

  El entrenamiento usa solo `reservado = 0` y `particion = entrenamiento`. Si el volumen lo permite, se añade una validación *leave-one-province-out*.
- **Consecuencias:** una caída notable en B o C **no se considera un fallo del proyecto**, sino una limitación que se documenta.

### DEC-24. Métricas y tratamiento del desbalanceo

- **Fecha:** NF2 · **Estado:** Aceptada
- **Decisión:** métricas principales **Recall** y **AUC-PR** (adecuadas para clases desbalanceadas), **desagregadas por provincia** y con **intervalos de confianza por bootstrap** en los escenarios B y C. El desbalanceo se trata con **pesos de clase** y/o técnicas de balanceo cuando proceda.
- **Alternativas descartadas:** *accuracy* y ROC-AUC como métricas principales, por ser engañosas con muy pocos positivos.

### DEC-25. Variables excluidas del modelo

- **Fecha:** NF2 · **Estado:** Aceptada
- **Decisión:** no se usan como variables del modelo:
  - **Identificadores y coordenadas sin procesar** (`id_municipio`, `id_estacion_aemet`, centroides, coordenadas UTM), para que el modelo no reconozca municipios concretos.
  - **Información conocida a posteriori** (`causa`, `fecha_extincion`, `superficie_forestal_ha`), para evitar fuga de información.
- **Consecuencias:** se priorizan variables meteorológicas y derivadas generalizables. `dist_estacion_km` se conserva como indicador de calidad.

---

## Proceso, ética y accesibilidad

### DEC-26. Flujo Git basado en Pull Requests

- **Fecha:** NF1–NF2 · **Estado:** Aceptada
- **Decisión:** no se commitea directamente sobre `main`. Cada tarea tiene una issue asociada y se desarrolla en una rama `tipo/descripción` (`feat/`, `fix/`, `docs/`, `chore/`, `data/`…). Se integra mediante **Pull Request con *squash merge***, con pytest en GitHub Actions y revisión de al menos otro integrante en los cambios relevantes. Los criterios de aceptación de cada historia se revisan como *checklist* en el PR. Detalle en [CONTRIBUTING.md](../CONTRIBUTING.md).
- **Alternativas descartadas:** GitLab, por normas del proyecto.

### DEC-27. Datos personales y licencias

- **Fecha:** NF2 · **Estado:** Aceptada
- **Decisión:** el proyecto **no trata datos personales**. De los partes de EGIF solo se conservan los campos necesarios (**minimización**) y la información se agrega a nivel municipio-día. Todas las fuentes son públicas y se citarán en el README según sus condiciones:
  - **AEMET:** «Fuente: AEMET», sin sugerir respaldo de AEMET.
  - **EGIF-MITECO:** «Origen de los datos: Ministerio para la Transición Ecológica y el Reto Demográfico».
  - **EFFIS/Copernicus:** CC BY 4.0, con atribución e indicación de cambios.
  - **INE:** «Elaboración propia con datos extraídos del sitio web del INE: www.ine.es».
  - **CNIG/IGN:** «© Instituto Geográfico Nacional».
- **Consecuencias:** el RGPD no resulta de aplicación sustancial. El dashboard indica que Piriscope es una **herramienta experimental de apoyo**, no un sistema oficial, y presenta las predicciones como estimaciones probabilísticas para evitar interpretaciones erróneas o la estigmatización de municipios.

### DEC-28. Criterios de accesibilidad del dashboard

- **Fecha:** NF2 · **Estado:** Aceptada
- **Decisión:** el dashboard usa **paletas aptas para daltonismo**, **no depende solo del color** para comunicar el riesgo (etiqueta textual y valor numérico), incluye una **tabla ordenable de municipios** como alternativa al mapa, textos descriptivos y tamaño de fuente legible.

### DEC-29. Control de alcance y punto de control S5–S6

- **Fecha:** NF2 · **Estado:** Aceptada
- **Decisión:** el núcleo del proyecto es **EGIF + AEMET + FWI**. Vegetación/NDVI, terreno, bases de datos, PySpark, nuevas fuentes y la automatización completa son **opcionales**. En las semanas 5–6 hay un **punto de control**: si el cruce de datos no está validado, se recorta a las funcionalidades mínimas.

### DEC-30. Diseño de la ingesta en Bronze

- **Fecha:** 2026-10-17 · **Estado:** Aceptada
- **Responsable:** Cristina (ingesta) · **Issue/PR:** feat/ingesta-datos
- **Contexto:** HU-01, HU-02 y HU-03 piden descargar o incorporar AEMET, EGIF y FWI de forma reproducible, con reintentos, caché, trazabilidad y archivo en Hugging Face.
- **Decisión:**
  - Módulo `src/ingest` con una función `ingestar()` por fuente y CLI `python -m src.ingest <fuente>`; `run_pipeline.py --etapas ingesta` las encadena.
  - Cada ejecución crea `data/bronze/<fuente>/<AAAAMMDDTHHMMSSZ>/` con los ficheros **tal como llegan** y un `manifest.json` (parámetros, bytes, SHA-256 y origen de cada fichero, incidencias, atribución y commit del código). Se sube a Hugging Face en `raw/<FUENTE>/<timestamp>/` (apartado 5.3).
  - **AEMET:** una petición por estación gallega y tramo de ≤ 180 días (límite de 6 meses de la API), pausa entre peticiones y espera ante `429`. Los tramos con más de 30 días de antigüedad se reutilizan de ejecuciones previas (caché) para descargar el histórico una sola vez. Se genera `cobertura_estaciones.csv`.
  - **EGIF:** sin API; se ingieren las exportaciones Excel del buscador público dejadas en `data/landing/egif/`. Además de copiar los originales, se genera `egif_galicia_unificado.parquet` (cuatro provincias, columnas normalizadas, sin duplicados, `id_evento`), informe de duplicados y recuentos por provincia y año, contrastables con `docs/egif_estadisticas_oficiales.csv`. El tipado y el código INE se dejan para Silver.
  - **FWI:** dataset `cems-fire-historical-v1` del Early Warning Data Store vía `cdsapi`, un fichero NetCDF por mes recortado a la caja de Galicia (si el servidor rechaza el recorte, se descarga sin él y se anota).
  - **INE/CNIG:** el diccionario de municipios del INE se descarga automáticamente; los límites del CNIG se incorporan a mano desde `data/landing/ine_cnig/` porque su descarga exige aceptar la licencia en la web.
  - **Ingesta automática:** workflow `ingesta.yml` diario para AEMET (últimos 10 días, por el retraso de publicación) y mensual para el FWI (mes anterior, `intermediate_dataset`).
- **Alternativas descartadas:** petición `todasestaciones` de AEMET (descarga toda España y obliga a filtrar después); scraping del buscador de EGIF (frágil y sin API estable).
- **Consecuencias:** la caché local no existe en GitHub Actions, por lo que el cron sólo pide días recientes; el histórico se carga una vez en local y se sube con `--subir-hf`. **Pendiente de verificar con datos reales:** nombres exactos de columna de las exportaciones de EGIF (ampliar `MAPEO_COLUMNAS` si hace falta) y que el EWDS acepte el recorte `area`.

### DEC-31. Periodo y frecuencias de ingesta

- **Fecha:** 2026-10-18 · **Estado:** Aceptada (periodo pendiente del EDA)
- **Responsable:** Cristina (ingesta) · **Issue/PR:** feat/plan-ingesta
- **Contexto:** hay que fijar desde cuándo se cargan los datos y cada cuánto se actualiza cada fuente.
- **Decisión:**
  - **Carga inicial 2001–actualidad** para AEMET y FWI; EGIF, todo lo disponible (mínimo 2001). Se carga ancho y el periodo común se recorta en el EDA, porque EGIF sólo tiene años definitivos hasta 2021.
  - **AEMET diaria** (últimos 10 días, por el retraso de publicación). **FWI semanal** (últimos 14 días disponibles, con 5 días de retraso por ERA5). **EGIF anual**, con una recarga en S7. **INE anual**. **CNIG una vez**.
  - El FWI se pide al `consolidated_dataset` y, para los meses aún no consolidados, se recurre al `intermediate_dataset` (`fwi_AAAA_MM_intermedio.nc`). En Silver prevalece el consolidado.
  - Carga inicial por años completos con `scripts/carga_inicial.sh` / `.ps1`, reanudable gracias a la caché de Bronze.
- **Alternativas descartadas:** FWI diario (los últimos días aún no existen en ERA5); carga inicial desde GitHub Actions (el workflow se corta a las 2 horas y no conserva la caché).
- **Consecuencias:** procedimiento detallado en [plan_ingesta.md](plan_ingesta.md). **Pendiente:** registrar el periodo común real tras la carga inicial y cerrar DEC-21.

### DEC-32. Diseño de la capa Silver

- **Fecha:** 2026-10-24 · **Estado:** Aceptada
- **Responsable:** Guillermo (Data Engineering) · **Issue/PR:** feat/capa-silver
- **Contexto:** HU-04 pide unir espacial y temporalmente meteorología, incendios y FWI a nivel municipio-día; el apartado 5.2 asigna a Silver la limpieza, el tipado, la conversión a Parquet, el cruce y los controles de calidad.
- **Decisión:**
  - Módulo propio `src/silver` (separado de Gold), CLI `python -m src.silver` y etapa `silver` de `run_pipeline.py`.
  - Lee las ejecuciones de Bronze locales (`data/bronze/<fuente>/`) o su copia de Hugging Face (`data/hf/raw/<FUENTE>/`, `--descargar-hf`); si un timestamp está en ambas, prevalece la local.
  - Una tabla Parquet por entidad Silver del modelo de datos (`municipio`, `estacion_aemet`, `observacion_meteorologica`, `evento_incendio`, `indicador_fwi`) y los informes de `data/silver/informes/`.
  - Cada tabla tiene **exactamente las columnas del apartado 7 y el Anexo A** (`almacen.COLUMNAS`) más `fecha_insert`. Lo que el cruce calcula de más (celda FWI, origen consolidado/intermedio, método de asignación INE, nombre en EGIF…) va a `informes/`, no a las tablas; los polígonos municipales se leen de Bronze y no se guardan en Silver.
  - Identificadores como texto, `fecha` como `date32`, `fecha_insert` en UTC. Con `--desde/--hasta` sólo se sustituye ese periodo (DEC-15).
  - Textos sin espacios en los bordes, dobles ni duros y vacíos como nulo. Los nombres conservan su forma oficial con mayúscula inicial (no se pasan a minúsculas ni se quitan tildes); para comparar se usa una clave normalizada sólo en memoria. `nombre` del municipio es el del CNIG (orden natural, «A Cañiza»); el de la estación AEMET pasa de mayúsculas a mayúscula inicial («Santiago de Compostela Aeropuerto»). `reservado` se toma de `docs/municipios_reservados.csv` cuando ML lo publique; hasta entonces es nulo.
  - El centroide municipal (DEC-08, DEC-09) es el centroide geométrico del recinto del CNIG, calculado en UTM 29N. En un municipio muy cóncavo puede caer fuera de su propio término: se usa igual, porque es lo que fija el NF2, y la lista de esos municipios se guarda en `informe_calidad.json` (`municipio.centroides_fuera_del_municipio`). Hoy sólo le pasa a Barbadás (32008), a 66 m del límite. Descartado `representative_point()`, que siempre cae dentro pero no es el centroide del NF2; la comprobación `centroides_en_galicia` del validador sólo mira la caja de Galicia.
  - `--subir-hf` sube `data/silver` a `data/silver/` del dataset y borra allí las tablas e informes que ya no existen en local. Requiere un token *fine-grained* del dueño del dataset con escritura sobre él (un token *write* personal recibe 403).
- **Alternativas descartadas:** `src/features` compartido con Gold, para trabajar en paralelo sin conflictos. Columnas extra de trazabilidad en las tablas (`nombre_ine`, `id_provincia`, celda FWI, `metodo_asignacion`…) y `municipio_limites.parquet`: no están en el modelo de datos y repetían valores fijos por municipio en millones de filas.
- **Consecuencias:** Gold sólo necesita leer `data/silver` (o generarlo con `python -m src.silver --descargar-hf`). `pd.read_parquet` devuelve `fecha` como `datetime.date`; `src.silver.almacen.leer()` la devuelve como `datetime64`.

### DEC-33. Meteorología en Silver: estación fija y huecos nulos

- **Fecha:** 2026-10-24 · **Estado:** Aceptada
- **Responsable:** Guillermo · **Issue/PR:** feat/capa-silver
- **Contexto:** las 58 estaciones gallegas tienen huecos y no todas miden lo mismo; los tramos de AEMET se solapan entre ingestas.
- **Decisión:**
  - Aplicando DEC-08 tal cual: a cada municipio se le asigna **una única estación**, la más cercana a su centroide entre las que tienen algún dato (según `cobertura_estaciones.csv` de todas las ingestas, para que la asignación no cambie con el periodo reprocesado). Resultado: 56 estaciones usadas, distancia media 10,6 km y máxima 37,8 km (Negueira de Muñiz).
  - `observacion_meteorologica` tiene una fila por municipio y día; los días sin dato de la estación asignada quedan **nulos** (no se imputa en Silver).
  - Si un día de una estación aparece en varias ingestas, prevalece la **más reciente** (AEMET corrige con retraso).
  - Tipado: coma decimal; `Ip` (inapreciable) = 0,0 mm; `Acum`/`Varias` = nulo.
  - Sólo las cuatro variables del Anexo A, con la media diaria de AEMET: `temperatura` (`tmed`), `humedad_relativa` (`hrMedia`), `viento` (`velmedia`) y `precipitacion` (`prec`). Si el EDA elige otra medida, se cambia el campo de origen, no se añaden columnas.
- **Alternativas descartadas:** estación más cercana con dato cada día (menos huecos, pero se aparta de DEC-08 y mezcla estaciones en una misma serie). Conservar máximas, mínimas, racha, dirección, insolación y presión hasta el EDA (no están en el Anexo A).
- **Consecuencias:** con el histórico actual, ~18 % de días sin precipitación, ~17 % sin temperatura y ~56 % sin viento. La imputación queda pendiente para Gold/ML. Como la estación es fija, la de un municipio puede no tener dato en años enteros (172 municipios tienen algún año sin ninguna temperatura; en 39 de ellos el % global pasa del 80 %): `informes/cobertura_meteo_municipios_anual.csv` da la cobertura por municipio y año, y el validador avisa si cualquiera de las cuatro variables supera el 30 % de nulos (hoy, el viento).

### DEC-34. FWI en Silver: celda de tierra más cercana y prioridad del consolidado

- **Fecha:** 2026-10-24 · **Estado:** Aceptada
- **Responsable:** Guillermo · **Issue/PR:** feat/capa-silver
- **Contexto:** el FWI llega en rejilla de 0,25° con longitudes 0–360; las celdas de mar no tienen valor, y hay meses descargados del dataset intermedio y del consolidado.
- **Decisión:** muestreo en el centroide (DEC-09) tomando la **celda con dato más cercana**; la celda y su distancia (media 9,2 km, máximo 16,9 km) van a `informes/asignacion_celdas_fwi.csv`. La celda se fija **una sola vez**, con el consolidado más reciente de todo Bronze (no del periodo reprocesado), igual que la estación fija de DEC-33: si una celda de tierra viniera vacía un mes, ese mes queda nulo (y se avisa en el log) en vez de cambiar de celda sin que se note. Para cada municipio y día prevalece el **consolidado** sobre el intermedio y, a igualdad, la ingesta más reciente; los meses que siguen siendo intermedios se listan en `informe_calidad.json` (`meses_intermedio`) para recargarlos.
- **Alternativas descartadas:** interpolación bilineal o media zonal (DEC-09 ya las deja como mejora futura).
- **Consecuencias:** en Bronze faltan nueve meses de 2002 (enero–agosto y noviembre); se listan en `informe_calidad.json` (`meses_sin_fichero`) y deben recargarse en la ingesta.

### DEC-35. Asignación del código INE a los partes de EGIF

- **Fecha:** 2026-10-24 · **Estado:** Aceptada
- **Responsable:** Guillermo · **Issue/PR:** feat/capa-silver
- **Contexto:** HU-04 exige que al menos el 95 % de los partes tenga código INE, con informe de no asignados, muestra de validación y tabla de equivalencias.
- **Decisión:**
  1. Nombre + provincia del parte contra el catálogo vigente (nombre del CNIG y del INE, normalizados: sin tildes y con el artículo antepuesto) y `docs/tabla_equivalencias.csv` (fusiones Oza-Cesuras 2013 y Cerdedo-Cotobade 2016, y variantes de nombre de EGIF).
  2. Si no casa en su provincia, por nombre en el resto de Galicia cuando el nombre es único.
  3. Validación con las coordenadas UTM cruzadas con los recintos del CNIG. Datum `ETRS89` si el parte lo indica; si viene vacío se asume **ED50**.
  4. Si el nombre no casa pero las coordenadas caen en Galicia, se usa el municipio de las coordenadas.
  - Si nombre y coordenadas discrepan, **prevalece el nombre**: las discrepancias están a una mediana de 27 km del municipio declarado, lo que apunta a errores de coordenadas y no a un problema de datum.
  - `causa` conserva el formato de EGIF (código y texto) sin el espacio doble: «[400] Intencionado». Descartado guardar sólo el código con un catálogo aparte o reformatearlo («400 - Intencionado»): Silver limpia, no cambia la forma del dato.
  - El método seguido se resume en `informe_calidad.json`; los partes con discrepancia o asignados sólo por coordenadas van a `informes/incendios_asignacion_a_revisar.csv`. La tabla `evento_incendio` sólo tiene las columnas del Anexo A.
  - De la exportación sólo se conservan los campos necesarios (DEC-27); `motivacion` y similares no pasan a Silver.
  - **Muestra de validación manual (50 partes):** se genera una sola vez con `python -m src.silver.muestra_validacion` en `docs/incendios_muestra_validacion.csv`, versionada en git y fuera del pipeline, para que no cambie con los datos ni se pierda lo anotado. Partes desde 2001 con coordenadas (semilla 42): 40 al azar, que estiman el acierto global, y 10 con discrepancia nombre/coordenadas, que se evalúan aparte. Descartado regenerarla en `informes/` en cada ejecución (se borraba la validación) y tomarla entre todos los partes (casi la mitad, anteriores a 2001 o sin coordenadas, no se podían comprobar en el mapa).
- **Resultado (2001–2023):** 99,98 % de partes con código INE; 96,0 % validados por coordenadas, 3,5 % con discrepancia y 0,3 % sin coordenadas. Los no asignados son «PORTUGAL», «OTRA PROVINCIA» y municipios de León y Zamora. Antes de 1983 los partes figuran como «INDETERMINADO» sin coordenadas y quedan sin asignar.
- **Consecuencias:** `informes/incendios_asignacion_a_revisar.csv` identifica (por `id_evento`) los casos dudosos por si Gold quiere excluirlos. **Pendiente:** completar la validación manual de `docs/incendios_muestra_validacion.csv` y anotar aquí el resultado por estrato. **Aviso para la ingesta:** el `MAPEO_COLUMNAS` de `src/ingest/egif.py` no reconoce `detectado`, `extinguido`, `coordenadax/y` ni `superficietotalforestal`; Silver los resuelve con sus propios alias.

### DEC-36. Controles de calidad y validación de Silver

- **Fecha:** 2026-10-24 · **Estado:** Aceptada
- **Responsable:** Guillermo · **Issue/PR:** feat/capa-silver
- **Decisión:** Silver comprueba duplicados de clave, cobertura y rangos físicos (temperatura −30/50 °C, humedad 0–100 %, precipitación 0–500 mm, FFMC 0–101, índices FWI ≥ 0…, ver `src/silver/calidad.py`). Los valores fuera de rango **se ponen a nulo y se registran** en `valores_fuera_rango.csv` (con su clave y fecha; el fichero se escribe siempre y una ejecución parcial sólo sustituye las filas de las tablas y el periodo reprocesados); no se corrigen ni se imputan.
- **Validación de las tablas generadas:** `src/silver/validacion.py` comprueba las tablas de `data/silver` contra el modelo de datos (exactamente sus columnas, tipos, textos sin espacios sobrantes ni vacíos, claves primarias y su formato, claves foráneas, 313 municipios por provincia, serie diaria completa, estación fija, rangos y ≥ 95 % de partes con código INE desde 2001, HU-04). Se ejecuta al final de `python -m src.silver` y de la etapa `silver` de `run_pipeline.py`; si hay **errores**, la etapa falla y no se sube nada a Hugging Face. Los **avisos** (p. ej. `reservado` sin definir o días sin FWI) no la detienen. Resultado en `informes/validacion.json`.
- **Alternativas descartadas:** validar sólo con tests de pytest, porque no ven los datos reales de cada ejecución.
- **Consecuencias:** el tratamiento de faltantes queda para Gold/ML, documentado en «Decisiones pendientes». Con los datos actuales: 94 comprobaciones, 0 errores y 2 avisos. Los tests de pytest (en GitHub Actions en cada PR) cubren la lógica con datos sintéticos y comprueban que el validador detecta tablas defectuosas.

### DEC-37. Contrato de entrada del módulo ML

- **Fecha:** 2026-10-25 · **Estado:** Aceptada
- **Responsable:** Hikari (ML) · **Issue/PR:** feat/ml-explainability
- **Contexto:** la capa Gold aún no existe, pero ML necesita fijar qué columnas y valores recibirá para desarrollarse y probarse en paralelo, a partir del Anexo A.
- **Decisión:** el dataset de entrenamiento debe incluir `id_municipio`, `provincia`, `fecha`, `reservado` (0/1), `particion`, `incendio_d1` (0/1), las cuatro variables meteorológicas, las derivadas (`racha_dias_secos`, `precip_acum_7d`, `mes`) y el FWI con sus cinco componentes. `particion` solo admite `entrenamiento`, `validacion` y `test`, **sin tilde**. `mes` entra como entero 1-12, sin one-hot. Antes de entrenar se valida el esquema (columnas, valores permitidos, sin duplicados por municipio y día) y la integridad de las particiones (orden temporal estricto y ningún municipio reservado en entrenamiento).
- **Alternativas descartadas:** `validación` con tilde (riesgo de problemas de codificación entre Parquet, CSV y Windows); one-hot del mes (los modelos de árboles no lo necesitan).
- **Consecuencias:** Gold debe escribir exactamente esos nombres y valores; si cambian, se modifican las constantes de `src/ml/config.py`. Las variables definitivas siguen condicionadas al EDA, como indica el Anexo A.

### DEC-38. Variantes del modelo y variables permitidas

- **Fecha:** 2026-10-25 · **Estado:** Aceptada
- **Responsable:** Hikari (ML) · **Issue/PR:** feat/ml-explainability
- **Contexto:** DEC-06 exige evaluar el modelo con y sin FWI, y DEC-25 fija las variables excluidas. Hace falta concretar qué conjuntos de variables se entrenan.
- **Decisión:** tres variantes: `con_fwi` (meteorología, derivadas, FWI y sus componentes), `sin_fwi` (meteorología y derivadas) y `solo_meteo` (las cuatro variables meteorológicas, sin derivadas). La lista de columnas que nunca entran al modelo (identificadores, coordenadas, datos conocidos a posteriori, fechas, columnas de trazabilidad, `provincia`, `reservado`, `particion`, `dist_estacion_km` y el objetivo) está en `src/ml/config.py`, y un test comprueba que ninguna variante la incluye. `provincia` solo se usa para estratificar y desagregar métricas.
- **Alternativas descartadas:** usar `provincia` como variable, para mantener un modelo basado en variables meteorológicas generalizables (DEC-25).
- **Consecuencias:** `con_fwi` frente a `sin_fwi` mide el valor que añade el FWI, y `sin_fwi` frente a `solo_meteo`, el de las derivadas. `dist_estacion_km` sigue siendo indicador de calidad (DEC-25); si el EDA sugiere probarla como variable, se añade en `config.py`.

### DEC-39. Protocolo de entrenamiento

- **Fecha:** 2026-10-25 · **Estado:** Sustituida en parte por DEC-47 (periodo de entrenamiento, hiperparámetros y `scale_pos_weight`; el resto sigue vigente)
- **Responsable:** Hikari (ML) · **Issue/PR:** feat/ml-explainability
- **Contexto:** DEC-17 y DEC-24 fijan los modelos y el tratamiento del desbalanceo, pero no el procedimiento concreto de entrenamiento.
- **Decisión:** XGBoost y Random Forest se entrenan solo con filas `reservado = 0` y `particion = entrenamiento`. Desbalanceo: `scale_pos_weight` (negativos entre positivos) en XGBoost y `class_weight="balanced_subsample"` en Random Forest. El *early stopping* de XGBoost mide AUC-PR sobre la validación de municipios **no reservados**; el test y los municipios reservados no se usan nunca durante el entrenamiento. Semilla fija (42). Los hiperparámetros son valores por defecto razonables, sin ajustar.
- **Alternativas descartadas:** *early stopping* sobre municipios reservados (contaminaría los escenarios B y C); técnicas de remuestreo (se valorarán cuando se conozca el desbalanceo real).
- **Consecuencias:** el ajuste de hiperparámetros por AUC-PR sobre validación queda para cuando exista la capa Gold. DEC-17 y DEC-24 siguen pendientes de registrar el modelo final y la estrategia de balanceo.

### DEC-40. Comparación con el baseline FWI

- **Fecha:** 2026-10-25 · **Estado:** Sustituida en parte por DEC-47 (el bootstrap pasa a ser por bloques de 7 días; el resto sigue vigente)
- **Responsable:** Hikari (ML) · **Issue/PR:** feat/ml-explainability
- **Contexto:** DEC-23 y DEC-24 piden comparar modelo y FWI sobre los mismos datos, con Recall, AUC-PR e intervalos de confianza. El FWI no es una probabilidad, así que hay que decidir cómo medir el Recall.
- **Decisión:** modelo y FWI se evalúan sobre **exactamente las mismas filas** (se descartan de ambos las filas sin FWI o sin probabilidad). El FWI se usa como puntuación continua. El AUC-PR es la métrica principal, porque no depende del umbral. El Recall se mide con un umbral **fijado en validación**, con el mismo Recall objetivo para ambos sistemas, nunca en el conjunto evaluado; así, la comparación en test se lee en Precision (falsas alarmas). Los intervalos se calculan por bootstrap sobre filas (500 remuestras, semilla fija), global y por provincia, en los tres escenarios.
- **Alternativas descartadas:** umbral de 0,5 (sin sentido con una prevalencia tan baja); convertir el FWI en probabilidad (añade un modelo más); bootstrap por bloques (más correcto, pero más complejo).
- **Consecuencias:** el bootstrap por filas ignora la correlación entre días y municipios vecinos, así que los intervalos son algo optimistas, sobre todo en B y C. Pendiente valorar bootstrap por bloques (municipio o comarca).

### DEC-41. Niveles de riesgo por Recall objetivo

- **Fecha:** 2026-10-25 · **Estado:** Pendiente (valores provisionales)
- **Responsable:** Hikari (ML) · **Issue/PR:** feat/ml-explainability
- **Contexto:** DEC-04 define tres niveles con umbrales fijados en validación, pero no el criterio para elegirlos.
- **Decisión:** los umbrales se fijan sobre la validación de municipios no reservados mediante Recall objetivo: **medio** = mayor umbral con Recall ≥ 0,8 y **alto** = mayor umbral con Recall ≥ 0,5. Se guardan junto al modelo (`<id>_umbrales.json`) y el código rechaza aplicar los umbrales de otro modelo.
- **Alternativas descartadas:** cuantiles de probabilidad y cortes fijos (0,33 / 0,66), porque no tienen significado operativo con una prevalencia baja.
- **Consecuencias:** «medio o superior» captura ≈80 % de los días con incendio de validación y «alto» ≈50 %, con menos falsas alarmas. Con muy pocos positivos, «medio» puede generar muchas alarmas; se revisará con la Precision real. Los porcentajes 0,8 y 0,5 se reajustan tras el EDA; hasta entonces DEC-04 sigue pendiente.
- **Hallazgos (inferencia 2024-2026, DEC-48):**
  - Los niveles absolutos saturan en episodios de sequía real: 84 de las 1.024 fechas predichas tienen al menos el 95 % de los municipios en «medio» o «alto» (máximo, 100 % el 2024-04-23). Se reparten así: 46 en agosto, 13 en abril, 12 en julio, 6 en junio, 5 en septiembre, 1 en marzo y 1 en octubre.
  - No es un fallo de datos. Todas son filas del modelo principal (con FWI), los nulos son normales (0 % en FWI y componentes) y la meteorología de esos días es extrema: FWI medio 22,6 frente a 8,6 en el histórico de esos meses, FFMC 88,6 frente a 72,8, DC 417 frente a 249, precipitación 0,02 mm frente a 2,62 y racha de días secos 16 frente a 5,9. El SHAP lo dominan `ffmc`, `fwi` y `bui`; `mes` apenas pesa (aparece en el 26 % de las predicciones saturadas, con SHAP medio 0,06), así que tampoco es saturación estacional.
  - El modelo sigue ordenando los municipios en esas fechas (en las cinco más saturadas, entre 76 y 103 probabilidades distintas, de 0,03 a 0,18); lo que se pierde es la capacidad de los niveles absolutos para priorizar.
  - Porcentaje de municipio-días en «medio» o «alto» por mes en 2024-2026: 36 % en abril, 68 % en julio y 82 % en agosto, frente a prevalencias históricas (2009-2023) del 1,5 %, 3,0 % y 5,0 %. Tratamiento en DEC-49.

### DEC-42. Inferencia batch D+1 y salidas de Gold

- **Fecha:** 2026-10-25 · **Estado:** Aceptada
- **Responsable:** Hikari (ML) · **Issue/PR:** feat/ml-explainability
- **Contexto:** DEC-12, DEC-14 y DEC-15 establecen que las predicciones se calculan en batch y se escriben en Gold con trazabilidad.
- **Decisión:** para cada municipio con fila del día D se predice D+1 (`fecha_objetivo = D + 1`). `predicciones.parquet` contiene `id_prediccion`, `id_municipio`, `fecha_objetivo`, `version_modelo`, `probabilidad`, `nivel_riesgo`, `fwi` (copia del día D) y `fecha_insert`. `explicaciones.parquet` contiene `id_prediccion`, `variable` y `valor_shap`. Al escribir, las predicciones nuevas sustituyen a las existentes del mismo municipio y `fecha_objetivo` (aunque tengan otra `version_modelo`) y a sus explicaciones, y se conserva el resto (DEC-48). Por defecto se predice el último día disponible, y se admite un rango de fechas. En producción se predicen todos los municipios, reservados incluidos.
- **Consecuencias:** `version_modelo` (p. ej. `v1`) forma parte del id. Las variantes del modelo comparten versión y se distinguen en el nombre del fichero; en producción solo predice una, así que no hay colisión.
- **Ampliada por DEC-48:** `version_modelo` puede valer `<versión>-respaldo` (predicción con el modelo sin FWI) y una fecha puede tener menos de 313 municipios, porque no se predicen las filas fuera del dominio del modelo ni las que no tienen ninguna variable con dato. La inferencia se ejecuta con `python -m src.ml.inferir`. Como consecuencia, la afirmación anterior de que «en producción solo predice una [variante], así que no hay colisión» deja de ser cierta: la unicidad que importa es una predicción vigente por municipio y fecha, no por `id_prediccion`.

### DEC-43. Explicabilidad SHAP y gráficos desde Gold

- **Fecha:** 2026-10-25 · **Estado:** Aceptada
- **Responsable:** Hikari (ML) · **Issue/PR:** feat/ml-explainability
- **Contexto:** DEC-18 elige SHAP y DEC-28 exige accesibilidad; HU-08 pide las 5 variables más influyentes y una vista global.
- **Decisión:** `TreeExplainer` sobre la fila del día D de cada predicción. Se guardan las 5 variables de mayor |SHAP| por predicción y se calcula la importancia global (media de |SHAP|). Las unidades son **log-odds en XGBoost** y **probabilidad en Random Forest** (`shap_units`). Los gráficos (Plotly) se construyen a partir de las tablas de Gold, sin cargar el modelo, con paleta apta para daltonismo y el sentido de cada contribución (sube o baja el riesgo) también en texto.
- **Alternativas descartadas:** guardar todas las variables (más volumen y poco valor en el dashboard); guardar el valor de cada variable, porque el esquema del Anexo A no tiene esa columna.
- **Consecuencias:** con solo `explicaciones.parquet` el dashboard no puede mostrar «temperatura = 31 °C»; habría que cruzar con Gold o ampliar el esquema (a acordar con Frontend). Los valores en log-odds y en probabilidad no son comparables entre algoritmos.
- Las explicaciones de las filas de respaldo (DEC-48) se calculan con el modelo de respaldo, así que no incluyen variables del FWI. Las figuras de ejemplo (`inferir.py --ejemplo-dir`) se generan siempre con el modelo principal y la última fecha predicha con él; la importancia global se calcula sobre la validación de municipios no reservados.
- Un primer ejemplo generado con las filas sin FWI (municipio 15015, 2026-10-22) se descartó por ser un artefacto del dato ausente (DEC-48); el ejemplo vigente es el municipio 32031 el 2026-10-19.

### DEC-44. Persistencia y versionado de modelos

- **Fecha:** 2026-10-25 · **Estado:** Aceptada (alojamiento pendiente)
- **Responsable:** Hikari (ML) · **Issue/PR:** feat/ml-explainability
- **Contexto:** DEC-11 y DEC-17 prevén modelos serializados y versionados en `models/`.
- **Decisión:** cada modelo se guarda con `joblib` en `models/<versión>_<algoritmo>_<variante>.joblib`, junto a un `.json` con versión, semilla, variables, parámetros, número de filas y fecha de entrenamiento, y un `_umbrales.json` (DEC-41).
- **Consecuencias:** los `.joblib` son pickles, así que solo se cargan ficheros generados por el propio proyecto. Los modelos de Random Forest pueden pesar mucho. ~~Pendiente decidir si `models/` se aloja en Hugging Face en vez de GitHub.~~
- **Decidido:** los `.joblib` no se suben al repositorio (son *pickles* y los binarios no se pueden revisar en un PR); se versionan los `.json` de metadatos y de umbrales, que documentan cada versión. Los dos modelos finales (`con_fwi` y `sin_fwi`, que necesita la inferencia por DEC-48) se publican en Hugging Face cuando se cierre la entrega.

### DEC-45. Selección de municipios reservados

- **Fecha:** 2026-10-25 · **Estado:** Aceptada
- **Responsable:** Hikari (ML) · **Issue/PR:** feat/ml-reserved-municipalities
- **Contexto:** DEC-24 exige una partición geográfica definida y congelada al inicio del proyecto, estratificada por provincia y frecuencia de incendios y hecha por bloques de municipios próximos (HU-06). Hace falta concretar qué es un bloque, cómo se estratifica, qué fracción se reserva, con qué años se mide la frecuencia y cómo se congela.
- **Decisión:**
  - **Bloque:** los municipios que comparten la misma estación AEMET asignada (Silver no incluye la comarca). Un bloque se reserva entero o no se reserva.
  - **Estratos:** provincia (la mayoritaria del bloque) × tercil de frecuencia de incendios del bloque (baja, media, alta). La frecuencia es la tasa de días distintos con al menos un parte de EGIF (`evento_incendio`) entre 2001 y 2021, años anteriores a validación y test (a confirmar con DEC-21).
  - **Selección:** objetivo del 20 % por estrato, eligiendo bloques completos al azar con semilla 42. Como cada estrato reserva al menos un bloque entero, el resultado real supera el objetivo. Los parámetros (fracción 0,20, años 2001-2021, semilla 42) son los valores por defecto del script y se fijaron antes de ver resultados.
  - **Resultado:** 81 municipios reservados (25,9 %) en 18 de 56 bloques: A Coruña 19 de 93 (20,4 %), Lugo 19 de 67 (28,4 %), Ourense 27 de 92 (29,3 %) y Pontevedra 16 de 61 (26,2 %). Días con incendio por municipio entre 2001 y 2021: 225,3 en los reservados y 250,2 en el resto (cociente 0,90). Lista en `docs/municipios_reservados.csv`; parámetros y bloques en `docs/municipios_reservados_meta.json`.
  - **Congelación:** la lista no se modifica durante el proyecto. El script se niega a sobrescribirla sin `--forzar`, y los años, la fracción y la semilla no se cambian tras ver resultados.
- **Alternativas descartadas:** reservar municipios sueltos al azar (los vecinos comparten clima y los resultados saldrían optimistas); reservar por comarca (no está en Silver; se valorará si se incorpora esa tabla); reservar provincias enteras (queda para la validación opcional leave-one-province-out).
  - Se probó también una fracción de 0,12 (el resultado fue 21,1 % con A Coruña en el 10,8 %), y se descartó por dejar esa provincia infrarrepresentada.
- **Consecuencias:**
  - Reservar por estación es una aproximación: un municipio vecino atendido por otra estación puede quedar en entrenamiento, así que los escenarios B y C pueden salir algo optimistas.
  - Con ~20 % de municipios, B y C tienen pocos días con incendio: los intervalos de confianza serán amplios (DEC-40).
  - Al regenerar Silver con esta lista, `municipio.reservado` deja de ser nulo y desaparece el aviso del validador.
  - Cambiar cualquier parámetro daría otra lista y obligaría a reentrenar y reevaluar todo, así que requeriría una nueva decisión.
  - El porcentaje reservado por provincia varía entre el 20 % y el 29 % por el tamaño de los bloques de estación; se tendrá en cuenta al interpretar las métricas por provincia en B y C.

### DEC-46. Dataset de Gold: variables derivadas y etiqueta

- **Fecha:** 2026-10-26 · **Estado:** Aceptada (parámetros provisionales hasta cerrar DEC-21)
- **Responsable:** Hikari (ML) · **Issue/PR:** feat/features-gold
- **Contexto:** Silver no incluye variables derivadas (DEC-07 y el apartado 7 del diseño las sitúan en Gold) y HU-05 exige al menos tres, documentadas, reproducibles y sin información posterior al día D. Hay que fijar sus definiciones, cómo se etiqueta la variable objetivo y cómo se reparten las particiones.
- **Decisión:**
  - **`racha_dias_secos`:** días consecutivos, hasta D incluido, con precipitación < 1 mm. Un día lluvioso la reinicia a 0. Un día sin dato la deja desconocida (NaN) hasta el siguiente día lluvioso: no se supone que los días sin dato fueran secos. Al inicio de la serie la cuenta arranca en 1.
  - **`precip_acum_7d`:** precipitación de los 7 días que terminan en D (D incluido, porque ya se conoce al predecir D+1). NaN si falta algún día de la ventana.
  - **`mes`:** mes del día D.
  - **Sin imputación** en Gold: los nulos de Silver se conservan.
  - **`incendio_d1`:** 1 si el municipio tiene al menos un parte de EGIF con fecha de detección D+1. Los partes sin municipio asignado se ignoran. Las filas cuyo D+1 cae después del final del año de test no llevan etiqueta ni partición: nunca se marcan como 0.
  - **Particiones por año de `fecha`:** entrenamiento hasta 2021, validación 2022 y test 2023 (parámetros del comando; ver DEC-21 y DEC-31).
  - **`reservado`** se toma de `docs/municipios_reservados.csv` (DEC-45), no de la columna de Silver.
  - **Salidas:** `dataset_entrenamiento.parquet` (filas etiquetadas, validado contra el contrato de ML), `dataset_municipio_dia.parquet` (todas las filas, para inferencia) y `gold_informe.json`.
- **Resultado:** 2.950.025 filas (2.628.887 etiquetadas) y 78.706 positivos (2,99 %). Prevalencia por partición: entrenamiento 3,18 %, validación 1,37 %, test 0,75 %. Nulos en filas etiquetadas: racha 23,1 % y acumulada 24,8 %, frente al 19,7 % de la precipitación original.
- **Alternativas descartadas:** contar como seco un día sin dato (sesgaría la racha al alza); relajar la ventana de la acumulada (`--min-dias-acum` < 7), porque la exigencia estricta añade solo 3-5 puntos de nulos; marcar como 0 los días sin cobertura de EGIF (daría falsos negativos); usar el mes de D+1.
- **Consecuencias:**
  - La prevalencia cae de ~6 % (2001-2005) a ~1 % (desde 2018), y la cobertura meteorológica mejora de golpe en 2009 (humedad: de ~70 % a ~13 % de nulos). Los nulos pueden actuar como señal de época. Si se recorta el entrenamiento o se excluyen variables (p. ej. el viento, ~50 % de nulos desde 2012, que identifica estaciones) se decide con la validación y se registra en una decisión aparte.
  - La Precision en test será menor que en validación por la menor prevalencia (0,75 % frente a 1,37 %), sin que eso indique peor modelo.
  - Si DEC-21 cambia los años de partición, se vuelve a generar Gold con las opciones `--anio-*`.

### DEC-47. Configuración del modelo y protocolo de evaluación

- **Fecha:** 2026-10-27 · **Estado:** Aceptada
- **Responsable:** Hikari (ML) · **Issue/PR:** feat/ml-experiments
- **Contexto:** con el dataset de Gold real (DEC-46) hay que fijar el periodo de entrenamiento, las variables, los hiperparámetros y el protocolo de evaluación antes de mirar el test y los escenarios geográficos. Se probaron 10 configuraciones sobre la validación de 2022 (municipios no reservados) y una validación walk-forward de 2016 a 2022. Esto concreta y en parte sustituye a DEC-39 y DEC-40.
- **Decisión:**
  - **Periodo de entrenamiento: desde 2009.** Antes de 2009 la humedad falta en ~70 % de las filas (menos del 13 % desde ese año) y la temperatura y la precipitación en ~50 %; además la tasa de incendios era mucho mayor (~6 % frente a ~1-2 %). Mejora el AUC-PR en las cuatro comparaciones posibles (de +0,001 a +0,004).
  - **Se excluye `viento`:** falta en el 45-72 % de las filas y su ausencia identifica estaciones concretas, lo que podría actuar como identificador de zona. En validación quitarlo no tiene efecto medible (de −0,002 a +0,0005 de AUC-PR).
  - **Hiperparámetros por defecto** (XGBoost, profundidad 5, hasta 500 árboles con parada temprana). No se ajustan más: profundidad 3, aprendizaje lento y sin ponderar dan un AUC-PR de 0,056 a 0,058 (ruido).
  - **`scale_pos_weight = 1`** (sin ponderar la clase positiva; sustituye al cociente negativos/positivos de DEC-39). Mismo AUC-PR, y la probabilidad queda en escala realista: el umbral del nivel medio pasa de 0,57 a 0,027. Random Forest mantiene `class_weight="balanced_subsample"`.
  - **Variantes finales:** `con_fwi` y `sin_fwi`.
  - **Intervalos de confianza por bootstrap por bloques de 7 días** (sustituye al bootstrap por filas de DEC-40, que daba intervalos demasiado estrechos porque los incendios se agrupan en olas) y diferencia de AUC-PR frente al FWI con bootstrap pareado.
  - **Validación walk-forward (2016-2022):** para cada año Y se entrena solo con años anteriores, se usa Y−1 para la parada temprana y los umbrales, y se evalúa en Y, solo con municipios no reservados y sin llegar al año de test.
  - **Dos fases:** desarrollo (solo validación de municipios no reservados) e informe final (A-test, B y C-test), que se ejecuta una sola vez con esta configuración.
- **Resultado:**
  - Validación 2022, `con_fwi`: AUC-PR 0,058 frente a 0,049 del FWI (diferencia +0,009; intervalo por bloques de 7 días de −0,007 a +0,020).
  - Walk-forward 2016-2022, `con_fwi`: supera al FWI en **6 de 7 años**; diferencia media +0,010 (mediana +0,008); AUC-PR medio 0,065 frente a 0,055 (+18 %); el intervalo queda por encima de cero en 4 de 7 años. Solo pierde en 2020 (−0,0003, empate).
  - Walk-forward, `sin_fwi`: supera al FWI en 5 de 7 años; diferencia media +0,004; intervalo por encima de cero en 2 de 7.
  - Un modelo al azar tendría un AUC-PR igual a la prevalencia (0,8-2,4 % según el año): el modelo y el FWI rinden entre 4 y 6 veces más que eso.
- **Alternativas descartadas:** más ajuste de hiperparámetros (sin efecto medible; riesgo de ajustarse al ruido de un solo año); mantener el viento; entrenar con todos los años desde 2001; bootstrap por filas.
- **Consecuencias:**
  - **Lo que se puede afirmar:** el modelo con FWI mejora al FWI de forma consistente pero modesta (~18 % de AUC-PR). El modelo sin FWI es prácticamente equivalente al FWI. No se afirma significación estadística global: un contraste de signos de 6 años de 7 daría p ≈ 0,06 (unilateral) y los años no son independientes.
  - **Los umbrales de riesgo (DEC-41) son orientativos:** el Recall fijado en el año anterior no se reproduce con exactitud en el siguiente (entre 0,67 y 0,91 con objetivo 0,80, tanto en el modelo como en el FWI). Habrá que revisarlos periódicamente.
  - **El informe final se publica sea cual sea su signo.** Si se cambia algo tras verlo, se documenta como excepción, porque esos resultados dejarían de ser una evaluación independiente.
  - **Información nueva** (día del año en lugar del mes, evolución meteorológica reciente, historial de incendios del municipio) queda para una iteración futura; la última requeriría revisar DEC-25.
  - El modelo para operación podrá reentrenarse con los años de validación y test una vez cerrada la evaluación (HU-11).
  - El primer informe final incluyó los años 2001-2008 en el escenario B por un error del script; se corrigió y el informe conserva ambas versiones. No cambió el modelo ni los parámetros.
  - **Corrección del informe final:** la primera ejecución incluyó en el escenario B los años 2001-2008, que el modelo no vio (se entrenó desde 2009), por un error del script. Con esos años, `sin_fwi` salía peor que el FWI (−0,018): un artefacto de la falta de meteorología en esos años. Se corrigió (B se recorta a los años de entrenamiento) y el informe conserva ambas versiones (`B` y `B-todos-los-anios`). A y C no cambiaron; no se modificó el modelo ni ningún parámetro.
  - **Aviso derivado de DEC-48:** el modelo con FWI solo es válido cuando el FWI está disponible; en los días recientes sin FWI se usa el modelo sin FWI, que no mejora al FWI.

### DEC-48. Inferencia real: dominio del modelo, modelo de respaldo y filas sin predicción

- **Fecha:** 2026-10-29 · **Estado:** Aceptada
- **Responsable:** Hikari (ML) · **Issue/PR:** feat/ml-inference
- **Contexto:** la primera inferencia real con el modelo `con_fwi` (2024-01-02 a 2026-10-22) mostró que las tres últimas fechas objetivo (20, 21 y 22 de octubre) no tenían FWI en ninguno de los 313 municipios, porque el FWI llega con unos días de retraso respecto a la meteorología. El modelo se entrenó desde 2009 (DEC-47), años en los que el FWI nunca falta, y XGBoost no sabe qué hacer con un hueco que no ha visto: devolvió probabilidades arbitrarias de entre 0,034 y 0,18 y puso al 100 % de los municipios en «medio» o «alto» (301 de 313 en «alto» el día 22). La figura SHAP de ejemplo de esa ejecución lo reflejaba: sus cinco barras eran componentes del FWI vacío.
- **Decisión:**
  - **Dominio del modelo:** el script calcula qué variables no tuvieron ningún nulo en las filas de entrenamiento (aquí, el FWI y sus cinco componentes y `mes`). Si una de ellas falta en una fila, esa fila queda **fuera del dominio** y no se predice con ese modelo.
  - **Modelo de respaldo (`--modelo-respaldo`):** las filas fuera del dominio se predicen con el modelo sin FWI (`sin_fwi`), con sus propios umbrales de riesgo y sus propias explicaciones SHAP (que no usan variables del FWI). Se marcan con `version_modelo = <versión>-respaldo` (p. ej. `v1-respaldo`), y el `id_prediccion` las distingue.
  - **Filas sin información (`--min-variables`, por defecto 1):** una fila en la que todas las variables, salvo el mes, están vacías no se predice, ni con el modelo principal ni con el de respaldo. Solo recibiría la «climatología» del mes: un valor casi constante cuyo nivel de riesgo lo decide el ruido.
  - **Lo que no se puede predecir se omite:** no se rellena ni se inventa un nivel. En `predicciones.parquet` simplemente no hay fila para ese municipio y fecha, y el dashboard lo muestra como «sin dato». No se añade un cuarto nivel a `nivel_riesgo`, para respetar el esquema de `Prediccion` (bajo, medio, alto).
  - **Una predicción vigente por municipio y fecha:** al escribir, las predicciones nuevas sustituyen a las que ya existían para el mismo (`id_municipio`, `fecha_objetivo`), aunque tengan otra `version_modelo`, junto con sus explicaciones. Así, cuando llega el FWI de unos días predichos con el respaldo, `v1` reemplaza a `v1-respaldo` sin duplicados y sin borrar el resto del histórico, y la ejecución diaria con ventana corta es segura. Antes de escribir se comprueba que no hay duplicados y que las explicaciones coinciden con las predicciones.
  - **`--reemplazar`:** descarta todo el contenido anterior de las salidas; queda solo para empezar de cero.
  - **Escritura atómica y orden:** todo se escribe en ficheros temporales y solo al final se reemplazan los reales. Las figuras de ejemplo se generan después de las salidas: si fallan, el código de salida es 2 pero las predicciones ya están escritas.
  - **Avisos por stderr** (no detienen la ejecución): cobertura inferior a `--min-cobertura` (0,8), fechas con al menos `--aviso-alarmas` (0,95) de municipios en «medio» o «alto», variables meteorológicas con muchos nulos en el último día D y, en el modo «último día», retraso respecto a hoy superior a `--max-retraso`. `--min-variables-respaldo` permite exigir más variables con dato al modelo de respaldo.
  - **Verificación:** `verificar_salidas.py` comprueba unicidad, coherencia entre predicciones y explicaciones, ausencia de FWI en el respaldo, cobertura y coherencia de los niveles con los `_umbrales.json`.
- **Resultado (fechas objetivo 2024-01-03 a 2026-10-22, verificado con `verificar_salidas.py`):** 320.437 predicciones de 320.512 posibles (1.024 fechas × 313 municipios). 319.573 con el modelo principal (`v1`), 864 con el de respaldo (`v1-respaldo`, fechas 20 a 22 de octubre, sin FWI) y 75 sin predicción (25 municipios sin ninguna variable meteorológica con dato, por 3 fechas). Los 25 municipios excluidos recibían la climatología (~0,022), que supera el umbral «medio» del modelo de respaldo (0,0220) pero no el del principal (0,0273): por eso salían en «medio».
- **Alternativas descartadas:** dejar que XGBoost trate el FWI vacío (valores arbitrarios, como se comprobó); imputar el FWI con el último valor disponible (introduce un dato inventado y el FWI cambia con rapidez en los días críticos); reentrenar permitiendo FWI vacío (no hay ejemplos desde 2009 con los que aprenderlo, y volver a incluir los años anteriores empeora los datos); añadir un nivel «sin dato» a `nivel_riesgo`.
- **Consecuencias:**
  - **Los 2 o 3 últimos días se predicen siempre con el respaldo**, mientras el FWI llegue con retraso. Según DEC-47, el modelo sin FWI equivale al FWI, no lo mejora: esas predicciones son menos fiables.
  - **Las predicciones de respaldo varían poco:** dependen solo de la meteorología de la estación (7 a 16 valores distintos por fecha, de 0,019 a 0,027, frente a 115 en un día normal con el modelo completo).
  - **Para el dashboard:** `v1-respaldo` debe mostrarse como «estimación sin FWI», y los municipios y fechas ausentes, como «sin dato». Una fecha puede tener menos de 313 municipios.
  - La inferencia necesita guardados **los dos modelos** (`con_fwi` y `sin_fwi`) con sus umbrales.
  - Si en el futuro se añaden al modelo más variables con cobertura total, entran automáticamente en el dominio.
  - Las filas de 2024 a 2026 no tienen etiqueta (EGIF no llega); las de fechas hasta 2022 no son *out-of-sample* (el modelo se entrenó con 2009-2021 y paró en 2022): sirven para demostrar el flujo, no para evaluar.
  - **Los niveles del respaldo discriminan muy poco:** sus umbrales «medio» y «alto» (0,0220 y 0,0241) están separados por 0,002 y sus probabilidades se mueven en un rango estrecho (0,019 a 0,027), de modo que una diferencia de 0,001 cambia el nivel. Equivale al FWI (DEC-47), no lo mejora.
  - **Desfase de los datos:** la última fecha objetivo (22/10) iba por detrás de la fecha de ejecución (29/10). El FWI y la meteorología llegan con retraso (DEC-31), así que las predicciones más recientes son una estimación con varios días de desfase respecto a hoy.
  - **FWI intermedio:** los meses aún no consolidados usan el dataset intermedio (DEC-31, DEC-34), pero el modelo se entrenó con FWI consolidado. No se ha comparado el efecto sobre las predicciones recientes (pendiente).

### DEC-49. Niveles de riesgo: saturación en episodios de sequía y presentación

- **Fecha:** 2026-10-29 · **Estado:** Pendiente
- **Responsable:** Hikari (ML) · **Issue/PR:** feat/ml-inference
- **Contexto:** con los umbrales de DEC-41 (medio 0,0273 y alto 0,0449, fijados por Recall objetivo sobre 2022), 84 de las 1.024 fechas predichas en 2024-2026 tienen al menos el 95 % de los municipios en «medio» o «alto» (hallazgos en DEC-41). Son episodios de sequía real, no un fallo de datos ni un efecto del mes, pero en ellos el nivel absoluto deja de servir para priorizar la vigilancia.
- **Decisión:** pendiente. Mientras no se cierre se mantienen los niveles de DEC-41 y la probabilidad se publica tal cual (DEC-42). Opciones en estudio:
  1. Mantener el nivel absoluto y mostrar además el percentil del municipio dentro de su fecha, calculado desde `probabilidad`, sin cambiar el esquema de `Prediccion`. Solo como información complementaria, no para definir el nivel. El bloque 5 de `analisis_niveles.py` confirma que el ranking dentro de los días saturados concentra los incendios mejor que el azar (lift 1,4-1,7), aunque de forma modesta. Se calcularía por fecha y por `version_modelo`, para no mezclar el respaldo con el principal.
  2. Fijar los umbrales por presupuesto de alarmas en lugar de por Recall. Con 2022, por ejemplo, «medio» = 20 % (umbral 0,0356, Recall 0,65, Precision 0,046) y «alto» = 5 % (umbral 0,0671, Recall 0,25, Precision 0,071); se estimarían con el walk-forward 2016-2022 de DEC-47, que tiene más positivos que 2022 sola. Reduce las alarmas, pero con umbrales absolutos seguirá marcando casi todo en un episodio de sequía.
  3. Mantener los niveles y documentar la saturación como limitación.
- **Resultados (`analisis_niveles.py`, validación 2022 de municipios no reservados: 84.680 filas, 1.194 positivos, 1,41 %; optimistas porque los umbrales se fijaron ahí):**
  - Los umbrales actuales reproducen sus Recall: «medio o alto» marca el 28,4 % de los municipio-días con Recall 0,80 y Precision 0,040; «alto» marca el 13,8 % con Recall 0,50 y Precision 0,051. Con esa Precision, el 96 % de las alarmas «medio o alto» no coincide con un incendio.
  - El umbral global es la política más eficiente: marcando el 10 % de los municipio-días captura el 40 % de los incendios, y marcando el 20 %, el 65 %.
  - El top-x % de cada día pierde casi la mitad: marcando el 10 % captura el 18 % de los incendios, y el híbrido (top-x % del día y p ≥ «medio») captura el 17 % marcando el 5 %, frente al 25 % del umbral global. Los incendios se concentran en días de riesgo alto en toda la región, y un cupo diario fijo los reparte mal.
  - Con el umbral actual se marca el 81-84 % en julio y agosto (prevalencia del 4,3-4,9 %), pero también el 27 % en enero, el 36 % en febrero y el 30 % en mayo (prevalencia ≈ 1,2-1,4 %), y 0 % en noviembre y diciembre.
  - Calibración: la probabilidad media es 0,0215 frente a una tasa real de 0,0141 (1,5 veces). En los dos deciles superiores sobrestima entre 1,2 y 1,3 veces, y en los inferiores mucho más (de 2 a 40 veces, con probabilidades ≤ 0,02). La prevalencia de entrenamiento era mayor que la de validación (y la de test, 0,75 %), así que la sobrestimación es un efecto de la tasa base y puede seguir aumentando.
  - Ranking dentro de los días saturados (al menos el 50 % de los municipios sobre «medio»; son el 28 % de las filas, con prevalencia del 3,80 %): el top 10 % de cada día tiene Precision 6,4 % (lift 1,7 respecto a la tasa de esos días), el top 20 % tiene 5,4 % (lift 1,4) y el top 30 % tiene 5,3 % (lift 1,4). Hay señal dentro del día, pero modesta: incluso el top 10 % no coincide con un incendio en el 94 % de los casos. La señal principal sigue estando en el nivel del día (el top 10 % global tiene Precision 5,6 % frente a una tasa base del 1,4 %).
- **Propuesta provisional:** mantener los niveles de DEC-41, mostrar el percentil dentro del día como dato complementario (opción 1) y presentar la probabilidad como índice de riesgo. La alternativa de presupuesto de alarmas (opción 2) se decide según cuántas alarmas pueda atender el usuario final y con qué Recall mínimo.
- **Alternativas descartadas:** sustituir los niveles por el top-x % de cada día o por el híbrido (pierden Recall, ver arriba); umbrales por mes calculados solo con 2022 (unos 1.200 positivos en todo el año, demasiado pocos por mes); usar el test de 2023 para ajustar umbrales o calibrar (DEC-47 lo reserva para el informe final).
- **Consecuencias:** el dashboard muestra siempre la probabilidad junto al nivel. Si se adopta la opción 1 o la 2 hay que reflejarlo en DEC-04 y DEC-41. Tampoco se ha comprobado la calibración de la probabilidad (en noviembre y diciembre la media prevista es del 0,6-0,7 % frente a prevalencias históricas del 0,2-0,3 %); el análisis anterior confirma una sobrestimación de 1,5 veces en 2022. Como la prevalencia cambia de un año a otro (3 %, 1,4 %, 0,75 %), una calibración fijada en un año no se mantendría, y calibrar con una transformación monótona no cambia qué filas quedan en cada nivel si los umbrales se reajustan. Se propone presentarla como índice de riesgo y no como probabilidad exacta.

---

## Decisiones pendientes

Se cerrarán durante la adquisición y el EDA y se registrarán como nuevas entradas o actualizando las existentes:

- [ ] Periodo común disponible de AEMET, EGIF-MITECO y FWI, y volumen definitivo del dataset (carga 2001–actualidad, DEC-31).
- [ ] División concreta de años de la partición temporal (DEC-21): confirmar qué años de EGIF son definitivos (DEC-31 habla de 2021; Silver llega a 2023) antes de fijar el último año completo como test.
- [X] Lista de municipios reservados y criterio de bloque usado: comarca o estación (DEC-22).
- [ ] Medidas diarias de temperatura, humedad y viento (media, máxima, mínima…).
- [x] Imputación o descarte de variables meteorológicas con muchos faltantes (viento excluido; el resto se deja con nulos) → DEC-47.
- [ ] Umbral de "precipitación relevante" para `racha_dias_secos`.
- [ ] Formato definitivo de `id_evento` (¿se repite el número de parte entre provincias?) (DEC-14). La ingesta de EGIF lo detecta automáticamente y lo registra en `mapeo_columnas.json` (DEC-30).
- [ ] Umbrales de los niveles bajo / medio / alto (DEC-04).
- [X] Modelo final, hiperparámetros y estrategia de balanceo (DEC-17, DEC-24, DEC-39) → DEC-47.
- [ ] Recall objetivo de los niveles de riesgo (0,8 y 0,5) y cierre de DEC-04 (DEC-41).
- [x] Bootstrap por bloques en lugar de por filas (DEC-40) → DEC-47.
- [x] Alojamiento de los modelos serializados: los `.joblib` no se suben al repositorio y se publicarán en Hugging Face (DEC-44).
- [ ] Valor de cada variable y valor base de SHAP junto a las contribuciones (sin el valor base no se reconstruye la probabilidad): ampliar el esquema de `explicaciones.parquet` o cruzar con Gold (DEC-43).
- [ ] Revisar los umbrales de los niveles de riesgo con el informe final y con la calibración (DEC-41, DEC-47).
- [ ] Valorar nueva información para el modelo (día del año, tendencia reciente, historial municipal) en una iteración futura (DEC-47, DEC-25).
- [ ] Calibrar la probabilidad o presentarla como índice de riesgo en el dashboard: el modelo se entrena con una prevalencia del ~2 % y en 2023 fue del 0,75 % (DEC-47).
- [ ] Dashboard: mostrar `v1-respaldo` como «estimación sin FWI» y las ausencias como «sin dato» (DEC-48).
- [ ] Reducir los días de respaldo: valorar una fuente de FWI más reciente (el retraso actual es de 2 a 3 días) (DEC-48).
- [ ] Valorar umbrales o niveles alternativos: con el Recall del 80 %, en agosto el 82 % de los municipio-días entra en «medio o superior» (DEC-41). Por mes queda descartado por falta de positivos; cierre en DEC-49.
- [ ] Subir los modelos finales a Hugging Face y enlazarlos (DEC-44).
- [ ] Contrastar con las estadísticas oficiales si las etiquetas de 2022 y 2023 son definitivas (DEC-31, DEC-21).
- [ ] Cerrar DEC-49: cómo se muestran los niveles en el dashboard (percentil por fecha, umbrales por presupuesto de alarmas o limitación documentada).
- [ ] Comparar el FWI intermedio con el consolidado en los meses que tengan ambos: el modelo se entrena con el consolidado y las predicciones recientes usan el intermedio (DEC-34, DEC-48).
- [ ] Decidir si se calibra la probabilidad (sobrestimada 1,5 veces en 2022 y dependiente de la tasa base) o se presenta como índice de riesgo; si se calibra, hacerlo con el walk-forward 2016-2022 y no con el test de 2023 (DEC-47, DEC-49).
- [ ] Acordar con Frontend el contrato de lectura de Gold: una predicción vigente por municipio y fecha, `v1-respaldo` como «estimación sin FWI», ausencias como «sin dato» y la probabilidad siempre junto al nivel (DEC-48, DEC-49).
- [ ] Conectar `inferir.py` a `run_pipeline.py` con el rango de fechas (HU-11).

---

## Plantilla para nuevas decisiones

```markdown
### DEC-XX. Título breve de la decisión

- **Fecha:** AAAA-MM-DD · **Estado:** Aceptada | Pendiente | Sustituida por DEC-YY | Descartada
- **Responsable:** nombre · **Issue/PR:** #nn
- **Contexto:** qué problema o necesidad motiva la decisión.
- **Decisión:** qué se ha decidido, de forma concreta.
- **Alternativas descartadas:** qué otras opciones se valoraron y por qué no se eligieron.
- **Consecuencias:** qué implica (ventajas, limitaciones, trabajo derivado).
```

Al sustituir una decisión, no se borra: se cambia su estado a `Sustituida por DEC-YY` y se añade la nueva al final.
