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
  - `nombre` del municipio es el del CNIG (orden natural, «A Cañiza»). `reservado` se toma de `docs/municipios_reservados.csv` cuando ML lo publique; hasta entonces es nulo.
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
- **Consecuencias:** con el histórico actual, ~18 % de días sin precipitación, ~17 % sin temperatura y ~56 % sin viento. La imputación queda pendiente para Gold/ML.

### DEC-34. FWI en Silver: celda de tierra más cercana y prioridad del consolidado

- **Fecha:** 2026-10-24 · **Estado:** Aceptada
- **Responsable:** Guillermo · **Issue/PR:** feat/capa-silver
- **Contexto:** el FWI llega en rejilla de 0,25° con longitudes 0–360; las celdas de mar no tienen valor, y hay meses descargados del dataset intermedio y del consolidado.
- **Decisión:** muestreo en el centroide (DEC-09) tomando la **celda con dato más cercana**; la celda y su distancia (media 9,2 km, máximo 16,9 km) van a `informes/asignacion_celdas_fwi.csv`. Para cada municipio y día prevalece el **consolidado** sobre el intermedio y, a igualdad, la ingesta más reciente; los meses que siguen siendo intermedios se listan en `informe_calidad.json` (`meses_intermedio`) para recargarlos.
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
  - El método seguido se resume en `informe_calidad.json`; los partes con discrepancia o asignados sólo por coordenadas van a `informes/incendios_asignacion_a_revisar.csv`. La tabla `evento_incendio` sólo tiene las columnas del Anexo A.
  - De la exportación sólo se conservan los campos necesarios (DEC-27); `motivacion` y similares no pasan a Silver.
- **Resultado (2001–2023):** 99,98 % de partes con código INE; 96,0 % validados por coordenadas, 3,5 % con discrepancia y 0,3 % sin coordenadas. Los no asignados son «PORTUGAL», «OTRA PROVINCIA» y municipios de León y Zamora. Antes de 1983 los partes figuran como «INDETERMINADO» sin coordenadas y quedan sin asignar.
- **Consecuencias:** `informes/incendios_asignacion_a_revisar.csv` identifica (por `id_evento`) los casos dudosos por si Gold quiere excluirlos. **Pendiente:** completar la validación manual de `incendios_muestra_validacion.csv`. **Aviso para la ingesta:** el `MAPEO_COLUMNAS` de `src/ingest/egif.py` no reconoce `detectado`, `extinguido`, `coordenadax/y` ni `superficietotalforestal`; Silver los resuelve con sus propios alias.

### DEC-36. Controles de calidad y validación de Silver

- **Fecha:** 2026-10-24 · **Estado:** Aceptada
- **Responsable:** Guillermo · **Issue/PR:** feat/capa-silver
- **Decisión:** Silver comprueba duplicados de clave, cobertura y rangos físicos (temperatura −30/50 °C, humedad 0–100 %, precipitación 0–500 mm, FFMC 0–101, índices FWI ≥ 0…, ver `src/silver/calidad.py`). Los valores fuera de rango **se ponen a nulo y se registran** en `valores_fuera_rango.csv`; no se corrigen ni se imputan.
- **Validación de las tablas generadas:** `src/silver/validacion.py` comprueba las tablas de `data/silver` contra el modelo de datos (exactamente sus columnas, tipos, claves primarias y su formato, claves foráneas, 313 municipios por provincia, serie diaria completa, estación fija, rangos y ≥ 95 % de partes con código INE desde 2001, HU-04). Se ejecuta al final de `python -m src.silver` y de la etapa `silver` de `run_pipeline.py`; si hay **errores**, la etapa falla y no se sube nada a Hugging Face. Los **avisos** (p. ej. `reservado` sin definir o días sin FWI) no la detienen. Resultado en `informes/validacion.json`.
- **Alternativas descartadas:** validar sólo con tests de pytest, porque no ven los datos reales de cada ejecución.
- **Consecuencias:** el tratamiento de faltantes queda para Gold/ML, documentado en «Decisiones pendientes». Con los datos actuales: 89 comprobaciones, 0 errores y 2 avisos. Los tests de pytest (en GitHub Actions en cada PR) cubren la lógica con datos sintéticos y comprueban que el validador detecta tablas defectuosas.

---

## Decisiones pendientes

Se cerrarán durante la adquisición y el EDA y se registrarán como nuevas entradas o actualizando las existentes:

- [ ] Periodo común disponible de AEMET, EGIF-MITECO y FWI, y volumen definitivo del dataset (carga 2001–actualidad, DEC-31).
- [ ] División concreta de años de la partición temporal (DEC-21).
- [ ] Lista de municipios reservados y criterio de bloque usado: comarca o estación (DEC-22).
- [ ] Medidas diarias de temperatura, humedad y viento (media, máxima, mínima…).
- [ ] Método de imputación de datos meteorológicos faltantes.
- [ ] Umbral de "precipitación relevante" para `racha_dias_secos`.
- [ ] Formato definitivo de `id_evento` (¿se repite el número de parte entre provincias?) (DEC-14). La ingesta de EGIF lo detecta automáticamente y lo registra en `mapeo_columnas.json` (DEC-30).
- [ ] Umbrales de los niveles bajo / medio / alto (DEC-04).
- [ ] Modelo final (XGBoost o Random Forest), hiperparámetros y estrategia de balanceo (DEC-17, DEC-24).

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
