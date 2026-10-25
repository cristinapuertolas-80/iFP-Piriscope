# `src/silver` · Capa Silver

Limpia, normaliza y tipa los datos crudos de Bronze, los convierte a **Parquet** y hace el **cruce espacio-temporal** a nivel municipio-día (apartado 5.2 del diseño técnico, HU-04). Genera las cinco entidades Silver del modelo de datos (apartado 7) y los informes de calidad.

No descarga nada de las fuentes externas: lee las carpetas de la ingesta en `data/bronze/<fuente>/<timestamp>/` o su copia de Hugging Face en `data/hf/raw/<FUENTE>/<timestamp>/`.

## Uso

```bash
# Si no has hecho la ingesta en local, baja antes los originales de Hugging Face (HF_TOKEN de lectura en .env)
python -m src.silver --descargar-hf

# Todas las tablas con todo lo que haya en Bronze
python -m src.silver

# Sólo un periodo: sustituye ese periodo en las tablas con fecha y conserva el resto
python -m src.silver --desde 2024-01-01 --hasta 2024-12-31

# Sólo algunas tablas (municipios se necesita antes que las demás)
python -m src.silver --tablas meteo fwi

# Desde el orquestador
python run_pipeline.py --desde 2001-01-01 --hasta 2024-12-31 --etapas silver

# Subir data/silver al dataset (HF_TOKEN con permiso de ESCRITURA); borra allí las tablas e informes que ya no existen.
# Siempre valida antes de subir: no se admite junto con --sin-validar
python -m src.silver --subir-hf
```

Con todo el histórico (2001–2026, unos 3 M filas municipio-día) tarda menos de un minuto.

## Tablas que genera (`data/silver/`)

Cada tabla tiene **exactamente** las columnas del modelo de datos (apartado 7 y Anexo A del NF2), definidas en `almacen.COLUMNAS`; el validador falla si sobra o falta alguna. Lo que el cruce calcula de más (celda FWI, método de asignación INE…) va a `informes/`.

| Fichero | Entidad | Clave | Columnas |
|---|---|---|---|
| `municipio.parquet` | Municipio | `id_municipio` | `nombre` (CNIG), `provincia`, `lat_centroide`, `lon_centroide`, `reservado` |
| `estacion_aemet.parquet` | EstacionAemet | `id_estacion_aemet` | `nombre`, `latitud`, `longitud`, `altitud` |
| `observacion_meteorologica.parquet` | ObservacionMeteorologica | `id_obs_meteo` (`15001_2020-05-14`) | `id_municipio`, `id_estacion_aemet`, `fecha`, `dist_estacion_km`, `temperatura`, `humedad_relativa`, `viento`, `precipitacion` |
| `indicador_fwi.parquet` | IndicadorFWI | `id_indicador_fwi` | `id_municipio`, `fecha`, `fwi`, `ffmc`, `dmc`, `dc`, `isi`, `bui` |
| `evento_incendio.parquet` | EventoIncendio | `id_evento` | `id_municipio`, `fecha_deteccion`, `fecha_extincion`, `causa`, `superficie_forestal_ha`, `coordenada_x`, `coordenada_y`, `huso`, `datum` |

Todas llevan `fecha_insert` (UTC). Los identificadores son texto; `fecha` se guarda como **fecha** (`date32`); `fecha_deteccion` y `fecha_extincion` conservan la hora del parte (Anexo A).

**Textos:** sin espacios en los bordes, dobles ni duros; los vacíos (`""`, `-`) son nulos. Los nombres que se muestran conservan su forma oficial con mayúscula inicial: `municipio.nombre` es el del CNIG y `estacion_aemet.nombre` pasa de las mayúsculas de AEMET a mayúscula inicial (`Santiago de Compostela Aeropuerto`, `Ourense, Instituto`), sin añadir ni quitar tildes. `causa` mantiene el formato de EGIF sin el espacio doble (`[400] Intencionado`); el código se obtiene con `causa.str.extract(r"\[(\d+)\]")`. Para comparar nombres se usa una clave normalizada sólo en memoria (`municipios.normalizar_nombre`).

> **Leer las tablas:** `pd.read_parquet` devuelve `fecha` como objetos `datetime.date`. Para trabajar con `datetime64` usa
> `from src.silver.almacen import leer; df = leer("observacion_meteorologica")`.

### Variables meteorológicas

| Columna | Campo AEMET | Unidad |
|---|---|---|
| `temperatura` | `tmed` (media diaria) | °C |
| `humedad_relativa` | `hrMedia` (media diaria) | % |
| `viento` | `velmedia` (media diaria) | m/s |
| `precipitacion` | `prec` | mm (`Ip` = 0,0) |

Sólo las cuatro del Anexo A. Si el EDA elige otra medida diaria (p. ej. `tmax`), se cambia el campo de origen en `meteo.VARIABLES_AEMET`, sin añadir columnas. Las variables derivadas (`racha_dias_secos`, `precip_acum_7d`, `mes`…) y la imputación de faltantes **no** se hacen aquí: son de Gold.

## Cómo se hace el cruce

- **Municipio–estación (DEC-08):** cada municipio se asocia a la **estación AEMET más cercana a su centroide** entre las que tienen datos. La asignación es fija para todo el periodo. Los días sin dato de esa estación quedan **nulos**. `dist_estacion_km` es el indicador de calidad (media 10,6 km, máximo 37,8 km).
- **FWI (DEC-09):** se toma la celda de la rejilla de 0,25° más cercana al centroide. Las celdas de mar no tienen FWI, así que a un municipio costero se le da la celda de tierra más próxima (media 9,2 km, máximo 16,9 km; en `informes/asignacion_celdas_fwi.csv`). Si un día está en varios ficheros, el **consolidado prevalece sobre el intermedio**; los meses que aún son intermedios figuran en `informe_calidad.json` → `meses_intermedio`.
- **Incendios (DEC-07):** el código INE vigente se asigna por **nombre + provincia** (con `docs/tabla_equivalencias.csv` para fusiones y variantes). Si el nombre no casa en su provincia, se busca en el resto de Galicia. Después se **valida con las coordenadas UTM**, cruzando el punto con los recintos del CNIG (que se leen de Bronze; no se guardan en Silver). Si el nombre no casa pero las coordenadas caen en Galicia, se usan las coordenadas. El camino seguido (`nombre_validado` · `nombre_coordenadas_discrepan` · `nombre_sin_coordenadas` · `coordenadas` · `sin_asignar`) se resume en `informe_calidad.json` y los casos a revisar van a `informes/incendios_asignacion_a_revisar.csv`.
- **Repetidos entre ingestas:** en AEMET prevalece la ingesta más reciente de cada estación y día.

## Informes (`data/silver/informes/`)

| Fichero | Para qué |
|---|---|
| `informe_calidad.json` | Resumen por tabla: filas, periodo, duplicados de clave, % de nulos, valores fuera de rango, % de partes con código INE… |
| `valores_fuera_rango.csv` | Valores físicamente imposibles que se han puesto a nulo |
| `asignacion_estaciones.csv` | Estación asignada a cada municipio y su distancia (sirve para la reserva por bloques, DEC-22) |
| `asignacion_celdas_fwi.csv` | Celda FWI usada para cada municipio (`lat_celda`, `lon_celda`) y su distancia al centroide |
| `cobertura_meteo_municipios.csv` | % de días con dato por municipio y variable |
| `incendios_no_asignados.csv` | Partes sin código INE agrupados por nombre (HU-04) |
| `incendios_muestra_validacion.csv` | 50 partes al azar (semilla 42) para validar a mano; columna `validacion_manual` vacía (HU-04) |
| `incendios_asignacion_por_campana.csv` | % de partes asignados por campaña y provincia |
| `incendios_asignacion_a_revisar.csv` | Partes en que nombre y coordenadas no coinciden o asignados sólo por coordenadas, con ambos municipios |

## Validación de las tablas

Al terminar, `python -m src.silver` (y la etapa `silver` de `run_pipeline.py`) valida las tablas generadas contra el modelo de datos (`validacion.py`). Si hay errores, devuelve el código 1 y **no sube nada a Hugging Face**. También se puede lanzar sola:

```bash
python -m src.silver.validacion      # resultado en data/silver/informes/validacion.json
```

| Tipo | Qué comprueba |
|---|---|
| Esquema | Exactamente las columnas del modelo de datos (ni faltan ni sobran), ids como texto, `fecha` como fecha (`date32`), `fecha_insert` en UTC y sin nulos |
| Textos | Ninguna columna de texto con espacios en los bordes, dobles o duros, ni cadenas vacías |
| Claves primarias | Únicas y no nulas; `id_municipio` de 5 cifras; `id_obs_meteo` e `id_indicador_fwi` = `id_municipio_fecha`; una fila por municipio y día |
| Claves foráneas | Todo `id_municipio` está en Municipio y todo `id_estacion_aemet` en EstacionAemet |
| Municipio | 313 municipios (93/67/92/61 por provincia), provincia coherente con el código, centroides dentro de Galicia |
| Meteorología | Serie diaria completa para los 313 municipios, una estación fija por municipio, distancia ≤ 50 km, rangos físicos |
| FWI | Los 313 municipios todos los días, rangos, días sin FWI |
| Incendios | `id_evento` único, **≥ 95 % de partes con código INE desde 2001 (HU-04)**, extinción posterior a la detección, huso 29/30 |

Cada comprobación es un **error** (la tabla no cumple el contrato) o un **aviso** (dato a revisar en el EDA). Con los datos actuales: 94 comprobaciones, 0 errores y 2 avisos (`reservado` aún sin definir y los nueve meses de FWI de 2002 que faltan).

## Resultados con los datos actuales

- **Meteorología:** 56 de las 58 estaciones quedan asignadas a algún municipio. Faltan datos en ~18 % de los días de precipitación, ~17 % de temperatura y ~56 % de viento; la imputación queda para Gold/ML.
- **FWI:** sin nulos en los días con fichero.
- **Incendios (2001–2023):** el 99,98 % de los partes tiene código INE. El 96,0 % está validado por coordenadas, el 3,5 % tiene coordenadas que no coinciden con el municipio declarado (mediana de 27 km: son errores de coordenadas, por eso prevalece el nombre) y el 0,3 % no tiene coordenadas. Antes de 1983 los partes figuran como «INDETERMINADO» y sin coordenadas, así que quedan sin asignar.

## Limitaciones y pendientes

- En Bronze faltan nueve meses de FWI de 2002 (enero–agosto y noviembre); aparecen en `informe_calidad.json` → `meses_sin_fichero` y hay que recargarlos en la ingesta.
- `reservado` queda nulo hasta que ML publique `docs/municipios_reservados.csv`.
- Falta completar a mano la columna `validacion_manual` de `incendios_muestra_validacion.csv` (HU-04).
- Si un parte no indica el datum de sus coordenadas, se asume ED50.
- El `MAPEO_COLUMNAS` de `src/ingest/egif.py` no reconoce `detectado`, `extinguido`, `coordenadax/y` ni `superficietotalforestal`; Silver los resuelve con sus propios alias (`ALIAS` en `incendios.py`).

Las decisiones de diseño, con sus alternativas descartadas, están en [docs/decisiones.md](../../docs/decisiones.md) (DEC-32 a DEC-36).

## Tests

```bash
pytest tests/test_silver_*.py
```

- Los tests de lógica usan un Bronze sintético (dos municipios, dos estaciones, un mes de FWI y unos pocos partes) y no necesitan red ni datos reales.
- `test_silver_validacion.py` comprueba además que el validador detecta tablas defectuosas (claves duplicadas, valores fuera de rango, claves foráneas huérfanas, huecos, columnas fuera del modelo, textos con espacios sobrantes…) y, si existen las tablas reales en `data/silver`, que cumplen el contrato. En GitHub Actions no hay datos y ese test se omite.
