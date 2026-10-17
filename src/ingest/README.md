# `src/ingest` · Ingesta de datos (capa Bronze)

Descarga o incorpora las cuatro fuentes de Piriscope y las guarda **sin modificar** en `data/bronze/<fuente>/<AAAAMMDDTHHMMSSZ>/`, con un `manifest.json` de trazabilidad. Opcionalmente sube cada ejecución al dataset de Hugging Face en `raw/<FUENTE>/<timestamp>/`.

| Fuente | Módulo | Cómo se obtiene | Historia |
|---|---|---|---|
| AEMET OpenData | `aemet.py` | API (requiere `AEMET_API_KEY`) | HU-01 |
| EGIF-MITECO | `egif.py` | Exportaciones Excel del buscador público, dejadas en `data/landing/egif/` | HU-02 |
| EFFIS / Copernicus | `effis.py` | Early Warning Data Store con `cdsapi` (requiere `CDSAPI_KEY`) | HU-03 |
| INE / CNIG | `ine_cnig.py` | Diccionario de municipios del INE (automático) + límites del CNIG (manual, `data/landing/ine_cnig/`) | HU-04 (insumo) |

## Configuración

```bash
pip install -r environment/requirements.txt
cp .env.example .env      # y rellena AEMET_API_KEY, CDSAPI_KEY, HF_TOKEN
```

## Uso

```bash
python -m src.ingest --help

# AEMET: histórico (una sola vez, reanudable) y actualización diaria
python -m src.ingest aemet --desde 2010-01-01 --hasta 2024-12-31
python -m src.ingest aemet --dias 10 --subir-hf

# EGIF: tras exportar los partes de las 4 provincias a data/landing/egif/
python -m src.ingest egif

# FWI (un fichero por mes, recortado a Galicia)
python -m src.ingest effis --desde 2010-01-01 --hasta 2024-12-31

# Municipios del INE + límites del CNIG
python -m src.ingest ine_cnig --anio 2026

# Subir a Hugging Face lo que esté pendiente / descargar para analizar
python -m src.ingest subir
python -m src.ingest descargar aemet

# Todo a la vez, desde el orquestador
python run_pipeline.py --desde 2020-01-01 --hasta 2024-12-31 --etapas ingesta --subir-hf
```

## Qué deja cada ejecución

```
data/bronze/aemet/20261017T041700Z/
├── manifest.json               # parámetros, ficheros (bytes + SHA-256 + origen), incidencias, atribución
├── inventario_estaciones.json  # inventario completo, tal cual
├── estaciones_galicia.csv      # estaciones seleccionadas, con coordenadas en decimal
├── cobertura_estaciones.csv    # informe de cobertura por estación (HU-01)
└── diarios/<indicativo>_<ini>_<fin>.json

data/bronze/egif/<timestamp>/
├── manifest.json
├── originales/*.xlsx           # exportaciones sin tocar
├── egif_galicia_unificado.parquet   # 4 provincias, sin duplicados, con id_evento (HU-02)
├── informe_duplicados.csv
├── recuentos_provincia_anio.csv     # contraste con docs/egif_estadisticas_oficiales.csv si existe
└── mapeo_columnas.json

data/bronze/effis/<timestamp>/fwi_<AAAA>_<MM>.nc
data/bronze/ine_cnig/<timestamp>/{ine,cnig}/...
```

## Comportamiento relevante

- **AEMET**: protocolo en dos pasos (petición → URL de `datos`), tramos de ≤ 180 días (AEMET limita a 6 meses), pausa entre peticiones, espera y reintento ante `429`, `404` tratado como "sin datos". Los tramos con más de 30 días de antigüedad que ya estén en una ejecución previa de Bronze **no se vuelven a pedir** (caché); `--forzar` lo desactiva.
- **EGIF**: detecta la fila de cabecera aunque la exportación traiga títulos, normaliza los nombres de columna y los mapea a los del Anexo A cuando los reconoce. Quita partes duplicados (misma campaña, provincia y nº) y comprueba si el nº de parte se repite entre provincias: si ocurre, `id_evento = campaña_provincia_número`; si no, `campaña_número` (DEC-14).
- **EFFIS**: pide la caja de Galicia; si el servidor rechaza el recorte, lo reintenta sin él y lo anota como incidencia. Los meses completos ya descargados se reutilizan.
- **Hugging Face**: cada carpeta de ejecución se sube a `raw/<FUENTE>/<timestamp>/` y queda marcada con `.subido_hf`.
- Una fuente que falla no detiene a las demás en `run_pipeline.py`; todas las incidencias quedan en el manifiesto.

## Tests

```bash
pytest
```

No hacen peticiones reales: AEMET, EWDS y Hugging Face se simulan.
