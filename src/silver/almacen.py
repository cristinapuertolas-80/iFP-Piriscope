"""Escritura y lectura de las tablas Silver en Parquet.

* Los identificadores se guardan como texto, las fechas como ``date32`` y
  ``fecha_insert`` como marca de tiempo UTC (Anexo A).
* Cada ejecución **sobrescribe sólo el periodo reprocesado** (apartado 7,
  Trazabilidad): si la tabla ya existe y tiene columna de fecha, se conservan
  las filas fuera de ``[desde, hasta]`` y se sustituyen las de dentro.
* Cada tabla tiene **exactamente** las columnas del modelo de datos (apartado 7
  y Anexo A, :data:`COLUMNAS`). Lo que se calcula de más en el cruce
  (celda FWI, método de asignación INE…) va a ``informes/``, no a la tabla.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.ingest import config

log = logging.getLogger("piriscope.silver")

#: Columnas de tipo fecha (sin hora) que se guardan como date32.
COLUMNAS_FECHA = ("fecha",)

#: Columnas de cada entidad Silver según el apartado 7 y el Anexo A (``fecha_insert`` se añade al guardar).
COLUMNAS: dict[str, list[str]] = {
    "municipio": ["id_municipio", "nombre", "provincia", "lat_centroide", "lon_centroide", "reservado"],
    "estacion_aemet": ["id_estacion_aemet", "nombre", "latitud", "longitud", "altitud"],
    "observacion_meteorologica": ["id_obs_meteo", "id_municipio", "id_estacion_aemet", "fecha",
                                  "dist_estacion_km", "temperatura", "humedad_relativa", "viento",
                                  "precipitacion"],
    "indicador_fwi": ["id_indicador_fwi", "id_municipio", "fecha", "fwi", "ffmc", "dmc", "dc", "isi", "bui"],
    "evento_incendio": ["id_evento", "id_municipio", "fecha_deteccion", "fecha_extincion", "causa",
                        "superficie_forestal_ha", "coordenada_x", "coordenada_y", "huso", "datum"],
}


def silver_dir(base_dir: Path | None = None) -> Path:
    d = Path(base_dir or config.SILVER_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def informes_dir(base_dir: Path | None = None) -> Path:
    d = silver_dir(base_dir) / "informes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ruta_tabla(tabla: str, base_dir: Path | None = None) -> Path:
    return silver_dir(base_dir) / f"{tabla}.parquet"


def ahora_utc() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).floor("s")


def _a_arrow(df: pd.DataFrame) -> pa.Table:
    tabla = pa.Table.from_pandas(df, preserve_index=False)
    for col in COLUMNAS_FECHA:
        if col in tabla.column_names:
            i = tabla.column_names.index(col)
            tabla = tabla.set_column(i, col, tabla.column(col).cast(pa.timestamp("ms")).cast(pa.date32()))
    return tabla


def leer(tabla: str, base_dir: Path | None = None) -> pd.DataFrame:
    """Lee una tabla Silver; las fechas se devuelven como ``datetime64``."""
    df = pd.read_parquet(ruta_tabla(tabla, base_dir))
    for col in COLUMNAS_FECHA:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def guardar(df: pd.DataFrame, tabla: str, columna_fecha: str | None = None,
            desde: date | None = None, hasta: date | None = None,
            ordenar_por: list[str] | None = None, base_dir: Path | None = None,
            fecha_insert: pd.Timestamp | None = None) -> Path:
    """Guarda ``df`` en ``data/silver/<tabla>.parquet`` añadiendo ``fecha_insert``.

    Sólo se escriben las columnas de :data:`COLUMNAS`. Con ``columna_fecha`` y
    un rango, sólo se sustituye ese periodo de la tabla existente. Sin rango, la
    tabla se reescribe entera.
    """
    df = df[COLUMNAS[tabla]].copy() if tabla in COLUMNAS else df.copy()
    df["fecha_insert"] = fecha_insert if fecha_insert is not None else ahora_utc()
    ruta = ruta_tabla(tabla, base_dir)

    if columna_fecha and (desde or hasta):
        ini = pd.Timestamp(desde) if desde else pd.Timestamp.min
        fin = pd.Timestamp(hasta) + pd.Timedelta(days=1) if hasta else pd.Timestamp.max
        dentro = lambda s: (s >= ini) & (s < fin)  # noqa: E731
        df = df[dentro(pd.to_datetime(df[columna_fecha]))]
        if ruta.exists():
            previo = leer(tabla, base_dir)
            previo = previo[~dentro(pd.to_datetime(previo[columna_fecha]))]
            log.info("%s: se conservan %d filas fuera del periodo reprocesado", tabla, len(previo))
            df = pd.concat([previo, df], ignore_index=True)

    if ordenar_por:
        df = df.sort_values(ordenar_por, kind="stable")
    df = df.reset_index(drop=True)
    pq.write_table(_a_arrow(df), ruta)
    log.info("Silver %s: %d filas -> %s", tabla, len(df), ruta)
    return ruta
