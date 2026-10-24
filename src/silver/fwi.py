"""Entidad Silver **IndicadorFWI**: FWI y componentes por municipio y día (HU-03).

* Se leen los NetCDF mensuales de Bronze (``fwi_AAAA_MM[_intermedio].nc``),
  se pasan las longitudes de 0–360 a -180–180 y se recorta la caja de Galicia
  (por si el servidor rechazó el recorte en la ingesta).
* El FWI se **muestrea en el centroide** de cada municipio (DEC-09): se toma la
  celda de la rejilla (0,25°) más cercana al centroide entre las celdas con
  dato; las celdas de mar no tienen FWI, así que a un municipio costero se le
  asigna la celda de tierra más próxima.
* Si un día está en varios ficheros, prevalece el **consolidado** sobre el
  intermedio y, a igualdad, la ingesta más reciente (DEC-31).

:func:`procesar` devuelve además ``origen_fwi``, ``lat_celda``, ``lon_celda`` y
``dist_celda_km`` para los informes (``asignacion_celdas_fwi.csv`` e
``informe_calidad.json``); a ``indicador_fwi.parquet`` sólo pasan las columnas
del modelo de datos.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest import config

from . import fuentes
from .geo import mas_cercano

log = logging.getLogger("piriscope.silver.fwi")

#: Variable del NetCDF de CEMS -> columna Silver (Anexo A).
VARIABLES_NC: dict[str, str] = {
    "fwinx": "fwi",
    "ffmcode": "ffmc",
    "dufmcode": "dmc",
    "drtcode": "dc",
    "infsinx": "isi",
    "fbupinx": "bui",
}
VARIABLES = list(VARIABLES_NC.values())
_PATRON = re.compile(r"fwi_(\d{4})_(\d{2})(_intermedio)?\.(nc|grib)$")


def inventario(bronze_dir: Path | None = None, hf_dir: Path | None = None) -> pd.DataFrame:
    """Ficheros de FWI de Bronze con su mes, tipo (consolidado/intermedio) e ingesta."""
    filas = []
    for timestamp, ruta in fuentes.ficheros("effis", "fwi_*", bronze_dir, hf_dir):
        m = _PATRON.search(ruta.name)
        if m:
            filas.append({"anio": int(m.group(1)), "mes": int(m.group(2)),
                          "origen_fwi": "intermedio" if m.group(3) else "consolidado",
                          "timestamp_ingesta": timestamp, "ruta": ruta})
    return pd.DataFrame(filas, columns=["anio", "mes", "origen_fwi", "timestamp_ingesta", "ruta"])


def abrir(ruta: Path):
    """Abre un fichero de FWI como Dataset con lat/lon en grados (-180–180) recortado a Galicia."""
    import xarray as xr

    ds = xr.open_dataset(ruta, engine="cfgrib" if ruta.suffix == ".grib" else None)
    renombrar = {}
    for viejo, nuevo in (("time", "valid_time"), ("lat", "latitude"), ("lon", "longitude")):
        if viejo in ds.dims and nuevo not in ds.dims:
            renombrar[viejo] = nuevo
    ds = ds.rename(renombrar)
    ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180).sortby(["latitude", "longitude"])
    norte, oeste, sur, este = config.BBOX_GALICIA
    ds = ds.sel(latitude=slice(sur, norte), longitude=slice(oeste, este))
    return ds[[v for v in VARIABLES_NC if v in ds.data_vars]].rename(
        {k: v for k, v in VARIABLES_NC.items() if k in ds.data_vars})


def celdas_municipios(ds, municipios: pd.DataFrame) -> pd.DataFrame:
    """Celda de tierra más cercana al centroide de cada municipio."""
    con_dato = ds["fwi"].notnull().any(dim="valid_time").values  # (lat, lon)
    lat2d, lon2d = np.meshgrid(ds.latitude.values, ds.longitude.values, indexing="ij")
    lat_c, lon_c = lat2d[con_dato], lon2d[con_dato]
    if lat_c.size == 0:
        raise ValueError("El fichero de FWI no tiene ninguna celda con dato en Galicia")
    idx, dist = mas_cercano(municipios["lat_centroide"].to_numpy(), municipios["lon_centroide"].to_numpy(),
                            lat_c, lon_c)
    return pd.DataFrame({
        "id_municipio": municipios["id_municipio"].to_numpy(),
        "lat_celda": lat_c[idx], "lon_celda": lon_c[idx], "dist_celda_km": np.round(dist, 3),
    })


def muestrear(ds, celdas: pd.DataFrame) -> pd.DataFrame:
    """Serie diaria de FWI y componentes en la celda de cada municipio."""
    import xarray as xr

    lat = xr.DataArray(celdas["lat_celda"].to_numpy(), dims="m")
    lon = xr.DataArray(celdas["lon_celda"].to_numpy(), dims="m")
    puntos = ds.sel(latitude=lat, longitude=lon)
    df = puntos.to_dataframe().reset_index()
    df["id_municipio"] = celdas["id_municipio"].to_numpy()[df["m"].to_numpy()]
    df["fecha"] = pd.to_datetime(df["valid_time"]).dt.normalize()
    return df[["id_municipio", "fecha", *[v for v in VARIABLES if v in df.columns]]]


def procesar(municipios: pd.DataFrame, desde: date | None = None, hasta: date | None = None,
             bronze_dir: Path | None = None, hf_dir: Path | None = None) -> pd.DataFrame:
    inv = inventario(bronze_dir, hf_dir)
    if inv.empty:
        raise FileNotFoundError("No hay ficheros de FWI en Bronze (fwi_AAAA_MM.nc)")
    if desde:
        inv = inv[(inv["anio"] * 100 + inv["mes"]) >= desde.year * 100 + desde.month]
    if hasta:
        inv = inv[(inv["anio"] * 100 + inv["mes"]) <= hasta.year * 100 + hasta.month]
    log.info("FWI: %d ficheros (%d intermedios)", len(inv), int((inv["origen_fwi"] == "intermedio").sum()))

    lotes = []
    for fila in inv.itertuples():
        ds = abrir(fila.ruta)
        try:
            celdas = celdas_municipios(ds, municipios)
            df = muestrear(ds, celdas)
        finally:
            ds.close()
        df = df.merge(celdas, on="id_municipio", how="left")
        df["origen_fwi"] = fila.origen_fwi
        df["timestamp_ingesta"] = fila.timestamp_ingesta
        lotes.append(df)
    if not lotes:
        return pd.DataFrame(columns=["id_indicador_fwi", "id_municipio", "fecha", *VARIABLES])

    df = pd.concat(lotes, ignore_index=True)
    df["_prioridad"] = (df["origen_fwi"] == "consolidado").astype(int)
    df = (df.sort_values(["id_municipio", "fecha", "_prioridad", "timestamp_ingesta"])
            .drop_duplicates(["id_municipio", "fecha"], keep="last")
            .drop(columns=["_prioridad", "timestamp_ingesta"]))
    if desde:
        df = df[df["fecha"] >= pd.Timestamp(desde)]
    if hasta:
        df = df[df["fecha"] <= pd.Timestamp(hasta)]
    df.insert(0, "id_indicador_fwi", df["id_municipio"] + "_" + df["fecha"].dt.strftime("%Y-%m-%d"))
    for col in ("id_indicador_fwi", "id_municipio", "origen_fwi"):
        df[col] = df[col].astype("string")
    for col in VARIABLES:
        df[col] = df[col].astype("Float64").round(4)
    columnas = ["id_indicador_fwi", "id_municipio", "fecha", *VARIABLES, "origen_fwi",
                "lat_celda", "lon_celda", "dist_celda_km"]
    return df[columnas].reset_index(drop=True)


def meses_ausentes(inv: pd.DataFrame, desde: date, hasta: date) -> list[str]:
    presentes = {(a, m) for a, m in zip(inv["anio"], inv["mes"])}
    meses = pd.period_range(pd.Timestamp(desde), pd.Timestamp(hasta), freq="M")
    return [str(p) for p in meses if (p.year, p.month) not in presentes]
