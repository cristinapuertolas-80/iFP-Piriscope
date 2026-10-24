"""Entidades Silver **EstacionAemet** y **ObservacionMeteorologica** (HU-01, HU-04).

1. Se leen todos los JSON de valores diarios de AEMET de Bronze. Si un mismo
   día de una estación aparece en varias ingestas (tramos solapados, últimos 10
   días del cron), prevalece la **ingesta más reciente**, porque AEMET publica
   con retraso y corrige valores.
2. Se tipan los valores: coma decimal, ``Ip`` (precipitación inapreciable,
   < 0,1 mm) = 0,0, ``Acum``/``Varias`` = nulo; la dirección del viento pasa de
   decenas de grado a grados (``88``/``99`` = nulo).
3. Cada municipio se asocia a **su estación más cercana** (distancia del
   centroide a la estación, DEC-08), elegida entre las estaciones con datos; la
   asignación es fija para todo el periodo y ``dist_estacion_km`` queda como
   indicador de calidad.
4. ``ObservacionMeteorologica`` tiene una fila por municipio y día del periodo:
   los días en que la estación asignada no tiene dato quedan con nulos (no se
   imputa en Silver).
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import fuentes
from .geo import mas_cercano

log = logging.getLogger("piriscope.silver.meteo")

ENCODINGS = ("ISO-8859-15", "utf-8", "latin-1")

#: Campo de AEMET -> columna Silver. Unidades: °C, %, mm, m/s, grados, horas, hPa.
VARIABLES_AEMET: dict[str, str] = {
    "tmed": "temperatura",
    "tmax": "temperatura_max",
    "tmin": "temperatura_min",
    "hrMedia": "humedad_relativa",
    "hrMax": "humedad_relativa_max",
    "hrMin": "humedad_relativa_min",
    "prec": "precipitacion",
    "velmedia": "viento",
    "racha": "racha_max",
    "dir": "direccion_viento",
    "sol": "insolacion",
    "presMax": "presion_max",
    "presMin": "presion_min",
}
VARIABLES = list(VARIABLES_AEMET.values())


# ---------------------------------------------------------------------------
# Lectura y tipado
# ---------------------------------------------------------------------------
def decodificar(contenido: bytes) -> list[dict]:
    for enc in ENCODINGS:
        try:
            datos = json.loads(contenido.decode(enc))
            return datos if isinstance(datos, list) else []
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return []


def a_numero(serie: pd.Series, inapreciable: float | None = None) -> pd.Series:
    """'12,5' -> 12.5; 'Ip' -> ``inapreciable``; 'Acum', 'Varias', '' -> nulo."""
    texto = serie.astype("string").str.strip().str.replace(",", ".", regex=False)
    if inapreciable is not None:
        texto = texto.mask((texto.str.lower() == "ip").fillna(False), str(inapreciable))
    return pd.to_numeric(texto, errors="coerce").astype("Float64")


def direccion_grados(serie: pd.Series) -> pd.Series:
    """AEMET da la dirección de la racha en decenas de grado; 88 y 99 no son direcciones."""
    decenas = a_numero(serie)
    return (decenas.where((decenas >= 0) & (decenas <= 36)) * 10).astype("Float64")


def tipar_registros(registros: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "id_estacion_aemet": registros["indicativo"].astype("string").str.strip(),
        "fecha": pd.to_datetime(registros["fecha"], errors="coerce").dt.normalize(),
    })
    for campo, col in VARIABLES_AEMET.items():
        serie = registros[campo] if campo in registros.columns else pd.Series(pd.NA, index=registros.index)
        if campo == "dir":
            out[col] = direccion_grados(serie)
        else:
            out[col] = a_numero(serie, inapreciable=0.0 if campo == "prec" else None)
    out["timestamp_ingesta"] = registros["timestamp_ingesta"].astype("string")
    return out


def leer_diarios(bronze_dir: Path | None = None, hf_dir: Path | None = None,
                 desde: date | None = None, hasta: date | None = None) -> pd.DataFrame:
    """Observaciones diarias por estación, tipadas y sin duplicados (estación, fecha)."""
    lotes = []
    for timestamp, ruta in fuentes.ficheros("aemet", "diarios/*.json", bronze_dir, hf_dir):
        if desde or hasta:  # el nombre del tramo es <estacion>_<ini>_<fin>.json
            partes = ruta.stem.rsplit("_", 2)
            if len(partes) == 3:
                ini, fin = partes[1], partes[2]
                if (hasta and ini > hasta.isoformat()) or (desde and fin < desde.isoformat()):
                    continue
        registros = decodificar(ruta.read_bytes())
        if registros:
            df = pd.DataFrame(registros)
            df["timestamp_ingesta"] = timestamp
            lotes.append(df)
    if not lotes:
        raise FileNotFoundError("No hay valores diarios de AEMET en Bronze (diarios/*.json)")
    obs = tipar_registros(pd.concat(lotes, ignore_index=True))
    obs = obs.dropna(subset=["id_estacion_aemet", "fecha"])
    n = len(obs)
    obs = (obs.sort_values(["id_estacion_aemet", "fecha", "timestamp_ingesta"])
              .drop_duplicates(["id_estacion_aemet", "fecha"], keep="last"))
    log.info("AEMET: %d registros leídos, %d tras quitar repetidos entre ingestas", n, len(obs))
    if desde:
        obs = obs[obs["fecha"] >= pd.Timestamp(desde)]
    if hasta:
        obs = obs[obs["fecha"] <= pd.Timestamp(hasta)]
    return obs.reset_index(drop=True)


def leer_estaciones(bronze_dir: Path | None = None, hf_dir: Path | None = None) -> pd.DataFrame:
    """Entidad EstacionAemet: unión de ``estaciones_galicia.csv`` de todas las ingestas
    (la más reciente prevalece si una estación cambia de datos)."""
    lotes = []
    for timestamp, ruta in fuentes.ficheros("aemet", "estaciones_galicia.csv", bronze_dir, hf_dir):
        df = pd.read_csv(ruta, dtype=str)
        if df.empty:
            continue
        df["timestamp_ingesta"] = timestamp
        lotes.append(df)
    if not lotes:
        raise FileNotFoundError("No hay estaciones_galicia.csv de AEMET en Bronze")
    df = (pd.concat(lotes, ignore_index=True).sort_values("timestamp_ingesta")
            .drop_duplicates("indicativo", keep="last"))
    return pd.DataFrame({
        "id_estacion_aemet": df["indicativo"].astype("string").str.strip(),
        "nombre": df["nombre"].astype("string").str.strip(),
        "provincia": df["provincia"].astype("string").str.strip(),
        "latitud": pd.to_numeric(df["latitud_dec"], errors="coerce"),
        "longitud": pd.to_numeric(df["longitud_dec"], errors="coerce"),
        "altitud": a_numero(df["altitud"]),
    }).sort_values("id_estacion_aemet").reset_index(drop=True)


def estaciones_con_datos(bronze_dir: Path | None = None, hf_dir: Path | None = None) -> set[str]:
    """Estaciones con al menos un día de datos en alguna ingesta (``cobertura_estaciones.csv``).

    No depende del rango que se reprocese, así que la asignación
    municipio-estación es la misma en todas las ejecuciones.
    """
    con_datos: set[str] = set()
    for _, ruta in fuentes.ficheros("aemet", "cobertura_estaciones.csv", bronze_dir, hf_dir):
        df = pd.read_csv(ruta, dtype={"indicativo": str})
        if {"indicativo", "dias_con_datos"} <= set(df.columns):
            con_datos |= set(df.loc[df["dias_con_datos"] > 0, "indicativo"].str.strip())
    return con_datos


# ---------------------------------------------------------------------------
# Cruce espacial y temporal
# ---------------------------------------------------------------------------
def asignar_estaciones(municipios: pd.DataFrame, estaciones: pd.DataFrame,
                       con_datos: set[str] | None = None) -> pd.DataFrame:
    """Estación AEMET más cercana al centroide de cada municipio (DEC-08)."""
    candidatas = estaciones.dropna(subset=["latitud", "longitud"])
    if con_datos is not None:
        candidatas = candidatas[candidatas["id_estacion_aemet"].isin(con_datos)]
    if candidatas.empty:
        raise ValueError("No hay estaciones AEMET con coordenadas y datos para asignar")
    idx, dist = mas_cercano(municipios["lat_centroide"].to_numpy(), municipios["lon_centroide"].to_numpy(),
                            candidatas["latitud"].to_numpy(), candidatas["longitud"].to_numpy())
    return pd.DataFrame({
        "id_municipio": municipios["id_municipio"].to_numpy(),
        "id_estacion_aemet": candidatas["id_estacion_aemet"].to_numpy()[idx],
        "dist_estacion_km": np.round(dist, 3),
    })


def observacion_meteorologica(municipios: pd.DataFrame, asignacion: pd.DataFrame,
                              obs: pd.DataFrame, desde: date | None = None,
                              hasta: date | None = None) -> pd.DataFrame:
    """Una fila por municipio y día de ``[desde, hasta]`` con la meteo de su estación."""
    ini = pd.Timestamp(desde) if desde else obs["fecha"].min()
    fin = pd.Timestamp(hasta) if hasta else obs["fecha"].max()
    fechas = pd.date_range(ini, fin, freq="D")
    rejilla = pd.MultiIndex.from_product([municipios["id_municipio"], fechas],
                                         names=["id_municipio", "fecha"]).to_frame(index=False)
    rejilla = rejilla.merge(asignacion, on="id_municipio", how="left")
    df = rejilla.merge(obs[["id_estacion_aemet", "fecha", *VARIABLES]],
                       on=["id_estacion_aemet", "fecha"], how="left")
    df.insert(0, "id_obs_meteo", df["id_municipio"] + "_" + df["fecha"].dt.strftime("%Y-%m-%d"))
    df["id_municipio"] = df["id_municipio"].astype("string")
    df["id_obs_meteo"] = df["id_obs_meteo"].astype("string")
    df["id_estacion_aemet"] = df["id_estacion_aemet"].astype("string")
    return df


def cobertura_municipios(meteo: pd.DataFrame) -> pd.DataFrame:
    """% de días con temperatura y precipitación por municipio (informe de cobertura)."""
    g = meteo.groupby("id_municipio")
    return pd.DataFrame({
        "id_estacion_aemet": g["id_estacion_aemet"].first(),
        "dist_estacion_km": g["dist_estacion_km"].first(),
        "dias": g.size(),
        "cobertura_temperatura_pct": (100 * g["temperatura"].apply(lambda s: s.notna().mean())).round(2),
        "cobertura_precipitacion_pct": (100 * g["precipitacion"].apply(lambda s: s.notna().mean())).round(2),
        "cobertura_humedad_pct": (100 * g["humedad_relativa"].apply(lambda s: s.notna().mean())).round(2),
        "cobertura_viento_pct": (100 * g["viento"].apply(lambda s: s.notna().mean())).round(2),
    }).reset_index()
