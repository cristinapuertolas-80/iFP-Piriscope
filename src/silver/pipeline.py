"""Orquestación de la capa Silver: Bronze -> ``data/silver``.

Orden: municipios (necesarios para el resto) -> meteorología AEMET -> FWI ->
incendios EGIF. Con ``desde``/``hasta`` sólo se reprocesa ese periodo de las
tablas con fecha; ``municipio`` y ``estacion_aemet`` se reescriben siempre.

El municipio y sus límites se construyen siempre desde Bronze en memoria (el
cruce necesita ``id_provincia``, ``nombre_ine`` y los polígonos, que no forman
parte de la tabla Silver); sólo se escribe ``municipio.parquet`` si se pide la
tabla ``municipios``.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from . import almacen, calidad, fwi, incendios, meteo, municipios

log = logging.getLogger("piriscope.silver")

TABLAS = ("municipios", "meteo", "fwi", "incendios")


def _rango(df: pd.DataFrame, col: str) -> dict[str, str | None]:
    if df.empty:
        return {"desde": None, "hasta": None}
    return {"desde": str(pd.Timestamp(df[col].min()).date()), "hasta": str(pd.Timestamp(df[col].max()).date())}


def etapa_municipios(municipio: pd.DataFrame, informe: calidad.Informe, base_dir: Path | None) -> None:
    almacen.guardar(municipio, "municipio", ordenar_por=["id_municipio"], base_dir=base_dir)
    informe.tabla("municipio", filas=len(municipio),
                  duplicados=calidad.duplicados(municipio, "id_municipio"),
                  por_provincia=municipio["provincia"].value_counts().sort_index().to_dict(),
                  reservado_definido=bool(municipio["reservado"].notna().all()))


def etapa_meteo(municipio: pd.DataFrame, informe: calidad.Informe, desde: date | None,
                hasta: date | None, base_dir: Path | None, **kw) -> pd.DataFrame:
    estaciones = meteo.leer_estaciones(**kw)
    almacen.guardar(estaciones, "estacion_aemet", ordenar_por=["id_estacion_aemet"], base_dir=base_dir)

    obs = meteo.leer_diarios(desde=desde, hasta=hasta, **kw)
    # Clave estación_fecha para que valores_fuera_rango.csv identifique el día
    obs["_clave"] = obs["id_estacion_aemet"] + "_" + obs["fecha"].dt.strftime("%Y-%m-%d")
    obs, anomalias = calidad.aplicar_rangos(obs, "observacion_estacion", "_clave", "fecha")
    obs = obs.drop(columns="_clave")
    informe.anomalias(anomalias, "observacion_estacion", desde, hasta)

    asignacion = meteo.asignar_estaciones(municipio, estaciones, meteo.estaciones_con_datos(**kw))
    informe.csv("asignacion_estaciones.csv", asignacion.merge(
        municipio[["id_municipio", "nombre", "provincia"]], on="id_municipio").merge(
        estaciones[["id_estacion_aemet", "nombre"]].rename(columns={"nombre": "nombre_estacion"}),
        on="id_estacion_aemet", how="left"))

    obs_mun = meteo.observacion_meteorologica(municipio, asignacion, obs, desde, hasta)
    almacen.guardar(obs_mun, "observacion_meteorologica", "fecha", desde, hasta,
                    ordenar_por=["id_municipio", "fecha"], base_dir=base_dir)
    # Los informes de cobertura cubren toda la tabla, no sólo el periodo reprocesado
    obs_tabla = almacen.leer("observacion_meteorologica", base_dir)
    informe.csv("cobertura_meteo_municipios.csv", meteo.cobertura_municipios(obs_tabla))
    informe.csv("cobertura_meteo_municipios_anual.csv", meteo.cobertura_municipios_anual(obs_tabla))
    informe.tabla("estacion_aemet", filas=len(estaciones),
                  duplicados=calidad.duplicados(estaciones, "id_estacion_aemet"),
                  estaciones_asignadas=int(asignacion["id_estacion_aemet"].nunique()))
    informe.tabla("observacion_meteorologica", filas_periodo=len(obs_mun), **_rango(obs_mun, "fecha"),
                  duplicados=calidad.duplicados(obs_mun, "id_obs_meteo"),
                  dist_estacion_km={"media": round(float(asignacion["dist_estacion_km"].mean()), 2),
                                    "max": round(float(asignacion["dist_estacion_km"].max()), 2)},
                  nulos_pct=calidad.porcentaje_nulos(obs_mun, meteo.VARIABLES),
                  valores_fuera_rango=len(anomalias))
    return obs_mun


def etapa_fwi(municipio: pd.DataFrame, informe: calidad.Informe, desde: date | None,
              hasta: date | None, base_dir: Path | None, **kw) -> pd.DataFrame:
    df = fwi.procesar(municipio, desde, hasta, **kw)
    df, anomalias = calidad.aplicar_rangos(df, "indicador_fwi", "id_indicador_fwi", "fecha")
    informe.anomalias(anomalias, "indicador_fwi", desde, hasta)
    almacen.guardar(df, "indicador_fwi", "fecha", desde, hasta,
                    ordenar_por=["id_municipio", "fecha"], base_dir=base_dir)
    informe.csv("asignacion_celdas_fwi.csv", df.drop_duplicates("id_municipio")[
        ["id_municipio", "lat_celda", "lon_celda", "dist_celda_km"]].merge(
        municipio[["id_municipio", "nombre", "provincia", "lat_centroide", "lon_centroide"]], on="id_municipio"))
    rango = _rango(df, "fecha")
    ausentes = (fwi.meses_ausentes(fwi.inventario(**kw), date.fromisoformat(rango["desde"]),
                                   date.fromisoformat(rango["hasta"])) if rango["desde"] else [])
    intermedio = df.loc[df["origen_fwi"] == "intermedio", "fecha"]
    informe.tabla("indicador_fwi", filas_periodo=len(df), **rango,
                  duplicados=calidad.duplicados(df, "id_indicador_fwi"),
                  dias_intermedio=int(intermedio.nunique()),
                  meses_intermedio=sorted(intermedio.dt.strftime("%Y-%m").unique().tolist()),
                  meses_sin_fichero=ausentes,
                  dist_celda_km_max=round(float(df["dist_celda_km"].max()), 2) if len(df) else None,
                  nulos_pct=calidad.porcentaje_nulos(df, fwi.VARIABLES),
                  valores_fuera_rango=len(anomalias))
    return df


def etapa_incendios(municipio: pd.DataFrame, limites, informe: calidad.Informe, desde: date | None,
                    hasta: date | None, base_dir: Path | None, **kw) -> pd.DataFrame:
    # EGIF es un único fichero: se procesa entero para que los informes de HU-04 cubran
    # todos los partes, y ``guardar`` sólo sustituye el periodo pedido.
    df = incendios.procesar(municipio, limites, **kw)
    df, anomalias = calidad.aplicar_rangos(df, "evento_incendio", "id_evento", "fecha_deteccion")
    informe.anomalias(anomalias, "evento_incendio")  # EGIF se procesa entero
    almacen.guardar(df, "evento_incendio", "fecha_deteccion", desde, hasta,
                    ordenar_por=["fecha_deteccion", "id_evento"], base_dir=base_dir)
    informe.csv("incendios_no_asignados.csv", incendios.informe_no_asignados(df))
    informe.csv("incendios_asignacion_por_campana.csv", incendios.resumen_asignacion(df))
    informe.csv("incendios_asignacion_a_revisar.csv", incendios.asignacion_a_revisar(df, municipio))
    metodos = df["metodo_asignacion"].value_counts().to_dict()
    informe.tabla("evento_incendio", filas_periodo=len(df), **_rango(df, "fecha_deteccion"),
                  duplicados_id_evento=calidad.duplicados(df, "id_evento"),
                  pct_con_codigo_ine=round(100 * float(df["id_municipio"].notna().mean()), 2) if len(df) else None,
                  metodos_asignacion=metodos,
                  asignados_otra_provincia=int(df["asignado_otra_provincia"].sum()),
                  sin_fecha_deteccion=int(df["fecha_deteccion"].isna().sum()),
                  valores_fuera_rango=len(anomalias))
    return df


def ejecutar(desde: date | None = None, hasta: date | None = None, tablas: list[str] | None = None,
             base_dir: Path | None = None, bronze_dir: Path | None = None,
             hf_dir: Path | None = None) -> calidad.Informe:
    """Genera las tablas Silver pedidas (por defecto, todas)."""
    tablas = list(tablas or TABLAS)
    kw = {"bronze_dir": bronze_dir, "hf_dir": hf_dir}
    informe = calidad.Informe(base_dir)
    municipio, limites = municipios.procesar(**kw)
    if "municipios" in tablas:
        log.info("=== Silver: municipios ===")
        etapa_municipios(municipio, informe, base_dir)
    if "meteo" in tablas:
        log.info("=== Silver: meteorología AEMET ===")
        etapa_meteo(municipio, informe, desde, hasta, base_dir, **kw)
    if "fwi" in tablas:
        log.info("=== Silver: FWI ===")
        etapa_fwi(municipio, informe, desde, hasta, base_dir, **kw)
    if "incendios" in tablas:
        log.info("=== Silver: incendios EGIF ===")
        etapa_incendios(municipio, limites, informe, desde, hasta, base_dir, **kw)
    ruta = informe.escribir()
    log.info("Informe de calidad: %s", ruta)
    return informe
