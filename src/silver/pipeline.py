"""Orquestación de la capa Silver: Bronze -> ``data/silver``.

Orden: municipios (necesarios para el resto) -> meteorología AEMET -> FWI ->
incendios EGIF. Con ``desde``/``hasta`` sólo se reprocesa ese periodo de las
tablas con fecha; ``municipio`` y ``estacion_aemet`` se reescriben siempre.
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


def etapa_municipios(informe: calidad.Informe, base_dir: Path | None, **kw):
    municipio, limites = municipios.procesar(**kw)
    almacen.guardar(municipio, "municipio", ordenar_por=["id_municipio"], base_dir=base_dir)
    municipios.guardar_limites(limites, base_dir)
    informe.tabla("municipio", filas=len(municipio),
                  duplicados=calidad.duplicados(municipio, "id_municipio"),
                  por_provincia=municipio["provincia"].value_counts().sort_index().to_dict(),
                  reservado_definido=bool(municipio["reservado"].notna().all()))
    return municipio, limites


def cargar_municipios(base_dir: Path | None):
    import geopandas as gpd

    ruta = almacen.ruta_tabla("municipio", base_dir)
    limites = almacen.silver_dir(base_dir) / "municipio_limites.parquet"
    if not ruta.exists() or not limites.exists():
        raise FileNotFoundError("Falta data/silver/municipio.parquet: ejecuta antes la tabla 'municipios'")
    return almacen.leer("municipio", base_dir), gpd.read_parquet(limites)


def etapa_meteo(municipio: pd.DataFrame, informe: calidad.Informe, desde: date | None,
                hasta: date | None, base_dir: Path | None, **kw) -> pd.DataFrame:
    estaciones = meteo.leer_estaciones(**kw)
    almacen.guardar(estaciones, "estacion_aemet", ordenar_por=["id_estacion_aemet"], base_dir=base_dir)

    obs = meteo.leer_diarios(desde=desde, hasta=hasta, **kw)
    obs, anomalias = calidad.aplicar_rangos(obs, "observacion_estacion", "id_estacion_aemet")
    informe.anomalias(anomalias)

    asignacion = meteo.asignar_estaciones(municipio, estaciones, meteo.estaciones_con_datos(**kw))
    informe.csv("asignacion_estaciones.csv", asignacion.merge(
        municipio[["id_municipio", "nombre", "provincia"]], on="id_municipio").merge(
        estaciones[["id_estacion_aemet", "nombre"]].rename(columns={"nombre": "nombre_estacion"}),
        on="id_estacion_aemet", how="left"))

    obs_mun = meteo.observacion_meteorologica(municipio, asignacion, obs, desde, hasta)
    almacen.guardar(obs_mun, "observacion_meteorologica", "fecha", desde, hasta,
                    ordenar_por=["id_municipio", "fecha"], base_dir=base_dir)
    cobertura = meteo.cobertura_municipios(obs_mun)
    informe.csv("cobertura_meteo_municipios.csv", cobertura)
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
    df, anomalias = calidad.aplicar_rangos(df, "indicador_fwi", "id_indicador_fwi")
    informe.anomalias(anomalias)
    almacen.guardar(df, "indicador_fwi", "fecha", desde, hasta,
                    ordenar_por=["id_municipio", "fecha"], base_dir=base_dir)
    rango = _rango(df, "fecha")
    ausentes = (fwi.meses_ausentes(fwi.inventario(**kw), date.fromisoformat(rango["desde"]),
                                   date.fromisoformat(rango["hasta"])) if rango["desde"] else [])
    informe.tabla("indicador_fwi", filas_periodo=len(df), **rango,
                  duplicados=calidad.duplicados(df, "id_indicador_fwi"),
                  dias_intermedio=int(df.loc[df["origen_fwi"] == "intermedio", "fecha"].nunique()),
                  meses_sin_fichero=ausentes,
                  dist_celda_km_max=round(float(df["dist_celda_km"].max()), 2) if len(df) else None,
                  nulos_pct=calidad.porcentaje_nulos(df, fwi.VARIABLES),
                  valores_fuera_rango=len(anomalias))
    return df


def etapa_incendios(municipio: pd.DataFrame, limites, informe: calidad.Informe, desde: date | None,
                    hasta: date | None, base_dir: Path | None, **kw) -> pd.DataFrame:
    df = incendios.procesar(municipio, limites, desde, hasta, **kw)
    df, anomalias = calidad.aplicar_rangos(df, "evento_incendio", "id_evento")
    informe.anomalias(anomalias)
    almacen.guardar(df, "evento_incendio", "fecha_deteccion", desde, hasta,
                    ordenar_por=["fecha_deteccion", "id_evento"], base_dir=base_dir)
    df = almacen.leer("evento_incendio", base_dir)  # los informes de HU-04 cubren toda la tabla
    informe.csv("incendios_no_asignados.csv", incendios.informe_no_asignados(df))
    informe.csv("incendios_muestra_validacion.csv", incendios.muestra_validacion(df, municipio))
    informe.csv("incendios_asignacion_por_campana.csv", incendios.resumen_asignacion(df))
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
    if "municipios" in tablas:
        log.info("=== Silver: municipios ===")
        municipio, limites = etapa_municipios(informe, base_dir, **kw)
    else:
        municipio, limites = cargar_municipios(base_dir)
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
