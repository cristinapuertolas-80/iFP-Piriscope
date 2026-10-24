"""Entidad Silver **Municipio**: catálogo vigente de los 313 municipios gallegos.

* Códigos y relación oficial: diccionario de municipios del INE
  (``ine/diccionarioAA.xlsx`` de la ingesta INE/CNIG).
* Nombre, límites y centroide: recintos municipales del CNIG (BDDAE,
  ``cnig/*.gpkg``). El centroide se calcula en ETRS89 / UTM 29N (EPSG:25829)
  para no deformarlo y se expresa en grados decimales (DEC-09).
* ``reservado`` se lee de ``docs/municipios_reservados.csv`` (lo fija ML,
  HU-06); mientras no exista, queda nulo.

Devuelve el municipio con ``id_provincia`` y ``nombre_ine`` (los usa el cruce de
los partes de EGIF) y los límites en EPSG:4326, que sólo se usan en memoria
para validar las coordenadas de los partes. A ``municipio.parquet`` sólo pasan
las columnas del modelo de datos (``almacen.COLUMNAS``).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.ingest import config

from . import fuentes

log = logging.getLogger("piriscope.silver.municipios")

FICHERO_RESERVADOS = config.REPO_ROOT / "docs" / "municipios_reservados.csv"
CAPA_RECINTOS = "recintos_municipales_inspire_peninbal_etrs89"
CRS_METRICO = "EPSG:25829"
CRS_GEOGRAFICO = "EPSG:4326"

_ARTICULOS = ("O", "A", "OS", "AS", "EL", "LA", "LOS", "LAS")


def normalizar_nombre(nombre: object) -> str:
    """Clave de comparación de nombres de municipio.

    Quita tildes, mayúsculas y signos, y antepone el artículo pospuesto, de modo
    que ``'Cañiza, A'``, ``'A Cañiza'`` y ``'CAÑIZA, A'`` dan ``'A CANIZA'``.
    """
    if nombre is None or (isinstance(nombre, float) and pd.isna(nombre)) or nombre is pd.NA:
        return ""
    texto = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode().upper().strip()
    m = re.match(rf"^(.*?)\s*,\s*({'|'.join(_ARTICULOS)})$", texto)
    if m:
        texto = f"{m.group(2)} {m.group(1)}"
    return re.sub(r"[^A-Z0-9]+", " ", texto).strip()


def leer_diccionario_ine(ruta: Path) -> pd.DataFrame:
    """``id_municipio``, ``id_provincia`` y ``nombre_ine`` de los municipios gallegos."""
    crudo = pd.read_excel(ruta, header=None, dtype=str)
    fila = next(i for i in range(min(10, len(crudo)))
                if {"CPRO", "CMUN"} <= {str(c).strip().upper() for c in crudo.iloc[i]})
    df = crudo.iloc[fila + 1:].copy()
    df.columns = [str(c).strip().upper() for c in crudo.iloc[fila]]
    df = df[df["CPRO"].isin(config.PROVINCIAS_GALICIA)]
    return pd.DataFrame({
        "id_municipio": (df["CPRO"].str.zfill(2) + df["CMUN"].str.zfill(3)).astype("string"),
        "id_provincia": df["CPRO"].str.zfill(2).astype("string"),
        "nombre_ine": df["NOMBRE"].str.strip().astype("string"),
    }).reset_index(drop=True)


def leer_recintos_cnig(ruta: Path):
    """Recintos municipales gallegos del CNIG con su código INE."""
    import geopandas as gpd

    return filtrar_recintos(gpd.read_file(ruta, layer=CAPA_RECINTOS))


def filtrar_recintos(recintos):
    """Recintos de Galicia con ``id_municipio`` a partir del NATCODE del CNIG."""
    recintos = recintos.copy()
    # NATCODE = 34 (país) + CCAA (2) + provincia (2) + código INE (5)
    recintos["id_municipio"] = recintos["NATCODE"].str[6:11].astype("string")
    recintos = recintos[recintos["NATCODE"].str[4:6].isin(config.PROVINCIAS_GALICIA)]
    return recintos[["id_municipio", "NAMEUNIT", "geometry"]].rename(columns={"NAMEUNIT": "nombre"})


def leer_reservados(ruta: Path = FICHERO_RESERVADOS) -> pd.DataFrame | None:
    if not Path(ruta).exists():
        return None
    df = pd.read_csv(ruta, dtype={"id_municipio": str})
    df["id_municipio"] = df["id_municipio"].str.zfill(5)
    if "reservado" not in df.columns:  # lista simple de municipios reservados
        df["reservado"] = 1
    return df[["id_municipio", "reservado"]]


def construir(ine: pd.DataFrame, recintos, reservados: pd.DataFrame | None = None):
    """Devuelve (municipio, limites) a partir del catálogo del INE y los recintos del CNIG."""
    import geopandas as gpd

    faltan_cnig = sorted(set(ine["id_municipio"]) - set(recintos["id_municipio"]))
    sobran_cnig = sorted(set(recintos["id_municipio"]) - set(ine["id_municipio"]))
    if faltan_cnig:
        log.warning("Municipios del INE sin recinto en el CNIG: %s", faltan_cnig)
    if sobran_cnig:
        log.warning("Recintos del CNIG que no están en el catálogo vigente del INE: %s", sobran_cnig)

    limites = gpd.GeoDataFrame(
        ine.merge(recintos, on="id_municipio", how="inner"), geometry="geometry", crs=recintos.crs
    ).dissolve(by="id_municipio", as_index=False, aggfunc="first")
    centroides = limites.to_crs(CRS_METRICO).centroid.to_crs(CRS_GEOGRAFICO)

    municipio = pd.DataFrame({
        "id_municipio": limites["id_municipio"].astype("string"),
        "nombre": limites["nombre"].fillna(limites["nombre_ine"]).astype("string"),
        "nombre_ine": limites["nombre_ine"].astype("string"),
        "id_provincia": limites["id_provincia"].astype("string"),
        "provincia": limites["id_provincia"].map(config.PROVINCIAS_GALICIA).astype("string"),
        "lat_centroide": centroides.y.round(6).to_numpy(),
        "lon_centroide": centroides.x.round(6).to_numpy(),
    })
    if reservados is not None:
        municipio = municipio.merge(reservados, on="id_municipio", how="left")
        municipio["reservado"] = municipio["reservado"].fillna(0).astype("Int8")
    else:
        municipio["reservado"] = pd.array([pd.NA] * len(municipio), dtype="Int8")
    municipio = municipio.sort_values("id_municipio").reset_index(drop=True)

    limites = limites[["id_municipio", "nombre", "geometry"]].to_crs(CRS_GEOGRAFICO)
    return municipio, limites


def procesar(bronze_dir: Path | None = None, hf_dir: Path | None = None):
    """Lee la última ingesta INE/CNIG y construye (municipio, limites)."""
    ruta_ine = fuentes.exigir(fuentes.ultimo_fichero("ine_cnig", "ine/diccionario*.xlsx", bronze_dir, hf_dir),
                              "ine_cnig", "el diccionario de municipios del INE")
    ruta_cnig = fuentes.exigir(fuentes.ultimo_fichero("ine_cnig", "cnig/*.gpkg", bronze_dir, hf_dir),
                               "ine_cnig", "los recintos municipales del CNIG")
    log.info("Municipios: INE %s, CNIG %s", ruta_ine.name, ruta_cnig.name)
    reservados = leer_reservados()
    if reservados is None:
        log.info("Sin docs/municipios_reservados.csv: 'reservado' queda nulo hasta que ML lo fije")
    return construir(leer_diccionario_ine(ruta_ine), leer_recintos_cnig(ruta_cnig), reservados)
