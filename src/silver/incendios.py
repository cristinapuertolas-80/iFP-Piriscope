"""Entidad Silver **EventoIncendio**: partes de EGIF tipados y con código INE (HU-02, HU-04).

Parte del ``egif_galicia_unificado.parquet`` de la última ingesta de EGIF
(todo texto) y:

1. Resuelve los nombres de columna de la exportación (``detectado``,
   ``extinguido``, ``coordenadax``…) a los del Anexo A y conserva sólo los
   campos necesarios (minimización, DEC-27).
2. Tipa fechas (día/mes/año hora), números con coma decimal, huso y datum, y
   limpia los textos (``causa`` queda como ``'[400] Intencionado'``, formato de EGIF
   sin espacios dobles).
3. Asigna el **código INE vigente** (DEC-07):

   a. por nombre de municipio + provincia del parte, comparado con el catálogo
      vigente (``municipio.parquet``) y con ``docs/tabla_equivalencias.csv``
      (fusiones y variantes de nombre);
   b. si no casa en su provincia, por nombre en el resto de Galicia (partes que
      una provincia registra en un municipio de otra), si el nombre es único;
   c. se **valida con las coordenadas UTM**: el punto se proyecta a WGS84
      (datum ETRS89, o ED50 si el parte no lo indica) y se cruza con los
      recintos municipales;
   d. si el nombre no casa pero las coordenadas caen en Galicia, se usa el
      municipio de las coordenadas.

   ``metodo_asignacion`` deja constancia del camino seguido y
   ``id_municipio_coordenadas`` del municipio que indican las coordenadas.

Esas columnas de la asignación sirven para los informes de HU-04
(``informes/incendios_*.csv``); a ``evento_incendio.parquet`` sólo pasan las
columnas del modelo de datos.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.ingest import config

from . import fuentes
from .municipios import normalizar_nombre

log = logging.getLogger("piriscope.silver.incendios")

FICHERO_EQUIVALENCIAS = config.REPO_ROOT / "docs" / "tabla_equivalencias.csv"

#: Columna del Anexo A -> nombres con que puede llegar del unificado de Bronze.
ALIAS: dict[str, tuple[str, ...]] = {
    "fecha_deteccion": ("fecha_deteccion", "detectado"),
    "fecha_extincion": ("fecha_extincion", "extinguido"),
    "coordenada_x": ("coordenada_x", "coordenadax"),
    "coordenada_y": ("coordenada_y", "coordenaday"),
    "superficie_forestal_ha": ("superficie_forestal_ha", "superficietotalforestal"),
    "municipio": ("municipio",),
    "causa": ("causa",),
    "huso": ("huso",),
    "datum": ("datum",),
    "campana": ("campana",),
    "numero_parte": ("numero_parte",),
    "id_provincia": ("id_provincia",),
    "id_evento": ("id_evento",),
}

#: EPSG de las proyecciones UTM por datum y huso.
EPSG_UTM = {("ETRS89", 29): 25829, ("ETRS89", 30): 25830, ("ED50", 29): 23029, ("ED50", 30): 23030}
DATUM_POR_DEFECTO = "ED50"

METODOS = ("nombre_validado", "nombre_coordenadas_discrepan", "nombre_sin_coordenadas",
           "coordenadas", "sin_asignar")


# ---------------------------------------------------------------------------
# Tipado
# ---------------------------------------------------------------------------
def _texto(serie: pd.Series) -> pd.Series:
    """Texto limpio: sin espacios duros, dobles ni en los bordes; vacíos -> nulo."""
    s = serie.astype("string").str.replace(r"\s+", " ", regex=True).str.strip()
    return s.mask(s.isin(["", "-", "nan", "None"]))


def _numero(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(_texto(serie).str.replace(",", ".", regex=False), errors="coerce").astype("Float64")


def _fecha_hora(serie: pd.Series) -> pd.Series:
    texto = _texto(serie)
    fechas = pd.to_datetime(texto, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    resto = fechas.isna() & texto.notna()
    if resto.any():
        fechas[resto] = pd.to_datetime(texto[resto], dayfirst=True, errors="coerce")
    return fechas


def resolver_columnas(bruto: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=bruto.index)
    for canonica, candidatos in ALIAS.items():
        col = next((c for c in candidatos if c in bruto.columns), None)
        out[canonica] = bruto[col] if col else pd.Series(pd.NA, index=bruto.index, dtype="string")
    return out


def tipar(bruto: pd.DataFrame) -> pd.DataFrame:
    df = resolver_columnas(bruto)
    huso = _numero(df["huso"])
    datum = _texto(df["datum"]).str.upper()
    x, y = _numero(df["coordenada_x"]), _numero(df["coordenada_y"])
    return pd.DataFrame({
        "id_evento": _texto(df["id_evento"]),
        "campana": _numero(df["campana"]).astype("Int16"),
        "numero_parte": _texto(df["numero_parte"]),
        "id_provincia": _texto(df["id_provincia"]),
        "municipio_egif": _texto(df["municipio"]),
        "fecha_deteccion": _fecha_hora(df["fecha_deteccion"]),
        "fecha_extincion": _fecha_hora(df["fecha_extincion"]),
        "causa": _texto(df["causa"]),
        "superficie_forestal_ha": _numero(df["superficie_forestal_ha"]),
        "coordenada_x": x.mask(x <= 0),
        "coordenada_y": y.mask(y <= 0),
        "huso": huso.round().astype("Int8"),
        "datum": datum,
    })


# ---------------------------------------------------------------------------
# Asignación del código INE
# ---------------------------------------------------------------------------
def leer_equivalencias(ruta: Path = FICHERO_EQUIVALENCIAS) -> pd.DataFrame:
    if not Path(ruta).exists():
        return pd.DataFrame(columns=["id_provincia", "municipio_egif", "id_municipio"])
    return pd.read_csv(ruta, dtype=str)


def diccionario_nombres(municipios: pd.DataFrame, equivalencias: pd.DataFrame
                        ) -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """Devuelve ({(provincia, nombre): id}, {nombre único en Galicia: id})."""
    filas = []
    for col in ("nombre", "nombre_ine"):
        if col in municipios.columns:
            filas.append(pd.DataFrame({"id_provincia": municipios["id_provincia"],
                                       "clave": municipios[col].map(normalizar_nombre),
                                       "id_municipio": municipios["id_municipio"]}))
    if len(equivalencias):
        filas.append(pd.DataFrame({"id_provincia": equivalencias["id_provincia"].str.zfill(2),
                                   "clave": equivalencias["municipio_egif"].map(normalizar_nombre),
                                   "id_municipio": equivalencias["id_municipio"].str.zfill(5)}))
    nombres = pd.concat(filas, ignore_index=True).drop_duplicates()
    por_provincia = {(p, c): i for p, c, i in nombres[["id_provincia", "clave", "id_municipio"]].itertuples(index=False)}
    unicos = nombres.drop_duplicates(["clave", "id_municipio"])
    unicos = unicos[~unicos["clave"].duplicated(keep=False)]
    return por_provincia, dict(zip(unicos["clave"], unicos["id_municipio"]))


def asignar_por_nombre(df: pd.DataFrame, municipios: pd.DataFrame,
                       equivalencias: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """(id_municipio por nombre, ¿se ha resuelto fuera de su provincia?)."""
    por_provincia, unicos = diccionario_nombres(municipios, equivalencias)
    claves = df["municipio_egif"].map(normalizar_nombre)
    ids = pd.Series([por_provincia.get((p, c)) for p, c in zip(df["id_provincia"], claves)],
                    index=df.index, dtype="string")
    otra = ids.isna() & claves.map(unicos).notna()
    ids = ids.fillna(claves.map(unicos).astype("string"))
    return ids, otra


def asignar_por_coordenadas(df: pd.DataFrame, limites) -> pd.DataFrame:
    """``id_municipio_coordenadas``, ``latitud`` y ``longitud`` (WGS84) de cada parte."""
    import geopandas as gpd

    out = pd.DataFrame({"id_municipio_coordenadas": pd.Series(pd.NA, index=df.index, dtype="string"),
                        "latitud": np.nan, "longitud": np.nan}, index=df.index)
    datum = df["datum"].fillna(DATUM_POR_DEFECTO).where(df["datum"].isin(["ETRS89", "ED50"]), DATUM_POR_DEFECTO)
    validos = df["coordenada_x"].notna() & df["coordenada_y"].notna() & df["huso"].isin([29, 30])
    for (d, h), grupo in df[validos].groupby([datum[validos], df.loc[validos, "huso"].astype(int)]):
        epsg = EPSG_UTM.get((d, h))
        if epsg is None:
            continue
        puntos = gpd.GeoSeries(gpd.points_from_xy(grupo["coordenada_x"].astype(float),
                                                  grupo["coordenada_y"].astype(float)),
                               index=grupo.index, crs=f"EPSG:{epsg}").to_crs(limites.crs)
        out.loc[grupo.index, "latitud"] = puntos.y.round(6)
        out.loc[grupo.index, "longitud"] = puntos.x.round(6)
        cruce = gpd.sjoin(gpd.GeoDataFrame(geometry=puntos), limites[["id_municipio", "geometry"]],
                          how="left", predicate="within")
        cruce = cruce[~cruce.index.duplicated(keep="first")]
        out.loc[cruce.index, "id_municipio_coordenadas"] = cruce["id_municipio"].astype("string")
    return out


def asignar_municipio(df: pd.DataFrame, municipios: pd.DataFrame, limites,
                      equivalencias: pd.DataFrame | None = None) -> pd.DataFrame:
    equivalencias = leer_equivalencias() if equivalencias is None else equivalencias
    df = df.copy()
    por_nombre, otra_provincia = asignar_por_nombre(df, municipios, equivalencias)
    coords = asignar_por_coordenadas(df, limites)
    df = df.join(coords)

    por_coords = df["id_municipio_coordenadas"]
    metodo = pd.Series("sin_asignar", index=df.index, dtype="string")
    metodo[por_nombre.notna() & por_coords.isna()] = "nombre_sin_coordenadas"
    metodo[por_nombre.notna() & por_coords.notna() & (por_nombre == por_coords)] = "nombre_validado"
    metodo[por_nombre.notna() & por_coords.notna() & (por_nombre != por_coords)] = "nombre_coordenadas_discrepan"
    metodo[por_nombre.isna() & por_coords.notna()] = "coordenadas"

    df["id_municipio"] = por_nombre.fillna(por_coords).astype("string")
    df["metodo_asignacion"] = metodo
    df["asignado_otra_provincia"] = otra_provincia.astype(bool)
    return df


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
COLUMNAS = ["id_evento", "id_municipio", "campana", "numero_parte", "id_provincia", "municipio_egif",
            "fecha_deteccion", "fecha_extincion", "causa", "superficie_forestal_ha",
            "coordenada_x", "coordenada_y", "huso", "datum", "latitud", "longitud",
            "id_municipio_coordenadas", "metodo_asignacion", "asignado_otra_provincia"]


def procesar(municipios: pd.DataFrame, limites, desde: date | None = None, hasta: date | None = None,
             bronze_dir: Path | None = None, hf_dir: Path | None = None,
             equivalencias: pd.DataFrame | None = None) -> pd.DataFrame:
    ruta = fuentes.exigir(fuentes.ultimo_fichero("egif", "egif_galicia_unificado.parquet", bronze_dir, hf_dir),
                          "egif", "egif_galicia_unificado.parquet")
    log.info("EGIF: %s", ruta)
    df = tipar(pd.read_parquet(ruta))
    if desde:
        df = df[df["fecha_deteccion"] >= pd.Timestamp(desde)]
    if hasta:
        df = df[df["fecha_deteccion"] < pd.Timestamp(hasta) + pd.Timedelta(days=1)]
    df = asignar_municipio(df.reset_index(drop=True), municipios, limites, equivalencias)
    return df[COLUMNAS].sort_values(["fecha_deteccion", "id_evento"]).reset_index(drop=True)


def informe_no_asignados(eventos: pd.DataFrame) -> pd.DataFrame:
    """Partes sin código INE agrupados por provincia y nombre (HU-04)."""
    sin = eventos[eventos["id_municipio"].isna()]
    if sin.empty:
        return pd.DataFrame(columns=["id_provincia", "municipio_egif", "n_partes", "primera_campana",
                                     "ultima_campana", "con_coordenadas"])
    return (sin.assign(con_coord=sin["coordenada_x"].notna())
               .groupby(["id_provincia", "municipio_egif"], dropna=False)
               .agg(n_partes=("id_evento", "size"), primera_campana=("campana", "min"),
                    ultima_campana=("campana", "max"), con_coordenadas=("con_coord", "sum"))
               .reset_index().sort_values("n_partes", ascending=False))


ENLACE_MAPA = "https://www.google.com/maps/search/?api=1&query={lat:.5f},{lon:.5f}"


def muestra_validacion(eventos: pd.DataFrame, municipios: pd.DataFrame, n: int = 50,
                       n_discrepancias: int = 10, desde_campana: int = 2001,
                       semilla: int = 42) -> pd.DataFrame:
    """Muestra (con semilla) de partes asignados para validarla a mano (HU-04).

    Sólo partes desde ``desde_campana`` (periodo del modelo) con coordenadas, para
    poder comprobarlos en el mapa: ``n - n_discrepancias`` al azar (estrato
    ``aleatoria``, estima el acierto global) y ``n_discrepancias`` en que nombre y
    coordenadas no coinciden (estrato ``discrepancia``, los casos dudosos).
    """
    candidatos = eventos[eventos["id_municipio"].notna() & eventos["latitud"].notna()
                         & (eventos["campana"] >= desde_campana)]
    aleatoria = candidatos.sample(min(n - n_discrepancias, len(candidatos)), random_state=semilla)
    resto = candidatos.drop(aleatoria.index)
    resto = resto[resto["metodo_asignacion"] == "nombre_coordenadas_discrepan"]
    discrepancia = resto.sample(min(n_discrepancias, len(resto)), random_state=semilla)
    muestra = pd.concat([aleatoria.assign(estrato="aleatoria"), discrepancia.assign(estrato="discrepancia")])
    nombres = municipios.set_index("id_municipio")["nombre"]
    muestra = muestra.assign(
        nombre_asignado=muestra["id_municipio"].map(nombres),
        nombre_coordenadas=muestra["id_municipio_coordenadas"].map(nombres),
        enlace_mapa=[ENLACE_MAPA.format(lat=la, lon=lo) for la, lo in zip(muestra["latitud"], muestra["longitud"])],
        validacion_manual="", observaciones="")
    return muestra[["estrato", "id_evento", "campana", "municipio_egif", "id_provincia", "id_municipio",
                    "nombre_asignado", "id_municipio_coordenadas", "nombre_coordenadas", "metodo_asignacion",
                    "latitud", "longitud", "enlace_mapa", "validacion_manual", "observaciones"]]


def resumen_asignacion(eventos: pd.DataFrame) -> pd.DataFrame:
    """% de partes con código INE por campaña y provincia."""
    return (eventos.groupby(["campana", "id_provincia"], dropna=False)
                   .agg(n_partes=("id_evento", "size"),
                        n_asignados=("id_municipio", lambda s: int(s.notna().sum())),
                        n_validados=("metodo_asignacion", lambda s: int((s == "nombre_validado").sum())))
                   .assign(pct_asignados=lambda d: (100 * d["n_asignados"] / d["n_partes"]).round(2))
                   .reset_index())


def asignacion_a_revisar(eventos: pd.DataFrame, municipios: pd.DataFrame) -> pd.DataFrame:
    """Partes en que nombre y coordenadas no coinciden o sólo se asignaron por coordenadas (DEC-35)."""
    nombres = municipios.set_index("id_municipio")["nombre"]
    revisar = eventos[eventos["metodo_asignacion"].isin(["nombre_coordenadas_discrepan", "coordenadas"])]
    revisar = revisar.assign(nombre_asignado=revisar["id_municipio"].map(nombres),
                             nombre_coordenadas=revisar["id_municipio_coordenadas"].map(nombres))
    return revisar[["id_evento", "campana", "id_provincia", "municipio_egif", "metodo_asignacion",
                    "asignado_otra_provincia", "id_municipio", "nombre_asignado", "id_municipio_coordenadas",
                    "nombre_coordenadas", "latitud", "longitud"]]
