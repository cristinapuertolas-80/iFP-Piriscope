"""Ingesta del histórico de incendios de EGIF-MITECO (HU-02).

EGIF no ofrece una API: los partes se exportan a Excel por provincia desde el
buscador público (https://servicio.mapa.gob.es/incendios/Search/Publico) o se
obtienen como ficheros publicados por MITECO. Por eso esta ingesta trabaja con
ficheros locales (o URLs directas):

1. Los ficheros exportados se dejan en ``data/landing/egif/`` (o se pasan con
   ``--ficheros`` / ``--urls``).
2. Se copian **sin modificar** a ``data/bronze/egif/<timestamp>/originales/``.
3. Se genera, en la misma ejecución, un único Parquet con los partes de las
   cuatro provincias (``egif_galicia_unificado.parquet``): mismas filas y
   valores que los originales, con nombres de columna normalizados, columnas
   canónicas del Anexo A cuando se reconocen, el fichero de origen y el
   ``id_evento`` provisional. Se eliminan los partes duplicados por campaña y
   número de parte (exportaciones solapadas).
4. Informes: ``informe_duplicados.csv``, ``recuentos_provincia_anio.csv`` (y su
   contraste con las estadísticas oficiales si existe
   ``docs/egif_estadisticas_oficiales.csv``) y ``mapeo_columnas.json``.

El tipado definitivo, la asignación del código INE y la limpieza se hacen en
Silver (``src/features``).

Uso::

    python -m src.ingest egif                       # todo lo que haya en data/landing/egif
    python -m src.ingest egif --ficheros a.xlsx b.xlsx
"""

from __future__ import annotations

import io
import json
import logging
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from . import config
from .common import EjecucionBronze, crear_sesion

log = logging.getLogger("piriscope.ingest.egif")

EXTENSIONES = {".xlsx", ".xls", ".csv", ".txt"}
FICHERO_OFICIALES = config.REPO_ROOT / "docs" / "egif_estadisticas_oficiales.csv"

#: Columna canónica (Anexo A) -> nombres normalizados con que puede aparecer.
MAPEO_COLUMNAS: dict[str, tuple[str, ...]] = {
    "numero_parte": ("numero_parte", "numeroparte", "num_parte", "n_parte", "no_parte",
                     "numero_de_parte", "n_de_parte", "parte", "idpif", "id_pif", "id_parte"),
    "campana": ("campana", "campania", "ano", "anio", "ano_campana", "year"),
    "provincia": ("provincia", "nombre_provincia", "provincia_nombre"),
    "id_provincia": ("idprovincia", "id_provincia", "cod_provincia", "codigo_provincia"),
    "municipio": ("municipio", "nombre_municipio", "termino_municipal", "municipio_nombre"),
    "id_municipio_egif": ("idmunicipio", "id_municipio", "cod_municipio", "codigo_municipio",
                          "codigo_ine", "cod_ine"),
    "fecha_deteccion": ("fecha_deteccion", "fechadeteccion", "f_deteccion", "deteccion",
                        "fecha_de_deteccion", "fecha"),
    "fecha_extincion": ("fecha_extincion", "fechaextincion", "f_extincion", "extincion",
                        "fecha_de_extincion"),
    "causa": ("causa", "causa_incendio", "grupo_causa", "causa_probable", "idcausa"),
    "superficie_forestal_ha": ("superficie_forestal_ha", "superficie_forestal", "sup_forestal",
                               "superficie_total_forestal", "total_forestal", "sup_total_forestal"),
    "coordenada_x": ("coordenada_x", "coord_x", "x", "utm_x", "x_utm"),
    "coordenada_y": ("coordenada_y", "coord_y", "y", "utm_y", "y_utm"),
    "huso": ("huso", "huso_utm", "zona_utm"),
    "datum": ("datum", "sistema_referencia"),
}

_PROVINCIAS_POR_TOKEN = {
    "coruna": "15", "acoruna": "15", "15": "15",
    "lugo": "27", "27": "27",
    "ourense": "32", "orense": "32", "32": "32",
    "pontevedra": "36", "36": "36",
}


# ---------------------------------------------------------------------------
# Normalización de cabeceras
# ---------------------------------------------------------------------------
def normalizar_nombre(nombre: object) -> str:
    """'Fecha de Detección' -> 'fecha_de_deteccion'; 'Nº Parte' -> 'n_parte'."""
    texto = str(nombre).strip().replace("º", "").replace("ª", "")
    texto = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    texto = re.sub(r"\(.*?\)", "", texto)          # quita unidades: "(ha)"
    texto = re.sub(r"[^a-z0-9]+", "_", texto).strip("_")
    return texto or "columna"


def mapear_columnas(columnas: Iterable[str]) -> dict[str, str]:
    """Devuelve {columna_normalizada: columna_canonica} para las que se reconocen."""
    columnas = list(columnas)
    mapeo: dict[str, str] = {}
    usadas: set[str] = set()
    for canonica, candidatos in MAPEO_COLUMNAS.items():
        for cand in candidatos:
            if cand in columnas and cand not in usadas:
                mapeo[cand] = canonica
                usadas.add(cand)
                break
    return mapeo


def provincia_desde_nombre_fichero(nombre: str) -> str | None:
    tokens = re.split(r"[^a-z0-9]+", normalizar_nombre(Path(nombre).stem).replace("_", " "))
    compacto = normalizar_nombre(Path(nombre).stem).replace("_", "")
    for token in tokens:
        if token in _PROVINCIAS_POR_TOKEN:
            return _PROVINCIAS_POR_TOKEN[token]
    for clave, codigo in _PROVINCIAS_POR_TOKEN.items():
        if not clave.isdigit() and clave in compacto:
            return codigo
    return None


# ---------------------------------------------------------------------------
# Lectura de ficheros exportados
# ---------------------------------------------------------------------------
_PALABRAS_CABECERA = ("parte", "municipio", "provincia", "deteccion", "causa", "campana", "fecha")


def _detectar_cabecera(crudo: pd.DataFrame, max_filas: int = 20) -> int:
    """Índice de la fila que parece la cabecera (las exportaciones traen títulos encima)."""
    mejor, mejor_puntos = 0, -1
    for i in range(min(max_filas, len(crudo))):
        celdas = [normalizar_nombre(c) for c in crudo.iloc[i].tolist() if pd.notna(c) and str(c).strip()]
        puntos = sum(any(p in c for p in _PALABRAS_CABECERA) for c in celdas)
        if puntos > mejor_puntos:
            mejor, mejor_puntos = i, puntos
    return mejor


def _leer_csv(ruta: Path) -> pd.DataFrame:
    bruto = ruta.read_bytes()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            texto = bruto.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return pd.read_csv(io.StringIO(texto), sep=None, engine="python", header=None, dtype=str)


def leer_fichero(ruta: Path) -> list[pd.DataFrame]:
    """Lee un fichero exportado de EGIF y devuelve una tabla por hoja, todo como texto."""
    ruta = Path(ruta)
    sufijo = ruta.suffix.lower()
    if sufijo in {".csv", ".txt"}:
        hojas = {"csv": _leer_csv(ruta)}
    elif sufijo in {".xlsx", ".xls"}:
        hojas = pd.read_excel(ruta, sheet_name=None, header=None, dtype=str)
    else:
        raise ValueError(f"Formato no soportado: {ruta.name}")

    tablas = []
    for nombre_hoja, crudo in hojas.items():
        crudo = crudo.dropna(how="all").dropna(axis=1, how="all")
        if crudo.empty:
            continue
        fila = _detectar_cabecera(crudo)
        cabecera = [normalizar_nombre(c) for c in crudo.iloc[fila].tolist()]
        # Evita nombres repetidos ("x", "x" -> "x", "x_2")
        vistos: dict[str, int] = {}
        unicos = []
        for c in cabecera:
            vistos[c] = vistos.get(c, 0) + 1
            unicos.append(c if vistos[c] == 1 else f"{c}_{vistos[c]}")
        tabla = crudo.iloc[fila + 1:].copy()
        tabla.columns = unicos
        tabla = tabla.dropna(how="all").reset_index(drop=True)
        tabla["fichero_origen"] = ruta.name
        tabla["hoja_origen"] = str(nombre_hoja)
        tablas.append(tabla)
    return tablas


# ---------------------------------------------------------------------------
# Unificación
# ---------------------------------------------------------------------------
def _anio_desde_fecha(serie: pd.Series) -> pd.Series:
    fechas = pd.to_datetime(serie, errors="coerce", dayfirst=True)
    return fechas.dt.year.astype("Int64").astype("string")


def _limpiar_numero_parte(serie: pd.Series) -> pd.Series:
    # "12345.0" (Excel) -> "12345"; conserva ceros iniciales si vienen como texto
    return serie.astype("string").str.strip().str.replace(r"\.0+$", "", regex=True)


def unificar(tablas: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Une las tablas, añade columnas canónicas e ``id_evento`` y quita duplicados.

    Devuelve (unificado, duplicados_eliminados, info).
    """
    if not tablas:
        raise ValueError("No hay tablas de EGIF que unificar")

    preparadas, mapeos = [], {}
    for t in tablas:
        mapeo = mapear_columnas(t.columns)
        mapeos[f"{t['fichero_origen'].iat[0]}::{t['hoja_origen'].iat[0]}"] = mapeo
        t = t.rename(columns=mapeo)
        if "id_provincia" not in t.columns:
            t["id_provincia"] = provincia_desde_nombre_fichero(t["fichero_origen"].iat[0])
        preparadas.append(t)

    df = pd.concat(preparadas, ignore_index=True, sort=False).astype("string")

    if "numero_parte" in df.columns:
        df["numero_parte"] = _limpiar_numero_parte(df["numero_parte"])
    if "campana" not in df.columns and "fecha_deteccion" in df.columns:
        df["campana"] = _anio_desde_fecha(df["fecha_deteccion"])
    elif "campana" in df.columns and "fecha_deteccion" in df.columns:
        df["campana"] = df["campana"].fillna(_anio_desde_fecha(df["fecha_deteccion"]))
    if "campana" in df.columns:
        df["campana"] = df["campana"].str.replace(r"\.0+$", "", regex=True)

    info: dict = {"mapeos": mapeos, "filas_leidas": int(len(df))}
    faltan = [c for c in ("numero_parte", "campana") if c not in df.columns]
    if faltan:
        info["aviso"] = f"No se reconocen las columnas {faltan}: no se puede deduplicar ni crear id_evento"
        df["id_evento"] = pd.NA
        return df, df.iloc[0:0], info

    # 1) Duplicados exactos de parte (misma campaña, nº y provincia): exportaciones solapadas
    clave_base = ["campana", "id_provincia", "numero_parte"]
    duplicado = df.duplicated(subset=clave_base, keep="first")
    eliminados = df[duplicado].copy()
    df = df[~duplicado].copy()

    # 2) ¿Se repite el nº de parte entre provincias en la misma campaña? (DEC-14, EDA)
    repetidos = df.groupby(["campana", "numero_parte"])["id_provincia"].transform("nunique") > 1
    info["numero_parte_repetido_entre_provincias"] = int(repetidos.sum())
    if repetidos.any():
        df["id_evento"] = df["campana"] + "_" + df["id_provincia"].fillna("NA") + "_" + df["numero_parte"]
        info["formato_id_evento"] = "campana_provincia_numero"
    else:
        df["id_evento"] = df["campana"] + "_" + df["numero_parte"]
        info["formato_id_evento"] = "campana_numero"

    info["duplicados_eliminados"] = int(len(eliminados))
    info["filas_finales"] = int(len(df))
    return df.reset_index(drop=True), eliminados, info


def recuentos_provincia_anio(df: pd.DataFrame, oficiales: Path | None = FICHERO_OFICIALES) -> pd.DataFrame:
    """Nº de partes por provincia y campaña, contrastado con las cifras oficiales si existen.

    ``docs/egif_estadisticas_oficiales.csv`` debe tener las columnas
    ``id_provincia,campana,n_incendios_oficial``.
    """
    if not {"id_provincia", "campana"} <= set(df.columns):
        return pd.DataFrame()
    rec = (df.groupby(["id_provincia", "campana"], dropna=False).size()
             .rename("n_partes").reset_index())
    rec["provincia"] = rec["id_provincia"].map(config.PROVINCIAS_GALICIA)
    if oficiales is not None and Path(oficiales).exists():
        of = pd.read_csv(oficiales, dtype={"id_provincia": str, "campana": str})
        rec = rec.merge(of, on=["id_provincia", "campana"], how="outer")
        rec["diferencia"] = rec["n_partes"] - rec["n_incendios_oficial"]
        rec["diferencia_pct"] = (100 * rec["diferencia"] / rec["n_incendios_oficial"]).round(2)
    return rec.sort_values(["id_provincia", "campana"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
def localizar_ficheros(carpeta: Path | None = None) -> list[Path]:
    carpeta = Path(carpeta or config.LANDING_DIR / "egif")
    if not carpeta.exists():
        return []
    return sorted(p for p in carpeta.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONES)


def ingestar(ficheros: list[Path] | None = None, urls: list[str] | None = None,
             base_dir: Path | None = None, oficiales: Path | None = FICHERO_OFICIALES) -> EjecucionBronze:
    ficheros = [Path(f) for f in (ficheros or [])] or ([] if urls else localizar_ficheros())
    if not ficheros and not urls:
        raise FileNotFoundError(
            "No hay ficheros de EGIF. Exporta los partes por provincia desde "
            "https://servicio.mapa.gob.es/incendios/Search/Publico y déjalos en data/landing/egif/"
        )

    parametros = {"ficheros": [f.name for f in ficheros], "urls": urls or []}
    with EjecucionBronze("egif", parametros=parametros, base_dir=base_dir) as run:
        originales: list[Path] = []
        for f in ficheros:
            originales.append(run.copiar_fichero(f, Path("originales") / f.name))
        if urls:
            sesion = crear_sesion()
            for url in urls:
                r = sesion.get(url, timeout=120)
                r.raise_for_status()
                nombre = Path(url.split("?")[0]).name or "egif_descarga"
                originales.append(run.guardar_bytes(Path("originales") / nombre, r.content, origen=url))

        tablas: list[pd.DataFrame] = []
        for ruta in originales:
            try:
                tablas.extend(leer_fichero(ruta))
            except Exception as exc:  # noqa: BLE001 - se documenta y se sigue
                run.incidencia("lectura_fallida", f"{ruta.name}: {exc}")

        unificado, eliminados, info = unificar(tablas)
        unificado["fecha_ingesta"] = datetime.now(timezone.utc).isoformat()
        unificado.to_parquet(run.ruta("egif_galicia_unificado.parquet"), index=False)
        run.registrar_existente("egif_galicia_unificado.parquet", origen="derivado:originales")

        eliminados.to_csv(run.ruta("informe_duplicados.csv"), index=False)
        run.registrar_existente("informe_duplicados.csv", origen="derivado:originales")

        recuentos = recuentos_provincia_anio(unificado, oficiales)
        recuentos.to_csv(run.ruta("recuentos_provincia_anio.csv"), index=False)
        run.registrar_existente("recuentos_provincia_anio.csv", origen="derivado:originales")

        run.ruta("mapeo_columnas.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        run.registrar_existente("mapeo_columnas.json", origen="derivado:originales")

        if "aviso" in info:
            run.incidencia("columnas_no_reconocidas", info["aviso"])
        provincias = set(unificado.get("id_provincia", pd.Series(dtype="string")).dropna())
        for cod, nombre in config.PROVINCIAS_GALICIA.items():
            if cod not in provincias:
                run.incidencia("provincia_ausente", f"No hay partes de {nombre} ({cod})")
        log.info("EGIF: %s filas leídas, %s duplicados eliminados, %s partes finales",
                 info.get("filas_leidas"), info.get("duplicados_eliminados", 0), len(unificado))
    return run
