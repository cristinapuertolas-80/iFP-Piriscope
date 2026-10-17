"""Ingesta del catálogo de municipios (INE) y de los límites municipales (CNIG).

* **INE**: relación de municipios y códigos (``diccionarioAA.xlsx``). Se
  descarga automáticamente; si el del año pedido aún no está publicado, se
  prueba con años anteriores. También admite URLs adicionales (por ejemplo,
  las variaciones anuales de municipios para ``docs/tabla_equivalencias.csv``).
* **CNIG**: los límites municipales (recintos) se descargan a mano desde el
  Centro de Descargas del CNIG (requiere aceptar la licencia en la web) y se
  dejan en ``data/landing/ine_cnig/``; aquí se copian sin modificar.

Todo se guarda en ``data/bronze/ine_cnig/<timestamp>/``. El cálculo de
centroides y el filtrado a Galicia se hacen en Silver.

Uso::

    python -m src.ingest ine_cnig --anio 2026
    python -m src.ingest ine_cnig --ficheros lineas_limite.zip
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import requests

from . import config
from .common import EjecucionBronze, crear_sesion

log = logging.getLogger("piriscope.ingest.ine_cnig")

EXTENSIONES_CNIG = {".zip", ".shp", ".shx", ".dbf", ".prj", ".cpg", ".gpkg", ".geojson", ".json"}


def url_diccionario(anio: int, cfg: config.IneConfig | None = None) -> str:
    cfg = cfg or config.IneConfig()
    return cfg.url_diccionario.format(aa=f"{anio % 100:02d}")


def descargar_diccionario(anio: int, sesion: requests.Session, intentos_previos: int = 2
                          ) -> tuple[int, str, bytes]:
    """Descarga el diccionario del año pedido o, si no existe aún, del anterior."""
    ultimo_error: Exception | None = None
    for a in range(anio, anio - intentos_previos - 1, -1):
        url = url_diccionario(a)
        try:
            r = sesion.get(url, timeout=120)
        except requests.RequestException as exc:
            ultimo_error = exc
            continue
        tipo = r.headers.get("Content-Type", "")
        if r.status_code == 200 and r.content[:2] == b"PK" and "html" not in tipo:
            return a, url, r.content
        ultimo_error = RuntimeError(f"{url} -> HTTP {r.status_code}")
        log.info("Diccionario INE %s no disponible (%s); probando año anterior", a, r.status_code)
    raise RuntimeError(f"No se pudo descargar el diccionario de municipios del INE: {ultimo_error}")


def localizar_ficheros(carpeta: Path | None = None) -> list[Path]:
    carpeta = Path(carpeta or config.LANDING_DIR / "ine_cnig")
    if not carpeta.exists():
        return []
    return sorted(p for p in carpeta.rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONES_CNIG)


def ingestar(anio: int | None = None, ficheros: list[Path] | None = None,
             urls: list[str] | None = None, sin_ine: bool = False,
             base_dir: Path | None = None, sesion: requests.Session | None = None) -> EjecucionBronze:
    anio = anio or date.today().year
    ficheros = [Path(f) for f in (ficheros or [])] or localizar_ficheros()
    parametros = {"anio": anio, "ficheros": [f.name for f in ficheros], "urls": urls or [],
                  "sin_ine": sin_ine}
    sesion = sesion or crear_sesion()

    with EjecucionBronze("ine_cnig", parametros=parametros, base_dir=base_dir) as run:
        if not sin_ine:
            try:
                anio_real, url, contenido = descargar_diccionario(anio, sesion)
                run.guardar_bytes(f"ine/diccionario{anio_real % 100:02d}.xlsx", contenido,
                                  origen=url, anio=anio_real)
                if anio_real != anio:
                    run.incidencia("anio_alternativo", f"Se usa el diccionario de {anio_real}")
            except RuntimeError as exc:
                run.incidencia("error_descarga", str(exc))

        for url in urls or []:
            try:
                r = sesion.get(url, timeout=120)
                r.raise_for_status()
            except requests.RequestException as exc:
                run.incidencia("error_descarga", f"{url}: {exc}")
                continue
            nombre = Path(url.split("?")[0]).name or "descarga"
            run.guardar_bytes(Path("ine") / nombre, r.content, origen=url)

        for f in ficheros:
            run.copiar_fichero(f, Path("cnig") / f.name)
        if not ficheros:
            run.incidencia(
                "cnig_pendiente",
                "No hay límites municipales del CNIG en data/landing/ine_cnig/ "
                "(descárgalos de https://centrodedescargas.cnig.es)",
            )
    return run
