"""Ingesta del FWI y sus componentes de EFFIS/Copernicus (HU-03).

Se usa el conjunto *Fire danger indices historical data from the Copernicus
Emergency Management Service* (``cems-fire-historical-v1``), servido por el
Early Warning Data Store (EWDS) mediante ``cdsapi``. Requiere registro gratuito
y una API key de EWDS:

* ``CDSAPI_URL=https://ewds.climate.copernicus.eu/api``
* ``CDSAPI_KEY=<tu clave personal>``

(o un fichero ``~/.cdsapirc`` con esas dos líneas).

Se descarga un fichero por mes, recortado a la caja de Galicia
(:data:`config.BBOX_GALICIA`), y se guarda **sin modificar** en
``data/bronze/effis/<timestamp>/fwi_<AAAA>_<MM>.<nc|grib>``. El muestreo en el
centroide de cada municipio (DEC-09) se hace en Silver.

Uso::

    python -m src.ingest effis --desde 2015-01-01 --hasta 2024-12-31
"""

from __future__ import annotations

import calendar
import logging
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Callable

from . import config
from .common import EjecucionBronze, buscar_en_cache, meses_en_rango

log = logging.getLogger("piriscope.ingest.effis")


def construir_peticion(anio: int, mes: int, cfg: config.EffisConfig,
                       area: list[float] | None = config.BBOX_GALICIA,
                       dia_desde: int = 1, dia_hasta: int | None = None) -> dict[str, Any]:
    ultimo = calendar.monthrange(anio, mes)[1]
    dia_hasta = min(dia_hasta or ultimo, ultimo)
    peticion: dict[str, Any] = {
        "product_type": cfg.product_type,
        "variable": list(cfg.variables),
        "dataset_type": cfg.dataset_type,
        "system_version": [cfg.system_version],
        "year": [f"{anio:04d}"],
        "month": [f"{mes:02d}"],
        "day": [f"{d:02d}" for d in range(dia_desde, dia_hasta + 1)],
        "grid": cfg.grid,
        "data_format": cfg.data_format,
    }
    if area:
        peticion["area"] = list(area)  # [N, W, S, E]
    return peticion


def extension(cfg: config.EffisConfig) -> str:
    return "nc" if cfg.data_format.lower().startswith("netcdf") else "grib"


def nombre_fichero(anio: int, mes: int, cfg: config.EffisConfig, intermedio: bool = False) -> str:
    """``fwi_AAAA_MM.nc`` (consolidado) o ``fwi_AAAA_MM_intermedio.nc``.

    Los meses descargados del dataset intermedio llevan otro nombre para que la
    caché no los confunda con los consolidados: en la siguiente ejecución se
    vuelve a intentar el consolidado y, cuando ya existe, sustituye al intermedio.
    En Silver, si hay ambos para un mismo mes, prevalece el consolidado.
    """
    sufijo = "_intermedio" if intermedio else ""
    return f"fwi_{anio:04d}_{mes:02d}{sufijo}.{extension(cfg)}"


def _cliente_cdsapi(cfg: config.EffisConfig):
    try:
        import cdsapi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Instala cdsapi (pip install cdsapi) para descargar el FWI") from exc
    kwargs: dict[str, Any] = {"url": cfg.api_url, "quiet": True}
    if cfg.api_key:
        kwargs["key"] = cfg.api_key
    return cdsapi.Client(**kwargs)


def _descargar_mes(descargar: Callable[[str, dict, str], Any], dataset: str, peticion: dict,
                   destino: Path, run: EjecucionBronze, etiqueta: str) -> dict:
    """Descarga un mes; si el servidor rechaza el recorte ``area``, reintenta sin él."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_destino = Path(tmp) / destino.name
        try:
            descargar(dataset, peticion, str(tmp_destino))
        except Exception as exc:  # noqa: BLE001
            if "area" not in peticion:
                raise
            run.incidencia("recorte_rechazado", f"{etiqueta}: {exc}. Se reintenta sin 'area'")
            peticion = {k: v for k, v in peticion.items() if k != "area"}
            descargar(dataset, peticion, str(tmp_destino))
        destino.write_bytes(tmp_destino.read_bytes())
    return peticion


def ingestar(desde: date, hasta: date, recortar: bool = True, forzar: bool = False,
             base_dir: Path | None = None, cfg: config.EffisConfig | None = None,
             descargar: Callable[[str, dict, str], Any] | None = None,
             alternativa_intermedia: bool = True) -> EjecucionBronze:
    """Descarga el FWI y sus componentes, un fichero por mes, para ``[desde, hasta]``.

    :param recortar: pide sólo la caja de Galicia. Si el servidor rechaza el
        recorte, se reintenta sin él (el recorte se hará entonces en Silver).
    :param alternativa_intermedia: si un mes aún no está en el dataset
        consolidado, se descarga del intermedio (``fwi_AAAA_MM_intermedio.nc``).
        Así una misma orden sirve para el histórico y para los meses recientes.
    :param descargar: función ``(dataset, peticion, destino)``; por defecto,
        ``cdsapi.Client().retrieve``. Se inyecta en los tests.
    """
    cfg = cfg or config.EffisConfig()
    if descargar is None:
        cliente = _cliente_cdsapi(cfg)
        descargar = cliente.retrieve

    parametros = {"desde": desde.isoformat(), "hasta": hasta.isoformat(), "dataset": cfg.dataset,
                  "dataset_type": cfg.dataset_type, "alternativa_intermedia": alternativa_intermedia,
                  "variables": list(cfg.variables),
                  "area": config.BBOX_GALICIA if recortar else None, "formato": cfg.data_format}
    usar_alternativa = alternativa_intermedia and cfg.dataset_type == "consolidated_dataset"

    with EjecucionBronze("effis", parametros=parametros, base_dir=base_dir) as run:
        for anio, mes in meses_en_rango(desde, hasta):
            intermedio_cfg = cfg.dataset_type == "intermediate_dataset"
            relativa = nombre_fichero(anio, mes, cfg, intermedio=intermedio_cfg)
            completo = (anio, mes) != (hasta.year, hasta.month) or hasta.day == calendar.monthrange(anio, mes)[1]
            if not forzar and completo:
                cache = buscar_en_cache("effis", relativa, excluir=run.dir, base_dir=base_dir)
                if cache is not None:
                    run.reutilizar(relativa, cache)
                    continue

            d_ini = desde.day if (anio, mes) == (desde.year, desde.month) else 1
            d_fin = hasta.day if (anio, mes) == (hasta.year, hasta.month) else None
            area = config.BBOX_GALICIA if recortar else None
            peticion = construir_peticion(anio, mes, cfg, area, d_ini, d_fin)
            etiqueta = f"{anio}-{mes:02d}"
            log.info("EFFIS %s (%s)…", etiqueta, cfg.dataset_type)
            try:
                peticion = _descargar_mes(descargar, cfg.dataset, peticion, run.ruta(relativa), run, etiqueta)
            except Exception as exc:  # noqa: BLE001
                run.dir.joinpath(relativa).unlink(missing_ok=True)
                if not usar_alternativa:
                    run.incidencia("error_descarga", f"{etiqueta}: {exc}")
                    continue
                relativa = nombre_fichero(anio, mes, cfg, intermedio=True)
                peticion = {**peticion, "dataset_type": "intermediate_dataset"}
                run.incidencia("usa_intermedio",
                               f"{etiqueta}: no disponible en consolidated_dataset ({exc}); "
                               "se descarga intermediate_dataset")
                try:
                    peticion = _descargar_mes(descargar, cfg.dataset, peticion, run.ruta(relativa),
                                              run, etiqueta)
                except Exception as exc2:  # noqa: BLE001
                    run.dir.joinpath(relativa).unlink(missing_ok=True)
                    run.incidencia("error_descarga", f"{etiqueta}: {exc2}")
                    continue
            run.registrar_existente(relativa, origen=f"ewds:{cfg.dataset}",
                                    anio=anio, mes=mes, peticion=peticion)
    return run
