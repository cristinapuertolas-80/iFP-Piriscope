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


def nombre_fichero(anio: int, mes: int, cfg: config.EffisConfig) -> str:
    return f"fwi_{anio:04d}_{mes:02d}.{extension(cfg)}"


def _cliente_cdsapi(cfg: config.EffisConfig):
    try:
        import cdsapi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Instala cdsapi (pip install cdsapi) para descargar el FWI") from exc
    kwargs: dict[str, Any] = {"url": cfg.api_url, "quiet": True}
    if cfg.api_key:
        kwargs["key"] = cfg.api_key
    return cdsapi.Client(**kwargs)


def ingestar(desde: date, hasta: date, recortar: bool = True, forzar: bool = False,
             base_dir: Path | None = None, cfg: config.EffisConfig | None = None,
             descargar: Callable[[str, dict, str], Any] | None = None) -> EjecucionBronze:
    """Descarga el FWI y sus componentes, un fichero por mes, para ``[desde, hasta]``.

    :param recortar: pide sólo la caja de Galicia. Si el servidor rechaza el
        recorte, se reintenta sin él (el recorte se hará entonces en Silver).
    :param descargar: función ``(dataset, peticion, destino)``; por defecto,
        ``cdsapi.Client().retrieve``. Se inyecta en los tests.
    """
    cfg = cfg or config.EffisConfig()
    if descargar is None:
        cliente = _cliente_cdsapi(cfg)
        descargar = cliente.retrieve

    parametros = {"desde": desde.isoformat(), "hasta": hasta.isoformat(), "dataset": cfg.dataset,
                  "dataset_type": cfg.dataset_type, "variables": list(cfg.variables),
                  "area": config.BBOX_GALICIA if recortar else None, "formato": cfg.data_format}

    with EjecucionBronze("effis", parametros=parametros, base_dir=base_dir) as run:
        for anio, mes in meses_en_rango(desde, hasta):
            relativa = nombre_fichero(anio, mes, cfg)
            completo = (anio, mes) != (hasta.year, hasta.month) or hasta.day == calendar.monthrange(anio, mes)[1]
            if not forzar and completo:
                cache = buscar_en_cache("effis", relativa, excluir=run.dir, base_dir=base_dir)
                if cache is not None:
                    run.reutilizar(relativa, cache)
                    continue

            d_ini = desde.day if (anio, mes) == (desde.year, desde.month) else 1
            d_fin = hasta.day if (anio, mes) == (hasta.year, hasta.month) else None
            area = config.BBOX_GALICIA if recortar else None
            destino = run.ruta(relativa)
            peticion = construir_peticion(anio, mes, cfg, area, d_ini, d_fin)
            log.info("EFFIS %04d-%02d…", anio, mes)
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    tmp_destino = Path(tmp) / relativa
                    try:
                        descargar(cfg.dataset, peticion, str(tmp_destino))
                    except Exception as exc:  # noqa: BLE001
                        if area is None:
                            raise
                        run.incidencia("recorte_rechazado",
                                       f"{anio}-{mes:02d}: {exc}. Se reintenta sin 'area'")
                        peticion.pop("area", None)
                        descargar(cfg.dataset, peticion, str(tmp_destino))
                    destino.write_bytes(tmp_destino.read_bytes())
            except Exception as exc:  # noqa: BLE001
                run.incidencia("error_descarga", f"{anio}-{mes:02d}: {exc}")
                continue
            run.registrar_existente(relativa, origen=f"ewds:{cfg.dataset}",
                                    anio=anio, mes=mes, peticion=peticion)
    return run
