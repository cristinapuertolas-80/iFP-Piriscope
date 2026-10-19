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
import time
from datetime import date
from pathlib import Path
from typing import Any, Callable

from . import config
from .common import EjecucionBronze, ErrorRed, buscar_en_cache, error_red, es_error_red, meses_en_rango

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


class ColaEwdsLlena(ErrorRed):
    """El EWDS sigue rechazando trabajos por tener demasiadas peticiones en cola."""


def es_limite_cola(exc: BaseException) -> bool:
    """True si el EWDS rechaza el trabajo por el límite de peticiones en cola.

    El servidor responde con un título genérico («invalid request. Request has not
    produced a valid combination of values») seguido del motivo real («The job has
    been rejected. Number queued requests for this dataset is temporarly limited»),
    así que no debe confundirse con un rechazo del recorte ``area``.
    """
    texto = str(exc).lower()
    return any(m in texto for m in ("queued requests", "temporarly limited", "temporarily limited",
                                    "too many requests"))


def _descargar_con_espera(descargar: Callable[[str, dict, str], Any], dataset: str, peticion: dict,
                          destino: str, cfg: config.EffisConfig, run: EjecucionBronze, etiqueta: str,
                          dormir: Callable[[float], Any]) -> None:
    for intento in range(1, cfg.max_reintentos_cola + 1):
        try:
            descargar(dataset, peticion, destino)
            return
        except Exception as exc:  # noqa: BLE001
            if not es_limite_cola(exc):
                raise
            if intento == 1:
                run.incidencia("cola_llena", f"{etiqueta}: el EWDS limita las peticiones en cola; "
                                             f"se espera {cfg.espera_cola_segundos:.0f} s y se reintenta")
            log.warning("EWDS: cola llena (%s/%s). Esperando %.0f s…", intento,
                        cfg.max_reintentos_cola, cfg.espera_cola_segundos)
            dormir(cfg.espera_cola_segundos)
    raise ColaEwdsLlena(
        "EFFIS/EWDS: el servidor sigue rechazando trabajos por tener demasiadas peticiones en cola. "
        "Cancela las peticiones pendientes en https://ewds.climate.copernicus.eu/requests, comprueba "
        "que no hay otra carga en marcha y vuelve a lanzar la orden (lo ya descargado se reutiliza)."
    )


def _descargar_mes(descargar: Callable[[str, dict, str], Any], dataset: str, peticion: dict,
                   destino: Path, run: EjecucionBronze, etiqueta: str,
                   cfg: config.EffisConfig | None = None,
                   dormir: Callable[[float], Any] = time.sleep) -> dict:
    """Descarga un mes; si el servidor rechaza el recorte ``area``, reintenta sin él.

    Las esperas por cola llena y los errores de red no se confunden con un
    rechazo del recorte.
    """
    cfg = cfg or config.EffisConfig()
    with tempfile.TemporaryDirectory() as tmp:
        tmp_destino = Path(tmp) / destino.name
        try:
            _descargar_con_espera(descargar, dataset, peticion, str(tmp_destino), cfg, run, etiqueta, dormir)
        except Exception as exc:  # noqa: BLE001
            if "area" not in peticion or isinstance(exc, ColaEwdsLlena) or es_error_red(exc):
                raise
            run.incidencia("recorte_rechazado", f"{etiqueta}: {exc}. Se reintenta sin 'area' "
                                                "(y sin él el resto de meses de esta ejecución)")
            peticion = {k: v for k, v in peticion.items() if k != "area"}
            _descargar_con_espera(descargar, dataset, peticion, str(tmp_destino), cfg, run, etiqueta, dormir)
        destino.write_bytes(tmp_destino.read_bytes())
    return peticion


def ingestar(desde: date, hasta: date, recortar: bool = True, forzar: bool = False,
             base_dir: Path | None = None, cfg: config.EffisConfig | None = None,
             descargar: Callable[[str, dict, str], Any] | None = None,
             alternativa_intermedia: bool = True,
             dormir: Callable[[float], Any] = time.sleep) -> EjecucionBronze:
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
    recorte_activo = recortar

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
            area = config.BBOX_GALICIA if recorte_activo else None
            peticion = construir_peticion(anio, mes, cfg, area, d_ini, d_fin)
            etiqueta = f"{anio}-{mes:02d}"
            log.info("EFFIS %s (%s)…", etiqueta, cfg.dataset_type)
            try:
                peticion = _descargar_mes(descargar, cfg.dataset, peticion, run.ruta(relativa), run,
                                          etiqueta, cfg, dormir)
            except Exception as exc:  # noqa: BLE001
                run.dir.joinpath(relativa).unlink(missing_ok=True)
                if isinstance(exc, ColaEwdsLlena):
                    run.incidencia("cola_llena", f"{etiqueta}: {exc}")
                    raise
                if es_error_red(exc):
                    # Sin conexión no tiene sentido probar otro dataset ni otro mes
                    run.incidencia("error_red", f"{etiqueta}: {exc}")
                    raise error_red("EFFIS/EWDS", exc) from exc
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
                                              run, etiqueta, cfg, dormir)
                except Exception as exc2:  # noqa: BLE001
                    run.dir.joinpath(relativa).unlink(missing_ok=True)
                    if isinstance(exc2, ColaEwdsLlena):
                        run.incidencia("cola_llena", f"{etiqueta}: {exc2}")
                        raise
                    if es_error_red(exc2):
                        run.incidencia("error_red", f"{etiqueta}: {exc2}")
                        raise error_red("EFFIS/EWDS", exc2) from exc2
                    run.incidencia("error_descarga", f"{etiqueta}: {exc2}")
                    continue
            if area is not None and "area" not in peticion:
                recorte_activo = False  # el servidor no acepta el recorte: no se vuelve a intentar
            run.registrar_existente(relativa, origen=f"ewds:{cfg.dataset}",
                                    anio=anio, mes=mes, peticion=peticion)
    return run
