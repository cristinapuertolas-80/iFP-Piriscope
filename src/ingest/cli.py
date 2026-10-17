"""Línea de comandos de la ingesta.

Ejemplos::

    # Histórico de AEMET (una sola vez; se puede reanudar gracias a la caché)
    python -m src.ingest aemet --desde 2010-01-01 --hasta 2024-12-31

    # Últimos 10 días de AEMET y subida a Hugging Face (lo que hace el cron)
    python -m src.ingest aemet --dias 10 --subir-hf

    # EGIF desde las exportaciones dejadas en data/landing/egif/
    python -m src.ingest egif

    # FWI de EFFIS/Copernicus
    python -m src.ingest effis --desde 2015-01-01 --hasta 2024-12-31

    # Catálogo de municipios del INE + límites del CNIG
    python -m src.ingest ine_cnig --anio 2026

    # Subir a Hugging Face todo lo pendiente
    python -m src.ingest subir
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from .common import configurar_logging

log = logging.getLogger("piriscope.ingest")


def _fecha(texto: str) -> date:
    try:
        return date.fromisoformat(texto)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Fecha no válida (AAAA-MM-DD): {texto}") from exc


def resolver_rango(desde: date | None, hasta: date | None, dias: int | None,
                   hoy: date | None = None) -> tuple[date, date]:
    """``--dias N`` = los N días que terminan ayer. Si no, ``--desde``/``--hasta``."""
    hoy = hoy or date.today()
    if dias:
        fin = hasta or hoy - timedelta(days=1)
        return fin - timedelta(days=dias - 1), fin
    if desde is None:
        raise SystemExit("Indica --desde (y opcionalmente --hasta) o --dias")
    return desde, hasta or hoy - timedelta(days=1)


def _anadir_rango(p: argparse.ArgumentParser) -> None:
    p.add_argument("--desde", type=_fecha, help="Fecha inicial AAAA-MM-DD (incluida)")
    p.add_argument("--hasta", type=_fecha, help="Fecha final AAAA-MM-DD (incluida; por defecto ayer)")
    p.add_argument("--dias", type=int, help="Alternativa: los últimos N días hasta ayer")


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.ingest",
                                     description="Ingesta de datos de Piriscope (capa Bronze)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Más detalle en el log")
    sub = parser.add_subparsers(dest="fuente", required=True)

    p = sub.add_parser("aemet", help="Meteorología diaria de AEMET OpenData")
    _anadir_rango(p)
    p.add_argument("--estaciones", nargs="+", help="Indicativos concretos (por defecto, todas las gallegas)")
    p.add_argument("--forzar", action="store_true", help="Ignorar la caché de Bronze")
    p.add_argument("--subir-hf", action="store_true", help="Subir la ejecución a Hugging Face")

    p = sub.add_parser("egif", help="Histórico de incendios EGIF-MITECO (ficheros exportados)")
    p.add_argument("--ficheros", nargs="+", type=Path, help="Ficheros Excel/CSV (por defecto data/landing/egif/)")
    p.add_argument("--urls", nargs="+", help="URLs directas a ficheros de EGIF")
    p.add_argument("--subir-hf", action="store_true")

    p = sub.add_parser("effis", help="FWI y componentes de EFFIS/Copernicus (EWDS)")
    _anadir_rango(p)
    p.add_argument("--sin-recorte", action="store_true", help="No recortar a la caja de Galicia")
    p.add_argument("--forzar", action="store_true")
    p.add_argument("--subir-hf", action="store_true")

    p = sub.add_parser("ine_cnig", help="Catálogo de municipios (INE) y límites (CNIG)")
    p.add_argument("--anio", type=int, help="Año del diccionario de municipios del INE")
    p.add_argument("--ficheros", nargs="+", type=Path, help="Ficheros del CNIG (por defecto data/landing/ine_cnig/)")
    p.add_argument("--urls", nargs="+", help="URLs adicionales del INE (p. ej. variaciones de municipios)")
    p.add_argument("--sin-ine", action="store_true", help="No descargar el diccionario del INE")
    p.add_argument("--subir-hf", action="store_true")

    p = sub.add_parser("subir", help="Subir a Hugging Face las ejecuciones locales pendientes")
    p.add_argument("--solo", dest="fuente_filtro", choices=["aemet", "egif", "effis", "ine_cnig"],
                   help="Subir sólo esta fuente")

    p = sub.add_parser("descargar", help="Descargar de Hugging Face los datos originales de una fuente")
    p.add_argument("fuente_hf", choices=["aemet", "egif", "effis", "ine_cnig"])
    p.add_argument("--timestamp", help="Una ejecución concreta (por defecto, todas)")
    p.add_argument("--destino", type=Path)
    return parser


def ejecutar(args: argparse.Namespace) -> int:
    from . import hf_hub

    run = None
    if args.fuente == "aemet":
        from . import aemet
        desde, hasta = resolver_rango(args.desde, args.hasta, args.dias)
        run = aemet.ingestar(desde, hasta, estaciones=args.estaciones, forzar=args.forzar)
    elif args.fuente == "egif":
        from . import egif
        run = egif.ingestar(ficheros=args.ficheros, urls=args.urls)
    elif args.fuente == "effis":
        from . import effis
        desde, hasta = resolver_rango(args.desde, args.hasta, args.dias)
        run = effis.ingestar(desde, hasta, recortar=not args.sin_recorte, forzar=args.forzar)
    elif args.fuente == "ine_cnig":
        from . import ine_cnig
        run = ine_cnig.ingestar(anio=args.anio, ficheros=args.ficheros, urls=args.urls, sin_ine=args.sin_ine)
    elif args.fuente == "subir":
        subidas = hf_hub.subir_pendientes(args.fuente_filtro)
        log.info("Subidas: %s", subidas or "ninguna pendiente")
        return 0
    elif args.fuente == "descargar":
        ruta = hf_hub.descargar(args.fuente_hf, args.timestamp, args.destino)
        print(ruta)
        return 0

    if run is not None:
        log.info("Ejecución %s: %s (%d ficheros, %d incidencias) -> %s", run.fuente, run.estado,
                 len(run.ficheros), len(run.incidencias), run.dir)
        if getattr(args, "subir_hf", False):
            hf_hub.subir_ejecucion(run.dir)
        return 0 if run.estado != "fallida" else 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    configurar_logging(logging.DEBUG if args.verbose else logging.INFO)
    try:
        return ejecutar(args)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
