"""Línea de comandos de la capa Silver.

Ejemplos::

    # Todas las tablas, con todo lo que haya en Bronze
    python -m src.silver

    # Sólo un periodo (sustituye ese periodo en las tablas con fecha)
    python -m src.silver --desde 2024-01-01 --hasta 2024-12-31

    # Descargar antes los originales de Hugging Face y sólo la meteo
    python -m src.silver --descargar-hf --tablas municipios meteo
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from src.ingest.common import ErrorRed, configurar_logging

from .pipeline import TABLAS

log = logging.getLogger("piriscope.silver")


def construir_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m src.silver",
                                description="Capa Silver de Piriscope (Bronze -> data/silver)")
    p.add_argument("--desde", type=date.fromisoformat, help="Fecha inicial AAAA-MM-DD (incluida)")
    p.add_argument("--hasta", type=date.fromisoformat, help="Fecha final AAAA-MM-DD (incluida)")
    p.add_argument("--tablas", nargs="+", choices=TABLAS, default=list(TABLAS),
                   help="Tablas a generar (por defecto, todas)")
    p.add_argument("--descargar-hf", action="store_true",
                   help="Descargar antes de Hugging Face los originales de Bronze (raw/)")
    p.add_argument("--subir-hf", action="store_true",
                   help="Subir data/silver al dataset (requiere HF_TOKEN con escritura)")
    p.add_argument("--sin-validar", action="store_true",
                   help="No validar las tablas generadas contra el modelo de datos")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = construir_parser().parse_args(argv)
    configurar_logging(logging.DEBUG if args.verbose else logging.INFO)
    if args.desde and args.hasta and args.hasta < args.desde:
        log.error("--hasta debe ser posterior a --desde")
        return 2
    from . import hf_hub, pipeline, validacion

    try:
        if args.descargar_hf:
            hf_hub.descargar_bronze(args.tablas)
        pipeline.ejecutar(args.desde, args.hasta, args.tablas)
        if not args.sin_validar:
            v = validacion.validar()
            log.info("Validación: %d comprobaciones, %d errores, %d avisos (informes/validacion.json)",
                     len(v.resultados), len(v.errores), len(v.avisos))
            if v.errores:
                log.error("Las tablas Silver no cumplen el modelo de datos: no se suben a Hugging Face")
                return 1
        if args.subir_hf:
            hf_hub.subir_silver()
    except ErrorRed as exc:
        log.error("%s", exc)
        return 3
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
