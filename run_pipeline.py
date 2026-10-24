"""Orquestador único de Piriscope (apartado 5.8 del diseño técnico).

Encadena ingesta → Silver → Gold → entrenamiento/inferencia para un rango de
fechas, sin editar código (HU-11). En esta versión están implementadas las
etapas de **ingesta** (``src/ingest``) y **Silver** (``src/silver``); el resto se
irá incorporando en sus módulos de ``src/``.

Ejemplos::

    python run_pipeline.py --desde 2020-01-01 --hasta 2024-12-31
    python run_pipeline.py --desde 2024-01-01 --hasta 2024-12-31 --fuentes aemet effis --subir-hf
    python run_pipeline.py --desde 2001-01-01 --hasta 2024-12-31 --etapas silver
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

from src.ingest.common import configurar_logging

log = logging.getLogger("piriscope.pipeline")

ETAPAS = ("ingesta", "silver", "gold", "ml")
FUENTES = ("ine_cnig", "egif", "aemet", "effis")


def etapa_ingesta(desde: date, hasta: date, fuentes: list[str], subir_hf: bool) -> bool:
    from src.ingest import aemet, effis, egif, hf_hub, ine_cnig

    ok = True
    for fuente in fuentes:
        log.info("=== Ingesta %s ===", fuente)
        try:
            if fuente == "aemet":
                run = aemet.ingestar(desde, hasta)
            elif fuente == "effis":
                run = effis.ingestar(desde, hasta)
            elif fuente == "egif":
                if not egif.localizar_ficheros():
                    log.warning("EGIF: no hay exportaciones en data/landing/egif/; se omite")
                    continue
                run = egif.ingestar()
            elif fuente == "ine_cnig":
                run = ine_cnig.ingestar(anio=hasta.year)
            else:  # pragma: no cover
                raise ValueError(fuente)
        except Exception:  # noqa: BLE001 - una fuente caída no debe parar las demás
            log.exception("La ingesta de %s ha fallado", fuente)
            ok = False
            continue
        log.info("%s: %s, %d ficheros, %d incidencias", fuente, run.estado,
                 len(run.ficheros), len(run.incidencias))
        if subir_hf:
            hf_hub.subir_ejecucion(run.dir)
    return ok


def etapa_silver(desde: date, hasta: date, subir_hf: bool) -> bool:
    from src.silver import hf_hub, pipeline, validacion

    try:
        pipeline.ejecutar(desde, hasta)
        v = validacion.validar()
        log.info("Validación Silver: %d comprobaciones, %d errores, %d avisos",
                 len(v.resultados), len(v.errores), len(v.avisos))
        if v.errores:
            log.error("Las tablas Silver no cumplen el modelo de datos (data/silver/informes/validacion.json)")
            return False
        if subir_hf:
            hf_hub.subir_silver()
    except Exception:  # noqa: BLE001 - se informa y el orquestador devuelve error
        log.exception("La etapa Silver ha fallado")
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pipeline de Piriscope")
    p.add_argument("--desde", type=date.fromisoformat, required=True)
    p.add_argument("--hasta", type=date.fromisoformat, required=True)
    p.add_argument("--etapas", nargs="+", choices=ETAPAS, default=list(ETAPAS))
    p.add_argument("--fuentes", nargs="+", choices=FUENTES, default=list(FUENTES))
    p.add_argument("--subir-hf", action="store_true",
                   help="Subir a Hugging Face los originales (ingesta) y data/silver (Silver)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    configurar_logging(logging.DEBUG if args.verbose else logging.INFO)

    if args.hasta < args.desde:
        p.error("--hasta debe ser posterior a --desde")

    ok = True
    for etapa in args.etapas:
        if etapa == "ingesta":
            ok &= etapa_ingesta(args.desde, args.hasta, args.fuentes, args.subir_hf)
        elif etapa == "silver":
            ok &= etapa_silver(args.desde, args.hasta, args.subir_hf)
        else:
            log.warning("Etapa '%s' aún no implementada (pendiente en src/%s)", etapa,
                        {"gold": "features", "ml": "ml"}[etapa])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
