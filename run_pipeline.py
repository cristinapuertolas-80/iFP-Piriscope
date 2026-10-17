"""Orquestador único de Piriscope (apartado 5.8 del diseño técnico).

Encadena ingesta → Silver → Gold → entrenamiento/inferencia para un rango de
fechas, sin editar código (HU-11). En esta versión sólo está implementada la
etapa de **ingesta**; el resto se irá incorporando en sus módulos de ``src/``.

Ejemplos::

    python run_pipeline.py --desde 2020-01-01 --hasta 2024-12-31
    python run_pipeline.py --desde 2024-01-01 --hasta 2024-12-31 --fuentes aemet effis --subir-hf
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


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pipeline de Piriscope")
    p.add_argument("--desde", type=date.fromisoformat, required=True)
    p.add_argument("--hasta", type=date.fromisoformat, required=True)
    p.add_argument("--etapas", nargs="+", choices=ETAPAS, default=list(ETAPAS))
    p.add_argument("--fuentes", nargs="+", choices=FUENTES, default=list(FUENTES))
    p.add_argument("--subir-hf", action="store_true", help="Subir los originales a Hugging Face")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    configurar_logging(logging.DEBUG if args.verbose else logging.INFO)

    if args.hasta < args.desde:
        p.error("--hasta debe ser posterior a --desde")

    ok = True
    for etapa in args.etapas:
        if etapa == "ingesta":
            ok &= etapa_ingesta(args.desde, args.hasta, args.fuentes, args.subir_hf)
        else:
            log.warning("Etapa '%s' aún no implementada (pendiente en src/%s)", etapa,
                        {"silver": "features", "gold": "features", "ml": "ml"}[etapa])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
