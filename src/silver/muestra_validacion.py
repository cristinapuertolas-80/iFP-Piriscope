"""Muestra de partes para la validación manual de HU-04 (``docs/``).

Se genera **una sola vez** y se versiona en git, para que la muestra no cambie
con los datos y lo que se anote en ``validacion_manual`` no se pierda; por eso
no forma parte del pipeline y no sobrescribe el fichero si ya existe::

    python -m src.silver.muestra_validacion
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from src.ingest import config

from . import incendios, municipios

log = logging.getLogger("piriscope.silver.muestra")

FICHERO_MUESTRA = config.REPO_ROOT / "docs" / "incendios_muestra_validacion.csv"


def generar(ruta: Path = FICHERO_MUESTRA, bronze_dir: Path | None = None, hf_dir: Path | None = None) -> Path:
    if ruta.exists():
        raise FileExistsError(f"{ruta} ya existe y puede tener validaciones anotadas: no se sobrescribe. "
                              "Bórralo a mano si de verdad hay que sacar otra muestra.")
    municipio, limites = municipios.procesar(bronze_dir=bronze_dir, hf_dir=hf_dir)
    eventos = incendios.procesar(municipio, limites, bronze_dir=bronze_dir, hf_dir=hf_dir)
    muestra = incendios.muestra_validacion(eventos, municipio)
    muestra.to_csv(ruta, index=False)
    log.info("Muestra de validación (%d partes): %s", len(muestra), ruta)
    return ruta


def main(argv: list[str] | None = None) -> int:
    from src.ingest.common import configurar_logging

    configurar_logging()
    try:
        generar()
    except (FileExistsError, FileNotFoundError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
