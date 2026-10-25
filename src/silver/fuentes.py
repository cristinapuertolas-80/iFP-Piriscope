"""Localización de las ejecuciones de Bronze que lee Silver.

Silver no descarga nada de las fuentes externas: lee las carpetas de ejecución
que deja la ingesta, ``data/bronze/<fuente>/<timestamp>/``, o su copia
descargada de Hugging Face, ``data/hf/raw/<FUENTE>/<timestamp>/`` (lo que deja
``python -m src.ingest descargar <fuente>``). Si un mismo timestamp está en los
dos sitios, prevalece la copia local de Bronze.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.ingest import config

log = logging.getLogger("piriscope.silver")

#: Copia local del prefijo ``raw/`` del dataset de Hugging Face.
HF_RAW_DIR = config.DATA_DIR / "hf" / config.HuggingFaceConfig().prefijo_raw


def raices(fuente: str, bronze_dir: Path | None = None, hf_dir: Path | None = None) -> list[Path]:
    """Carpetas donde pueden estar las ejecuciones de ``fuente``, por prioridad."""
    if fuente not in config.FUENTES:
        raise ValueError(f"Fuente desconocida: {fuente!r}")
    return [Path(bronze_dir or config.BRONZE_DIR) / fuente,
            Path(hf_dir or HF_RAW_DIR) / config.FUENTES[fuente]]


def ejecuciones(fuente: str, bronze_dir: Path | None = None, hf_dir: Path | None = None) -> list[Path]:
    """Carpetas de ejecución de ``fuente``, de la más antigua a la más reciente."""
    por_timestamp: dict[str, Path] = {}
    for raiz in reversed(raices(fuente, bronze_dir, hf_dir)):  # Bronze local sobrescribe a HF
        if not raiz.exists():
            continue
        for run in raiz.iterdir():
            if run.is_dir():
                por_timestamp[run.name] = run
    return [por_timestamp[t] for t in sorted(por_timestamp)]


def ficheros(fuente: str, patron: str, bronze_dir: Path | None = None,
             hf_dir: Path | None = None) -> list[tuple[str, Path]]:
    """``(timestamp, ruta)`` de los ficheros que cumplen ``patron`` en todas las ejecuciones."""
    salida = []
    for run in ejecuciones(fuente, bronze_dir, hf_dir):
        salida.extend((run.name, p) for p in sorted(run.glob(patron)) if p.is_file())
    return salida


def ultimo_fichero(fuente: str, patron: str, bronze_dir: Path | None = None,
                   hf_dir: Path | None = None) -> Path | None:
    """Fichero que cumple ``patron`` en la ejecución más reciente que lo contenga."""
    encontrados = ficheros(fuente, patron, bronze_dir, hf_dir)
    return encontrados[-1][1] if encontrados else None


def exigir(ruta: Path | None, fuente: str, descripcion: str) -> Path:
    if ruta is None:
        raise FileNotFoundError(
            f"No se encuentra {descripcion} de {fuente} en data/bronze/{fuente}/ ni en "
            f"data/hf/raw/{config.FUENTES[fuente]}/. Ejecuta la ingesta o descárgalo con "
            f"'python -m src.silver --descargar-hf'."
        )
    return ruta
