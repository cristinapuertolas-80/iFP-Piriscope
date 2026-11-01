"""Descarga de los modelos versionados desde el dataset de Hugging Face (DEC-44).

Los ``.joblib`` no se suben al repositorio de GitHub: se publican en ``models/`` del dataset
``cpuertolas/piriscope`` junto a sus metadatos (``.json``) y umbrales (``_umbrales.json``). Este módulo
los baja a ``models/`` en local para poder ejecutar la inferencia sin reentrenar::

    python -m src.ml.modelos_hf                 # versión v1
    python -m src.ml.modelos_hf --version v2

Requiere ``HF_TOKEN`` con lectura sobre el dataset (es de acceso restringido). Los ``.joblib`` son pickles:
solo se descargan del dataset del propio proyecto.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from src.ingest import config as ingest_config
from src.ingest.common import configurar_logging

from . import config

log = logging.getLogger("piriscope.ml.modelos_hf")

RUTA_MODELOS_EN_HF = "models"
#: Ficheros que forman un modelo guardado (ver ``train.save_model`` y ``risk.save_thresholds``).
SUFIJOS = (".joblib", ".json", "_umbrales.json")


def descargar_modelos(version: str = "v1", destino: Path | None = None,
                      cfg: ingest_config.HuggingFaceConfig | None = None, descargar=None) -> list[Path]:
    """Copia a `destino` (por defecto `models/`) los ficheros de los modelos de `version`.

    Devuelve las rutas copiadas. Falla si el dataset no tiene ningún modelo completo de esa versión.
    """
    cfg = cfg or ingest_config.HuggingFaceConfig()
    destino = Path(destino or config.MODELS_DIR)
    if descargar is None:
        from huggingface_hub import snapshot_download as descargar

    cache = Path(descargar(repo_id=cfg.repo_id, repo_type=cfg.repo_type, token=cfg.token,
                           allow_patterns=[f"{RUTA_MODELOS_EN_HF}/{version}_*"]))
    origen = cache / RUTA_MODELOS_EN_HF
    ficheros = sorted(p for p in origen.glob(f"{version}_*") if p.is_file() and p.name.endswith(SUFIJOS))
    ids = sorted(p.name[:-len("_umbrales.json")] for p in ficheros if p.name.endswith("_umbrales.json")
                 and (origen / p.name.replace("_umbrales.json", ".joblib")).exists())
    if not ids:
        raise FileNotFoundError(f"No hay modelos completos de la versión {version!r} en "
                                f"hf://datasets/{cfg.repo_id}/{RUTA_MODELOS_EN_HF}/")
    destino.mkdir(parents=True, exist_ok=True)
    copiados = []
    for p in ficheros:
        shutil.copy2(p, destino / p.name)
        copiados.append(destino / p.name)
    log.info("Modelos %s descargados en %s: %s", version, destino, ", ".join(ids))
    return copiados


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.ml.modelos_hf",
                                 description="Descarga de Hugging Face los modelos versionados a models/")
    ap.add_argument("--version", default="v1", help="versión de los modelos (por defecto v1)")
    ap.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    a = ap.parse_args(argv)
    configurar_logging(logging.INFO)
    try:
        descargar_modelos(a.version, a.models_dir)
    except Exception as exc:  # noqa: BLE001 - mensaje claro en lugar de la traza
        log.error("No se pudieron descargar los modelos: %s", exc)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
