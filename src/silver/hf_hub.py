"""Intercambio de la capa Silver con el dataset de Hugging Face (DEC-11).

* :func:`descargar_bronze` baja ``raw/<FUENTE>/`` a ``data/hf/raw`` para que
  Silver pueda ejecutarse sin haber hecho la ingesta en local.
* :func:`subir_silver` sube ``data/silver`` a ``data/silver/`` del dataset.
  Requiere un ``HF_TOKEN`` con permiso de **escritura** sobre el dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.ingest import config
from src.ingest import hf_hub as ingest_hf
from src.ingest.common import error_red, es_error_red

log = logging.getLogger("piriscope.silver.hf")

#: Fuentes de Bronze que necesita cada tabla Silver.
FUENTES_POR_TABLA = {
    "municipios": ("ine_cnig",),
    "meteo": ("aemet",),
    "fwi": ("effis",),
    "incendios": ("egif",),
}
RUTA_SILVER_EN_REPO = "data/silver"


def descargar_bronze(tablas: list[str], cfg: config.HuggingFaceConfig | None = None) -> None:
    fuentes = sorted({f for t in tablas for f in FUENTES_POR_TABLA[t]})
    for fuente in fuentes:
        log.info("Descargando raw/%s de Hugging Face…", config.FUENTES[fuente])
        try:
            ingest_hf.descargar(fuente, cfg=cfg)
        except Exception as exc:  # noqa: BLE001
            if es_error_red(exc):
                raise error_red("Hugging Face", exc) from exc
            raise


def subir_silver(base_dir: Path | None = None, cfg: config.HuggingFaceConfig | None = None, api=None) -> str:
    cfg = cfg or config.HuggingFaceConfig()
    carpeta = Path(base_dir or config.SILVER_DIR)
    api = api or ingest_hf._api(cfg)
    log.info("Subiendo %s -> hf://datasets/%s/%s", carpeta, cfg.repo_id, RUTA_SILVER_EN_REPO)
    try:
        api.upload_folder(folder_path=str(carpeta), path_in_repo=RUTA_SILVER_EN_REPO, repo_id=cfg.repo_id,
                          repo_type=cfg.repo_type, commit_message="Silver: actualización de tablas e informes",
                          ignore_patterns=[".gitkeep"])
    except Exception as exc:  # noqa: BLE001
        if "403" in str(exc) or "401" in str(exc):
            raise RuntimeError(
                f"Hugging Face rechaza la subida a {cfg.repo_id}: el HF_TOKEN no tiene permiso de "
                "escritura sobre ese dataset. Un token 'write' personal sólo vale para tus propios "
                "repos; hace falta un token fine-grained del dueño con escritura sobre el dataset, "
                f"o ser miembro con rol write de la organización.\nDetalle: {exc}"
            ) from exc
        if es_error_red(exc):
            raise error_red("Hugging Face", exc) from exc
        raise
    return RUTA_SILVER_EN_REPO
