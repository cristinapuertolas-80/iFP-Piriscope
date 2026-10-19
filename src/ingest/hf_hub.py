"""Archivo de los datos originales en el dataset de Hugging Face (DEC-11).

Estructura en el dataset (apartado 5.3 del diseño técnico)::

    raw/
    ├── AEMET/<timestamp>/...
    ├── EGIF/<timestamp>/...
    ├── EFFIS/<timestamp>/...
    └── INE_CNIG/<timestamp>/...

Cada carpeta ``data/bronze/<fuente>/<timestamp>/`` se sube tal cual a
``raw/<FUENTE>/<timestamp>/``. Tras subirla se deja una marca local
``.subido_hf`` para no repetir la subida.

La autenticación usa ``HF_TOKEN``: un token *fine-grained* con permiso de
escritura sólo sobre el dataset, guardado en los Secrets del repositorio.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .common import ejecuciones_previas, error_red, es_error_red

log = logging.getLogger("piriscope.ingest.hf")

MARCA_SUBIDO = ".subido_hf"


def ruta_en_repo(fuente: str, timestamp: str, cfg: config.HuggingFaceConfig | None = None) -> str:
    cfg = cfg or config.HuggingFaceConfig()
    if fuente not in config.FUENTES:
        raise ValueError(f"Fuente desconocida: {fuente}")
    return f"{cfg.prefijo_raw}/{config.FUENTES[fuente]}/{timestamp}"


def _api(cfg: config.HuggingFaceConfig):
    from huggingface_hub import HfApi

    if not cfg.token:
        raise RuntimeError("Falta HF_TOKEN para subir datos a Hugging Face")
    return HfApi(token=cfg.token)


def subir_ejecucion(run_dir: Path, cfg: config.HuggingFaceConfig | None = None, api=None) -> str:
    """Sube una carpeta de ejecución de Bronze al dataset. Devuelve la ruta en el repo."""
    cfg = cfg or config.HuggingFaceConfig()
    run_dir = Path(run_dir)
    fuente, timestamp = run_dir.parent.name, run_dir.name
    destino = ruta_en_repo(fuente, timestamp, cfg)
    api = api or _api(cfg)

    manifiesto = {}
    if (run_dir / "manifest.json").exists():
        manifiesto = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    mensaje = (f"Ingesta {config.FUENTES[fuente]} {timestamp}: "
               f"{manifiesto.get('n_ficheros', '?')} ficheros ({manifiesto.get('estado', '?')})")

    log.info("Subiendo %s -> hf://datasets/%s/%s", run_dir, cfg.repo_id, destino)
    try:
        api.upload_folder(
            folder_path=str(run_dir),
            path_in_repo=destino,
            repo_id=cfg.repo_id,
            repo_type=cfg.repo_type,
            commit_message=mensaje,
            ignore_patterns=[MARCA_SUBIDO],
        )
    except Exception as exc:  # noqa: BLE001
        if es_error_red(exc):
            raise error_red("Hugging Face", exc) from exc
        raise
    (run_dir / MARCA_SUBIDO).write_text(
        json.dumps({"repo_id": cfg.repo_id, "ruta": destino,
                    "subido_utc": datetime.now(timezone.utc).isoformat()}),
        encoding="utf-8",
    )
    return destino


def subir_pendientes(fuente: str | None = None, base_dir: Path | None = None,
                     cfg: config.HuggingFaceConfig | None = None, api=None) -> list[str]:
    """Sube todas las ejecuciones locales que aún no estén en Hugging Face."""
    cfg = cfg or config.HuggingFaceConfig()
    fuentes = [fuente] if fuente else list(config.FUENTES)
    subidas = []
    for f in fuentes:
        for run_dir in sorted(ejecuciones_previas(f, base_dir)):
            if (run_dir / MARCA_SUBIDO).exists() or not (run_dir / "manifest.json").exists():
                continue
            api = api or _api(cfg)
            subidas.append(subir_ejecucion(run_dir, cfg, api))
    return subidas


def descargar(fuente: str, timestamp: str | None = None, destino: Path | None = None,
              cfg: config.HuggingFaceConfig | None = None) -> Path:
    """Descarga de Hugging Face los datos originales de una fuente (todas las
    ejecuciones o sólo ``timestamp``) para analizarlos en local con pandas."""
    from huggingface_hub import snapshot_download

    cfg = cfg or config.HuggingFaceConfig()
    patron = f"{ruta_en_repo(fuente, timestamp, cfg)}/**" if timestamp else \
        f"{cfg.prefijo_raw}/{config.FUENTES[fuente]}/**"
    ruta = snapshot_download(
        repo_id=cfg.repo_id,
        repo_type=cfg.repo_type,
        allow_patterns=[patron],
        local_dir=str(destino or config.DATA_DIR / "hf"),
        token=cfg.token,
    )
    return Path(ruta)
