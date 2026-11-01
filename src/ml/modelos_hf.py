"""Publicación y descarga de modelos versionados en el dataset de Hugging Face (DEC-44).

Los modelos se publican en ``models/`` del dataset junto a todos sus metadatos
(``.json``) y umbrales (``_umbrales.json``). La subida se realiza en un único
commit para evitar versiones incompletas.

Uso:
    python -m src.ml.modelos_hf --subir --version v1 --dry-run
    python -m src.ml.modelos_hf --subir --version v1
    python -m src.ml.modelos_hf --version v1

Requiere ``HF_TOKEN`` con permiso de escritura para subir y lectura para descargar.
Los ``.joblib`` son pickles: solo se descargan del dataset del propio proyecto.
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
#: Extensiones de los ficheros que forman parte de una versión de modelo.
SUFIJOS = (".joblib", ".json", "_umbrales.json")


def _configuracion() -> tuple[str, str, str]:
    """Devuelve (repo_id, repo_type, token) del dataset configurado para la ingesta."""
    cfg = ingest_config.HuggingFaceConfig()
    return cfg.repo_id, cfg.repo_type, (cfg.token or "").strip()


def _motivo(exc: Exception) -> str:
    texto = str(exc)
    if "401" in texto or "403" in texto:
        return "acceso denegado: el token necesita permiso sobre el dataset"
    return f"{type(exc).__name__}: {texto[:200]}"


def _ficheros_version(version: str, directorio: Path) -> list[Path]:
    """Localiza los artefactos de una versión sin incluir subdirectorios ni otras versiones."""
    return sorted(
        p for p in directorio.glob(f"{version}_*")
        if p.is_file() and p.name.endswith(SUFIJOS) and p.stat().st_size > 0
    )


def comprobar_modelos(version: str = "v1", models_dir: Path | None = None) -> dict:
    """Comprueba que existen los artefactos locales de la versión y devuelve sus rutas.

    Cada ``.joblib`` debe tener su correspondiente ``_umbrales.json``. Los JSON
    adicionales (por ejemplo, metadatos) se incluyen también en la publicación.
    """
    models_dir = Path(models_dir or config.MODELS_DIR)
    ficheros = _ficheros_version(version, models_dir)
    joblibs = [p for p in ficheros if p.name.endswith(".joblib")]
    if not joblibs:
        raise FileNotFoundError(
            f"No hay modelos .joblib de la versión {version!r} en {models_dir}"
        )

    faltan = [
        p.name[:-len(".joblib")] + "_umbrales.json"
        for p in joblibs
        if not (models_dir / (p.name[:-len(".joblib")] + "_umbrales.json")).is_file()
    ]
    if faltan:
        raise FileNotFoundError(
            "Faltan los ficheros de umbrales para estos modelos: " + ", ".join(faltan)
        )

    return {"rutas": ficheros, "n_modelos": len(joblibs)}


def subir_modelos(
    version: str = "v1",
    models_dir: Path | None = None,
    repo_id: str | None = None,
    repo_type: str | None = None,
    token: str | None = None,
    dry_run: bool = False,
    api=None,
) -> str | None:
    """Publica todos los artefactos de la versión en un único commit.

    ``api`` permite inyectar un cliente compatible con ``HfApi`` en los tests.
    Devuelve la URL del commit, o ``None`` cuando se usa ``dry_run``.
    """
    resumen = comprobar_modelos(version, models_dir)
    cfg_repo, cfg_tipo, cfg_token = _configuracion() if not (repo_id and repo_type and token) else ("", "", "")
    repo_id = repo_id or cfg_repo
    repo_type = repo_type or cfg_tipo
    token = (token or cfg_token).strip()

    nombres = [p.name for p in resumen["rutas"]]
    log.info(
        "Modelos a publicar en %s (%s/%s): %s",
        repo_id, repo_type, RUTA_MODELOS_EN_HF, ", ".join(nombres)
    )
    if dry_run:
        log.info(
            "Simulación: no se sube nada. Token %s.",
            "presente" if token else "AUSENTE (hace falta uno de escritura)"
        )
        return None
    if not token:
        raise ValueError("Falta HF_TOKEN con permiso de escritura (en .env o en el entorno)")

    from huggingface_hub import CommitOperationAdd, HfApi

    operaciones = [
        CommitOperationAdd(
            path_in_repo=f"{RUTA_MODELOS_EN_HF}/{ruta.name}",
            path_or_fileobj=str(ruta),
        )
        for ruta in resumen["rutas"]
    ]
    mensaje = f"Modelos {version}: publicación de {resumen['n_modelos']} modelo(s)"
    api = api or HfApi(token=token)
    try:
        info = api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=operaciones,
            commit_message=mensaje,
        )
    except Exception as exc:  # noqa: BLE001 - mensaje comprensible para el usuario
        raise RuntimeError(f"No se pudieron subir los modelos a Hugging Face ({_motivo(exc)})") from exc
    return str(getattr(info, "commit_url", info))


def descargar_modelos(
    version: str = "v1",
    destino: Path | None = None,
    cfg: ingest_config.HuggingFaceConfig | None = None,
    descargar=None,
) -> list[Path]:
    """Copia a ``destino`` (por defecto ``models/``) los ficheros de una versión."""
    cfg = cfg or ingest_config.HuggingFaceConfig()
    destino = Path(destino or config.MODELS_DIR)
    if descargar is None:
        from huggingface_hub import snapshot_download as descargar

    cache = Path(descargar(
        repo_id=cfg.repo_id,
        repo_type=cfg.repo_type,
        token=cfg.token,
        allow_patterns=[f"{RUTA_MODELOS_EN_HF}/{version}_*"],
    ))
    origen = cache / RUTA_MODELOS_EN_HF
    ficheros = sorted(
        p for p in origen.glob(f"{version}_*")
        if p.is_file() and p.name.endswith(SUFIJOS)
    )
    joblibs = [p for p in ficheros if p.name.endswith(".joblib")]
    ids_completos = [
        p.name[:-len(".joblib")]
        for p in joblibs
        if (origen / (p.name[:-len(".joblib")] + "_umbrales.json")).is_file()
    ]
    if not ids_completos:
        raise FileNotFoundError(
            f"No hay modelos completos de la versión {version!r} en "
            f"hf://datasets/{cfg.repo_id}/{RUTA_MODELOS_EN_HF}/"
        )

    destino.mkdir(parents=True, exist_ok=True)
    copiados = []
    for ruta in ficheros:
        shutil.copy2(ruta, destino / ruta.name)
        copiados.append(destino / ruta.name)
    log.info("Modelos %s descargados en %s: %s", version, destino, ", ".join(ids_completos))
    return copiados


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m src.ml.modelos_hf",
        description="Publica o descarga los modelos versionados de Hugging Face",
    )
    ap.add_argument("--version", default="v1", help="versión de los modelos (por defecto v1)")
    ap.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    ap.add_argument("--subir", action="store_true", help="publicar los artefactos locales en Hugging Face")
    ap.add_argument("--dry-run", action="store_true", help="con --subir, validar y mostrar sin publicar")
    a = ap.parse_args(argv)

    configurar_logging(logging.INFO)
    try:
        if a.subir:
            url = subir_modelos(a.version, a.models_dir, dry_run=a.dry_run)
            if url:
                print(f"Publicado: {url}")
        else:
            if a.dry_run:
                ap.error("--dry-run solo se puede utilizar junto con --subir")
            descargar_modelos(a.version, a.models_dir)
    except Exception as exc:  # noqa: BLE001 - mensaje claro en lugar de la traza
        log.error("No se pudieron procesar los modelos: %s", exc)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
