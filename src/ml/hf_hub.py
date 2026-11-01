"""Publicación de las predicciones de Gold en el dataset de Hugging Face (DEC-52).

El dashboard descarga `predicciones.parquet` y `explicaciones.parquet` de `data/gold/` del dataset
(`src/dashboard/datos.py`). Este módulo los sube **en un único commit**, de modo que nunca queda publicado un
par incoherente, y solo si pasan comprobaciones básicas: ficheros presentes, columnas del contrato, una
predicción por municipio y fecha y las mismas `id_prediccion` en ambos ficheros.

No sube modelos (DEC-44) ni los datasets de entrenamiento. Hace falta un token con **permiso de escritura**
(`HF_TOKEN` en `.env`, el mismo de la ingesta).

Uso:
    python -m src.ml.hf_hub --dry-run      # comprueba y dice qué subiría, sin subir nada
    python -m src.ml.hf_hub                # sube
    python run_pipeline.py --etapas ml --subir-hf   # tras inferir y verificar
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import config

log = logging.getLogger("piriscope.ml.hf")

RUTA_EN_HF = "data/gold"
FICHEROS = ("predicciones.parquet", "explicaciones.parquet")
COLUMNAS_PREDICCIONES = ("id_prediccion", "id_municipio", "fecha_objetivo", "version_modelo", "probabilidad",
                         "nivel_riesgo")
COLUMNAS_EXPLICACIONES = ("id_prediccion", "variable", "valor_shap")
COLUMNAS_DEC50 = ("valor_variable", "valor_base", "shap_resto")


def _nombres_columnas(ruta: Path) -> list[str]:
    import pyarrow.parquet as pq

    return pq.read_schema(ruta).names


def _leer_columnas(ruta: Path, columnas: list[str]) -> pd.DataFrame:
    return pd.read_parquet(ruta, columns=list(columnas))


def _configuracion() -> tuple[str, str, str]:
    """(repo_id, repo_type, token) del dataset configurado para la ingesta."""
    from src.ingest import config as ingesta

    cfg = ingesta.HuggingFaceConfig()
    return cfg.repo_id, cfg.repo_type, (cfg.token or "").strip()


def _motivo(exc: Exception) -> str:
    texto = str(exc)
    if "401" in texto or "403" in texto:
        return "acceso denegado: el token necesita permiso de escritura sobre el dataset"
    return f"{type(exc).__name__}: {texto[:200]}"


def comprobar_gold(gold_dir: Path | None = None) -> dict:
    """Valida el par de ficheros y devuelve un resumen. Lanza FileNotFoundError o ValueError si no es publicable."""
    gold_dir = Path(gold_dir or config.GOLD_DIR)
    rutas = {n: gold_dir / n for n in FICHEROS}
    for ruta in rutas.values():
        if not ruta.exists() or ruta.stat().st_size == 0:
            raise FileNotFoundError(f"Falta {ruta} (se generan con `run_pipeline.py --etapas ml`)")
    cols_p = _nombres_columnas(rutas["predicciones.parquet"])
    cols_e = _nombres_columnas(rutas["explicaciones.parquet"])
    for nombre, cols, requeridas in (("predicciones", cols_p, COLUMNAS_PREDICCIONES),
                                     ("explicaciones", cols_e, COLUMNAS_EXPLICACIONES)):
        faltan = [c for c in requeridas if c not in cols]
        if faltan:
            raise ValueError(f"{nombre}.parquet no tiene las columnas {faltan}")

    p = _leer_columnas(rutas["predicciones.parquet"], ["id_prediccion", "id_municipio", "fecha_objetivo"])
    e = _leer_columnas(rutas["explicaciones.parquet"], ["id_prediccion"])
    if p.empty:
        raise ValueError("predicciones.parquet está vacío")
    duplicadas = int(p.duplicated(["id_municipio", "fecha_objetivo"]).sum())
    if duplicadas:
        raise ValueError(f"predicciones.parquet tiene {duplicadas:,} predicciones duplicadas para un mismo "
                         "municipio y fecha")
    ids_p, ids_e = set(p["id_prediccion"]), set(e["id_prediccion"])
    if ids_p != ids_e:
        raise ValueError(f"Las explicaciones no coinciden con las predicciones ({len(ids_p - ids_e):,} "
                         f"predicciones sin explicación y {len(ids_e - ids_p):,} explicaciones sin predicción)")
    fechas = pd.to_datetime(p["fecha_objetivo"])
    return {"rutas": rutas, "n_predicciones": len(p), "n_explicaciones": len(e),
            "fecha_min": fechas.min(), "fecha_max": fechas.max(),
            "dec50": all(c in cols_e for c in COLUMNAS_DEC50)}


def subir_gold(gold_dir: Path | None = None, repo_id: str | None = None, repo_type: str | None = None,
               token: str | None = None, dry_run: bool = False, api=None) -> str | None:
    """Sube predicciones y explicaciones en un solo commit. Devuelve la URL del commit (None en `dry_run`).

    `api` permite inyectar un cliente (tests); por defecto se usa `huggingface_hub.HfApi`.
    """
    resumen = comprobar_gold(gold_dir)
    if not resumen["dec50"]:
        log.warning("Las explicaciones no tienen las columnas de DEC-50 (%s): el dashboard no podrá mostrar los "
                    "valores de las variables. Regenera con `inferir --reemplazar`.", ", ".join(COLUMNAS_DEC50))
    cfg_repo, cfg_tipo, cfg_token = _configuracion() if not (repo_id and repo_type and token) else ("", "", "")
    repo_id, repo_type, token = repo_id or cfg_repo, repo_type or cfg_tipo, (token or cfg_token).strip()
    f_min, f_max = resumen["fecha_min"], resumen["fecha_max"]
    log.info("Gold a publicar en %s (%s/%s): %s predicciones de %s a %s", repo_id, repo_type, RUTA_EN_HF,
             f"{resumen['n_predicciones']:,}".replace(",", "."), f_min.date(), f_max.date())
    if dry_run:
        log.info("Simulación: no se sube nada. Token %s.", "presente" if token else "AUSENTE (hace falta uno de escritura)")
        return None
    if not token:
        raise ValueError("Falta HF_TOKEN con permiso de escritura (en .env o en el entorno)")

    from huggingface_hub import CommitOperationAdd, HfApi

    operaciones = [CommitOperationAdd(path_in_repo=f"{RUTA_EN_HF}/{nombre}", path_or_fileobj=str(ruta))
                   for nombre, ruta in resumen["rutas"].items()]
    mensaje = (f"Gold: {resumen['n_predicciones']:,} predicciones ({f_min.date()} a {f_max.date()})"
               .replace(",", "."))
    api = api or HfApi(token=token)
    try:
        info = api.create_commit(repo_id=repo_id, repo_type=repo_type, operations=operaciones,
                                 commit_message=mensaje)
    except Exception as exc:  # noqa: BLE001 - se traduce a un mensaje comprensible
        raise RuntimeError(f"No se pudo subir a Hugging Face ({_motivo(exc)})") from exc
    return str(getattr(info, "commit_url", info))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Publica predicciones y explicaciones de Gold en Hugging Face")
    ap.add_argument("--gold-dir", type=Path, default=config.GOLD_DIR)
    ap.add_argument("--dry-run", action="store_true", help="solo comprobar y mostrar qué se subiría")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        url = subir_gold(a.gold_dir, dry_run=a.dry_run)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if url:
        print(f"Publicado: {url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
