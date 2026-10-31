"""Orquestador único de Piriscope (apartado 5.8 del diseño técnico).

Encadena ingesta → Silver → Gold → entrenamiento/inferencia, sin editar código (HU-11).
Etapas: ``ingesta`` (``src/ingest``), ``silver`` (``src/silver``), ``gold`` (``src/features``) y
``ml`` (``src/ml``: entrenamiento con ``experimento.py`` e inferencia con ``inferir.py``).

``--desde`` y ``--hasta`` delimitan la ingesta y Silver. Gold y ML trabajan sobre todo el histórico
disponible; la inferencia predice el último día disponible salvo que se indique ``--inferir-desde/--inferir-hasta``.

Si una etapa de Silver, Gold o ML falla, el pipeline se detiene (``--continuar`` lo evita).

Ejemplos::

    python run_pipeline.py --desde 2020-01-01 --hasta 2024-12-31 --etapas ingesta silver
    python run_pipeline.py --desde 2024-01-01 --hasta 2024-12-31 --fuentes aemet effis --subir-hf
    python run_pipeline.py --etapas gold ml                                  # Gold + predicciones del último día (modelo v1)
    python run_pipeline.py --etapas ml --inferir-desde 2026-09-01 --inferir-hasta 2026-10-20
    python run_pipeline.py --etapas gold ml --ml-acciones entrenar inferir --version-modelo v2   # reentrenar
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path

from src.ingest.common import configurar_logging

log = logging.getLogger("piriscope.pipeline")

ETAPAS = ("ingesta", "silver", "gold", "ml")
FUENTES = ("ine_cnig", "egif", "aemet", "effis")
ACCIONES_ML = ("entrenar", "inferir")
BLOQUEANTES = {"silver", "gold", "ml"}          # si fallan, las siguientes etapas no tienen sentido

# Configuración del modelo elegida (DEC-47). Cambiarla aquí cambia el nombre del modelo guardado.
ALGORITMO = "xgboost"
VARIANTES = ("con_fwi", "sin_fwi")              # principal y respaldo (DEC-48)
ANIO_DESDE = "2009"
EXCLUIR = ("viento",)
PARAMETROS = ("scale_pos_weight=1",)


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


def etapa_gold() -> bool:
    """Variables derivadas y datasets de Gold (`python -m src.features`)."""
    from src.ml import config

    log.info("=== Gold: python -m src.features ===")
    r = subprocess.run([sys.executable, "-m", "src.features"], check=False)
    if r.returncode != 0:
        log.error("La etapa Gold ha fallado (código %d)", r.returncode)
        return False
    esperados = [config.DATASET_ENTRENAMIENTO, config.GOLD_DIR / "dataset_municipio_dia.parquet"]
    faltan = [str(p) for p in esperados if not Path(p).exists()]
    if faltan:
        log.error("Gold no ha generado: %s", ", ".join(faltan))
        return False
    return True


def resolver_modelos(models_dir: Path, version: str, principal: str | None, respaldo: str | None) -> tuple[str, str]:
    """Ids del modelo principal (con FWI) y de respaldo (sin FWI) de una versión guardada en `models_dir`."""
    from src.ml import inferir

    ids = [i for i in inferir.listar_modelos(models_dir) if i.startswith(f"{version}_")]

    def elegir(rol: str, explicito: str | None, marca: str) -> str:
        if explicito:
            return explicito
        candidatos = [i for i in ids if marca in i]
        if len(candidatos) != 1:
            raise ValueError(f"No encuentro un único modelo {rol} de la versión {version} en {models_dir} "
                             f"(candidatos: {candidatos or 'ninguno'}); indícalo con --modelo / --modelo-respaldo")
        return candidatos[0]

    return elegir("principal", principal, "_con_fwi"), elegir("de respaldo", respaldo, "_sin_fwi")


def verificar_salidas(principal: str, respaldo: str) -> bool:
    """Ejecuta `src/ml/verificar_salidas.py` sobre las salidas de Gold (si existe el script)."""
    from src.ml import config

    script = Path(__file__).resolve().parent / "src" / "ml" / "verificar_salidas.py"
    if not script.exists():
        log.warning("No existe %s; se omite la verificación de las salidas", script)
        return True
    cmd = [sys.executable, str(script), "--pred", str(config.PREDICCIONES_PARQUET),
           "--expl", str(config.EXPLICACIONES_PARQUET), "--dataset", str(config.GOLD_DIR / "dataset_municipio_dia.parquet"),
           "--models-dir", str(config.MODELS_DIR), "--modelo", principal, "--respaldo", respaldo]
    return subprocess.run(cmd, check=False).returncode == 0


def etapa_ml(acciones: list[str], version: str, principal: str | None, respaldo: str | None,
             sobrescribir_modelo: bool, inferir_desde: date | None, inferir_hasta: date | None,
             reemplazar: bool, ejemplo_dir: Path | None, verificar: bool) -> bool:
    from src.ml import config, experimento, inferir

    models_dir = Path(config.MODELS_DIR)
    if "entrenar" in acciones:
        log.info("=== ML: entrenamiento (%s, fase de desarrollo; no toca el test) ===", version)
        existentes = sorted(models_dir.glob(f"{version}_*.joblib"))
        if existentes and not sobrescribir_modelo:
            log.error("Ya existen modelos de la versión %s (%s). Usa otra --version-modelo o --sobrescribir-modelo",
                      version, ", ".join(p.name for p in existentes))
            return False
        argv = ["--algoritmo", ALGORITMO, "--variantes", *VARIANTES, "--anio-desde", ANIO_DESDE,
                "--excluir", *EXCLUIR, "--param", *PARAMETROS, "--version", version, "--guardar",
                "--models-dir", str(models_dir),
                "--salida", str(config.GOLD_DIR / f"informe_desarrollo_{version}.csv")]
        if experimento.main(argv) != 0:
            log.error("El entrenamiento ha fallado")
            return False
    if "inferir" in acciones:
        log.info("=== ML: inferencia batch D+1 y SHAP ===")
        try:
            modelo, modelo_respaldo = resolver_modelos(models_dir, version, principal, respaldo)
        except ValueError as e:
            log.error("%s", e)
            return False
        argv = ["--modelo", modelo, "--modelo-respaldo", modelo_respaldo, "--models-dir", str(models_dir)]
        if inferir_desde:
            argv += ["--desde", inferir_desde.isoformat()]
        if inferir_hasta:
            argv += ["--hasta", inferir_hasta.isoformat()]
        if reemplazar:
            argv.append("--reemplazar")
        if ejemplo_dir:
            argv += ["--ejemplo-dir", str(ejemplo_dir)]
        if inferir.main(argv) != 0:
            log.error("La inferencia ha fallado (revisa los mensajes anteriores)")
            return False
        if verificar and not verificar_salidas(modelo, modelo_respaldo):
            log.error("La verificación de las salidas ha encontrado problemas")
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Pipeline de Piriscope")
    p.add_argument("--desde", type=date.fromisoformat, help="inicio de la ingesta y Silver")
    p.add_argument("--hasta", type=date.fromisoformat, help="fin de la ingesta y Silver")
    p.add_argument("--etapas", nargs="+", choices=ETAPAS, default=list(ETAPAS))
    p.add_argument("--fuentes", nargs="+", choices=FUENTES, default=list(FUENTES))
    p.add_argument("--subir-hf", action="store_true",
                   help="Subir a Hugging Face los originales (ingesta) y data/silver (Silver)")
    ml = p.add_argument_group("etapa ml")
    ml.add_argument("--ml-acciones", nargs="+", choices=ACCIONES_ML, default=["inferir"],
                    help="por defecto solo inferir; añade `entrenar` para reentrenar y guardar un modelo nuevo")
    ml.add_argument("--version-modelo", default="v1", help="versión de los modelos a entrenar o a usar (por defecto v1)")
    ml.add_argument("--modelo", default=None, help="id del modelo principal (por defecto, el `con_fwi` de la versión)")
    ml.add_argument("--modelo-respaldo", default=None, help="id del modelo de respaldo (por defecto, el `sin_fwi`)")
    ml.add_argument("--sobrescribir-modelo", action="store_true",
                    help="permite entrenar encima de una versión de modelo que ya existe")
    ml.add_argument("--inferir-desde", type=date.fromisoformat, help="primer día D a predecir (por defecto, el último disponible)")
    ml.add_argument("--inferir-hasta", type=date.fromisoformat, help="último día D a predecir")
    ml.add_argument("--reemplazar-predicciones", action="store_true",
                    help="empieza de cero: descarta TODO el contenido anterior de predicciones y explicaciones")
    ml.add_argument("--ejemplo-dir", type=Path, default=None, help="carpeta para las figuras SHAP de ejemplo")
    ml.add_argument("--sin-verificar", action="store_true", help="no ejecutar verificar_salidas.py tras la inferencia")
    p.add_argument("--continuar", action="store_true", help="seguir con las etapas siguientes aunque falle Silver, Gold o ML")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    configurar_logging(logging.DEBUG if args.verbose else logging.INFO)

    etapas = [e for e in ETAPAS if e in args.etapas]          # siempre en el orden del pipeline
    if {"ingesta", "silver"} & set(etapas) and not (args.desde and args.hasta):
        p.error("--desde y --hasta son obligatorios para las etapas ingesta y silver")
    if args.desde and args.hasta and args.hasta < args.desde:
        p.error("--hasta debe ser posterior a --desde")
    if args.inferir_desde and args.inferir_hasta and args.inferir_hasta < args.inferir_desde:
        p.error("--inferir-hasta debe ser posterior a --inferir-desde")

    ok = True
    for etapa in etapas:
        if etapa == "ingesta":
            exito = etapa_ingesta(args.desde, args.hasta, args.fuentes, args.subir_hf)
        elif etapa == "silver":
            exito = etapa_silver(args.desde, args.hasta, args.subir_hf)
        elif etapa == "gold":
            exito = etapa_gold()
        else:
            exito = etapa_ml(args.ml_acciones, args.version_modelo, args.modelo, args.modelo_respaldo,
                             args.sobrescribir_modelo, args.inferir_desde, args.inferir_hasta,
                             args.reemplazar_predicciones, args.ejemplo_dir, not args.sin_verificar)
        ok &= exito
        if not exito and etapa in BLOQUEANTES and not args.continuar:
            log.error("La etapa %s ha fallado: se detiene el pipeline (usa --continuar para seguir)", etapa)
            break
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
