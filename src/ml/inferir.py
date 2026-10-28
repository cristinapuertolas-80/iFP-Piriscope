"""Inferencia batch real: carga el modelo guardado y escribe predicciones y explicaciones SHAP (HU-08).

Lee `data/gold/dataset_municipio_dia.parquet` (todas las filas, también los días recientes sin etiqueta),
predice D+1 para cada municipio y escribe:

- data/gold/predicciones.parquet   (esquema `Prediccion`)
- data/gold/explicaciones.parquet  (esquema `ExplicacionPrediccion`, top-k variables por predicción)

Con `--ejemplo-dir` genera además figuras SHAP de ejemplo (HTML): una predicción concreta y la
importancia global de variables.

OJO: para fechas dentro del entrenamiento o la validación (hasta 2022) las predicciones NO son
out-of-sample; sirven para demostrar el flujo, no para evaluar el modelo.

Uso:
    python -m src.ml.inferir                                   # último día disponible
    python -m src.ml.inferir --desde 2026-09-01 --hasta 2026-10-20 --ejemplo-dir docs/resultados
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import config, predict, risk, train
from ..explainability import plots, shap_explain


def listar_modelos(models_dir: Path) -> list[str]:
    """Identificadores de los modelos con modelo y umbrales guardados en `models_dir`."""
    sufijo = "_umbrales.json"
    return sorted(p.name[:-len(sufijo)] for p in Path(models_dir).glob(f"*{sufijo}")
                  if (Path(models_dir) / f"{p.name[:-len(sufijo)]}.joblib").exists())


def cargar(models_dir: Path, model_id: str):
    """Devuelve (ModelBundle, RiskThresholds) de un modelo guardado."""
    d = Path(models_dir)
    modelo, umbrales = d / f"{model_id}.joblib", d / f"{model_id}_umbrales.json"
    if not modelo.exists() or not umbrales.exists():
        raise FileNotFoundError(f"No existe el modelo {model_id!r} en {d}. Disponibles: {listar_modelos(d) or 'ninguno'}")
    return train.load_model(modelo), risk.load_thresholds(umbrales)


def escribir_ejemplos(dir_: Path, bundle, pred: pd.DataFrame, expl: pd.DataFrame, todo: pd.DataFrame,
                      municipio: str | None) -> list[Path]:
    """Figuras SHAP de ejemplo (HTML) y la importancia global en CSV."""
    dir_.mkdir(parents=True, exist_ok=True)
    ultimo = pred[pred["fecha_objetivo"] == pred["fecha_objetivo"].max()]
    if municipio:
        ultimo = ultimo[ultimo["id_municipio"] == municipio]
        if ultimo.empty:
            raise ValueError(f"El municipio {municipio} no tiene predicción en la última fecha")
    fila = ultimo.sort_values("probabilidad", ascending=False).iloc[0]
    unidades = shap_explain.shap_units(bundle)
    salidas = []
    ruta = dir_ / f"shap_local_{fila['id_municipio']}_{pd.Timestamp(fila['fecha_objetivo']):%Y-%m-%d}.html"
    plots.fig_local(expl, fila["id_prediccion"], unidades).write_html(ruta, include_plotlyjs="cdn")
    salidas.append(ruta)

    val = todo[(todo[config.COL_PARTICION] == config.PART_VAL) & (todo[config.COL_RESERVADO] == 0)]
    if not val.empty:
        imp = shap_explain.global_importance(bundle, val)
        imp.to_csv(dir_ / "shap_global.csv", index=False)
        plots.fig_global(imp).write_html(dir_ / "shap_global.html", include_plotlyjs="cdn")
        salidas += [dir_ / "shap_global.csv", dir_ / "shap_global.html"]
    return salidas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inferencia batch D+1 y explicaciones SHAP con un modelo guardado")
    ap.add_argument("--modelo", default=None, help="id del modelo (p. ej. v1_xgboost_con_fwi-sin_viento-desde2009)")
    ap.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    ap.add_argument("--dataset", type=Path, default=config.GOLD_DIR / "dataset_municipio_dia.parquet")
    ap.add_argument("--desde", default=None, help="primer día D (por defecto, el último disponible)")
    ap.add_argument("--hasta", default=None, help="último día D")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--salida-predicciones", type=Path, default=config.PREDICCIONES_PARQUET)
    ap.add_argument("--salida-explicaciones", type=Path, default=config.EXPLICACIONES_PARQUET)
    ap.add_argument("--ejemplo-dir", type=Path, default=None, help="carpeta para figuras SHAP de ejemplo")
    ap.add_argument("--ejemplo-municipio", default=None, help="municipio del ejemplo (por defecto, el de mayor riesgo)")
    a = ap.parse_args(argv)

    try:
        ids = listar_modelos(a.models_dir)
        modelo = a.modelo or (ids[0] if len(ids) == 1 else None)
        if modelo is None:
            raise ValueError(f"Indica --modelo. Disponibles: {ids or 'ninguno (ejecuta experimento.py --guardar)'}")
        bundle, thr = cargar(a.models_dir, modelo)
        todo = pd.read_parquet(a.dataset)
        pred = predict.build_predictions(todo, bundle, thr, desde=a.desde, hasta=a.hasta)
        expl = shap_explain.build_explanations(pred, todo, bundle, top_k=a.top_k)
        extra = escribir_ejemplos(a.ejemplo_dir, bundle, pred, expl, todo, a.ejemplo_municipio) if a.ejemplo_dir else []
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    predict.write_predictions(pred, a.salida_predicciones)
    shap_explain.write_explanations(expl, a.salida_explicaciones)
    f = pd.to_datetime(pred["fecha_objetivo"])
    print(f"Modelo {modelo} · {len(pred):,} predicciones para {f.min():%Y-%m-%d} a {f.max():%Y-%m-%d} "
          f"({pred['id_municipio'].nunique()} municipios)")
    ultimo = pred[f == f.max()]
    print("Niveles de riesgo en la última fecha:", ultimo["nivel_riesgo"].value_counts().reindex(config.RISK_LEVELS, fill_value=0).to_dict())
    top = ultimo.nlargest(5, "probabilidad")[["id_municipio", "fecha_objetivo", "probabilidad", "nivel_riesgo", "fwi"]]
    print(top.assign(probabilidad=top["probabilidad"].round(4), fwi=top["fwi"].round(2)).to_string(index=False))
    print(f"Escrito: {a.salida_predicciones} y {a.salida_explicaciones}")
    for ruta in extra:
        print(f"Ejemplo: {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
