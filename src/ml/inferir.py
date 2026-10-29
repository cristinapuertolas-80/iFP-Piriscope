"""Inferencia batch real: carga el modelo guardado y escribe predicciones y explicaciones SHAP (HU-08).

Lee `data/gold/dataset_municipio_dia.parquet` (todas las filas, también los días recientes sin etiqueta),
predice D+1 para cada municipio y escribe:

- data/gold/predicciones.parquet   (esquema `Prediccion`)
- data/gold/explicaciones.parquet  (esquema `ExplicacionPrediccion`, top-k variables por predicción)

Con `--ejemplo-dir` genera además figuras SHAP de ejemplo (HTML): una predicción concreta y la
importancia global de variables.

DOMINIO DEL MODELO: si una variable nunca faltó en el entrenamiento (p. ej. el FWI desde 2009) y en un día
falta, el modelo no sabe qué hacer y devuelve valores arbitrarios. Esas filas NO se predicen con ese
modelo: se predicen con `--modelo-respaldo` (p. ej. el modelo sin FWI, versión `<v>-respaldo`) o, si no
se indica, se omiten. Nunca se escriben predicciones fuera del dominio.

SIN INFORMACIÓN: una fila en la que todas las variables (salvo el mes) están vacías solo recibiría la
"climatología" del mes: un valor casi constante cuyo nivel de riesgo lo decide el ruido. Esas filas
tampoco se predicen (`--min-variables`, por defecto 1 variable con dato además del mes).

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

from . import config, predict, risk, splits, train
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


def features_sin_nulos(bundle, todo: pd.DataFrame) -> list[str]:
    """Variables del modelo que NO tuvieron ningún nulo en sus filas de entrenamiento.

    Si en inferencia una de ellas falta, la fila queda fuera del dominio del modelo.
    """
    tr = splits.select_train(todo, getattr(bundle, "anio_desde", None))
    if tr.empty:
        return []
    return [f for f in bundle.features if tr[f].notna().all()]


def con_informacion(df: pd.DataFrame, bundle, minimo: int = 1) -> pd.Series:
    """True en las filas con al menos `minimo` variables con dato (sin contar el mes, que es calendario)."""
    informativas = [f for f in bundle.features if f != "mes"]
    if not informativas:
        return pd.Series(True, index=df.index)
    return df[informativas].notna().sum(axis=1) >= minimo


def separar_por_dominio(todo: pd.DataFrame, bundle, desde: pd.Timestamp, hasta: pd.Timestamp):
    """Filas del rango [desde, hasta] dentro y fuera del dominio del modelo."""
    f = pd.to_datetime(todo[config.COL_FECHA]).dt.normalize()
    sub = todo[(f >= desde) & (f <= hasta)]
    dominio = features_sin_nulos(bundle, todo)
    fuera = sub[dominio].isna().any(axis=1) if dominio else pd.Series(False, index=sub.index)
    return sub[~fuera], sub[fuera]


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
    ap.add_argument("--modelo-respaldo", default=None,
                    help="modelo para las filas fuera del dominio del principal (p. ej. el modelo sin FWI)")
    ap.add_argument("--min-variables", type=int, default=1,
                    help="mínimo de variables con dato (sin contar el mes) para predecir una fila")
    ap.add_argument("--reemplazar", action="store_true", help="borra las salidas existentes antes de escribir")
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
        fechas = pd.to_datetime(todo[config.COL_FECHA]).dt.normalize()
        desde = pd.Timestamp(a.desde) if a.desde else None
        hasta = pd.Timestamp(a.hasta) if a.hasta else None
        if desde is None and hasta is None:
            desde = hasta = fechas.max()
        desde, hasta = desde or fechas.min(), hasta or fechas.max()

        dentro, fuera = separar_por_dominio(todo, bundle, desde, hasta)
        info = con_informacion(dentro, bundle, a.min_variables)
        n_sin_info, dentro = int((~info).sum()), dentro[info]
        preds, expls, principal = [], [], None
        if not dentro.empty:
            principal = predict.build_predictions(dentro, bundle, thr, desde=desde, hasta=hasta)
            preds.append(principal)
            expls.append(shap_explain.build_explanations(principal, todo, bundle, top_k=a.top_k))
        n_respaldo = 0
        if a.modelo_respaldo and not fuera.empty:
            b2, t2 = cargar(a.models_dir, a.modelo_respaldo)
            dom2 = features_sin_nulos(b2, todo)
            ok = fuera[~fuera[dom2].isna().any(axis=1)] if dom2 else fuera
            ok = ok[con_informacion(ok, b2, a.min_variables)]
            if not ok.empty:
                p2 = predict.build_predictions(ok, b2, t2, desde=desde, hasta=hasta,
                                               version_modelo=f"{b2.version}-respaldo")
                preds.append(p2)
                expls.append(shap_explain.build_explanations(p2, todo, b2, top_k=a.top_k))
                n_respaldo = len(p2)
        if not preds:
            raise ValueError("No hay filas predecibles en el rango: todas quedan fuera del dominio del modelo o sin "
                             "datos suficientes (¿falta el FWI o la meteorología de los últimos días?). "
                             "Prueba --modelo-respaldo con el modelo sin FWI")
        pred = pd.concat(preds, ignore_index=True)
        expl = pd.concat(expls, ignore_index=True)
        sin_prediccion = len(fuera) - n_respaldo
        extra = (escribir_ejemplos(a.ejemplo_dir, bundle, principal, expls[0], todo, a.ejemplo_municipio)
                 if a.ejemplo_dir and principal is not None else [])
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if a.reemplazar:
        for ruta in (a.salida_predicciones, a.salida_explicaciones):
            Path(ruta).unlink(missing_ok=True)
    predict.write_predictions(pred, a.salida_predicciones)
    shap_explain.write_explanations(expl, a.salida_explicaciones)
    f = pd.to_datetime(pred["fecha_objetivo"])
    print(f"Modelo {modelo} · {len(pred):,} predicciones para {f.min():%Y-%m-%d} a {f.max():%Y-%m-%d} "
          f"({pred['id_municipio'].nunique()} municipios)")
    ultimo = pred[f == f.max()]
    print("Niveles de riesgo en la última fecha:", ultimo["nivel_riesgo"].value_counts().reindex(config.RISK_LEVELS, fill_value=0).to_dict())
    top = ultimo.nlargest(5, "probabilidad")[["id_municipio", "fecha_objetivo", "probabilidad", "nivel_riesgo", "fwi"]]
    print(top.assign(probabilidad=top["probabilidad"].round(4), fwi=top["fwi"].round(2)).to_string(index=False))
    if n_sin_info:
        print(f"Sin información suficiente (menos de {a.min_variables} variable(s) con dato, sin contar el mes): "
              f"{n_sin_info:,} filas del modelo principal sin predicción")
    if n_respaldo or sin_prediccion:
        print(f"Fuera del dominio del modelo principal (falta alguna variable que nunca faltó al entrenar): "
              f"{len(fuera):,} filas -> {n_respaldo:,} con el modelo de respaldo, {sin_prediccion:,} sin predicción")
    print(f"Escrito: {a.salida_predicciones} y {a.salida_explicaciones}")
    for ruta in extra:
        print(f"Ejemplo: {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
