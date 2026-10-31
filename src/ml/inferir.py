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
tampoco se predicen (`--min-variables`, por defecto 1 variable con dato además del mes; el respaldo
puede exigir más con `--min-variables-respaldo`).

UNA PREDICCIÓN POR MUNICIPIO Y FECHA: al escribir, las predicciones nuevas sustituyen a las que ya había
para el mismo (municipio, fecha_objetivo), aunque tengan otra `version_modelo`, junto con sus
explicaciones. Así, cuando llega el FWI de unos días que se predijeron con el respaldo, `v1` reemplaza
a `v1-respaldo` sin duplicados y sin borrar el resto del histórico. `--reemplazar` sigue existiendo para
empezar de cero. La escritura se hace en ficheros temporales y solo se sustituyen los reales si todo ha ido bien.

Las figuras de ejemplo se generan DESPUÉS de escribir las salidas: un fallo en ellas no impide la inferencia
(el código de salida es 2, pero las predicciones ya están escritas).

AVISOS (stderr, no cambian el código de salida): cobertura por debajo de `--min-cobertura`, fechas con una
proporción de medio+alto >= `--aviso-alarmas`, variables meteorológicas con muchos nulos en el último día D y
retraso respecto a hoy mayor que `--max-retraso` días (solo en el modo "último día disponible").

OJO: para fechas dentro del entrenamiento o la validación (hasta 2022) las predicciones NO son
out-of-sample; sirven para demostrar el flujo, no para evaluar el modelo.

Uso:
    python -m src.ml.inferir                                   # último día disponible
    python -m src.ml.inferir --desde 2026-09-01 --hasta 2026-10-20 --ejemplo-dir docs/resultados
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from . import config, predict, risk, splits, train
from ..explainability import plots, shap_explain

NIVELES_ALARMA = ("medio", "alto")
UMBRAL_NULOS_AVISO = 0.30


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


def validar_salidas(pred: pd.DataFrame, expl: pd.DataFrame) -> None:
    """Comprobaciones previas a escribir: una predicción por municipio y fecha, y explicaciones coherentes."""
    dup = int(pred.duplicated(["id_municipio", "fecha_objetivo"]).sum())
    if dup:
        raise ValueError(f"{dup} predicciones duplicadas para un mismo municipio y fecha_objetivo")
    if set(expl["id_prediccion"]) != set(pred["id_prediccion"]):
        raise ValueError("Las explicaciones no coinciden con las predicciones generadas")


def _clave(d: pd.DataFrame) -> pd.Series:
    """Clave de sustitución: municipio y fecha_objetivo (sin la versión del modelo)."""
    return d["id_municipio"].astype(str) + "|" + pd.to_datetime(d["fecha_objetivo"]).dt.strftime("%Y-%m-%d")


def escribir_salidas(pred: pd.DataFrame, expl: pd.DataFrame, ruta_pred: Path, ruta_expl: Path,
                     reemplazar: bool = False) -> int:
    """Escribe predicciones y explicaciones; devuelve cuántas predicciones previas se han sustituido.

    Trabaja sobre ficheros temporales y solo al final reemplaza los reales. Salvo `reemplazar`, conserva lo
    anterior excepto las filas con el mismo (municipio, fecha_objetivo) que las nuevas (y sus explicaciones).
    """
    ruta_pred, ruta_expl = Path(ruta_pred), Path(ruta_expl)
    tmp_pred = ruta_pred.with_name(ruta_pred.name + ".tmp")
    tmp_expl = ruta_expl.with_name(ruta_expl.name + ".tmp")
    for t in (tmp_pred, tmp_expl):
        t.parent.mkdir(parents=True, exist_ok=True)
        t.unlink(missing_ok=True)
    sustituidas = 0
    try:
        if not reemplazar and ruta_pred.exists():
            previas = pd.read_parquet(ruta_pred)
            choca = _clave(previas).isin(set(_clave(pred)))
            sustituidas = int(choca.sum())
            previas[~choca].to_parquet(tmp_pred, index=False)
            if ruta_expl.exists():
                e = pd.read_parquet(ruta_expl)
                fuera = set(previas.loc[choca, "id_prediccion"])
                e[~e["id_prediccion"].isin(fuera)].to_parquet(tmp_expl, index=False)
        predict.write_predictions(pred, tmp_pred)
        shap_explain.write_explanations(expl, tmp_expl)
        os.replace(tmp_pred, ruta_pred)
        os.replace(tmp_expl, ruta_expl)
    finally:
        for t in (tmp_pred, tmp_expl):
            t.unlink(missing_ok=True)
    return sustituidas


def calcular_avisos(pred: pd.DataFrame, todo: pd.DataFrame, bundle, d_ult: pd.Timestamp,
                    min_cobertura: float, aviso_alarmas: float) -> list[str]:
    """Avisos de calidad de la ejecución (no detienen nada)."""
    avisos = []
    f = pd.to_datetime(pred["fecha_objetivo"]).dt.normalize()

    n_mun = todo[config.COL_MUNICIPIO].nunique()
    cob = pred.groupby(f)["id_municipio"].nunique() / n_mun
    baja = cob[cob < min_cobertura]
    if not baja.empty:
        fechas = ", ".join(f"{d:%Y-%m-%d} ({c:.0%})" for d, c in baja.head(5).items())
        avisos.append(f"cobertura por debajo del {min_cobertura:.0%} en {len(baja)} fecha(s): {fechas}")

    alarma = pred["nivel_riesgo"].isin(NIVELES_ALARMA).groupby(f).mean()
    alta = alarma[alarma >= aviso_alarmas]
    if not alta.empty:
        fechas = ", ".join(f"{d:%Y-%m-%d} ({p:.0%})" for d, p in alta.head(5).items())
        avisos.append(f"{len(alta)} fecha(s) con >= {aviso_alarmas:.0%} de municipios en medio o alto: {fechas}. "
                      "Si todas son filas del modelo principal (con FWI) suele ser sequía real: los umbrales absolutos "
                      "pierden capacidad de discriminar (DEC-49). Si hay filas de respaldo o con nulos, revisa los datos de entrada")

    ult = todo[pd.to_datetime(todo[config.COL_FECHA]).dt.normalize() == d_ult]
    meteo = [c for c in getattr(config, "METEO_FEATURES", []) if c in bundle.features and c in ult.columns]
    for c in meteo:
        p = float(ult[c].isna().mean()) if len(ult) else 0.0
        if p >= UMBRAL_NULOS_AVISO:
            avisos.append(f"{p:.0%} de los municipios sin {c} el día D {d_ult:%Y-%m-%d}")
    return avisos


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
    fecha_obj = pd.Timestamp(fila["fecha_objetivo"])
    sufijo = f"{fila['id_municipio']}_{fecha_obj:%Y-%m-%d}"
    titulo = (f"Municipio {fila['id_municipio']} · {fecha_obj:%d/%m/%Y} · índice de riesgo "
              f"{100 * fila['probabilidad']:.1f} % ({fila['nivel_riesgo']})".replace(".", ","))
    ampliada = "valor_variable" in expl.columns
    ruta = dir_ / f"shap_local_{sufijo}.html"
    plots.fig_local(expl, fila["id_prediccion"], unidades, titulo=titulo, etiquetas=plots.ETIQUETAS,
                    mostrar_valores=ampliada, incluir_resto="shap_resto" in expl.columns
                    ).write_html(ruta, include_plotlyjs="cdn")
    salidas.append(ruta)
    if "valor_base" in expl.columns:
        ruta_c = dir_ / f"shap_cascada_{sufijo}.html"
        plots.fig_cascada(expl, fila["id_prediccion"], unidades, titulo=titulo, etiquetas=plots.ETIQUETAS
                          ).write_html(ruta_c, include_plotlyjs="cdn")
        salidas.append(ruta_c)

    val = todo[(todo[config.COL_PARTICION] == config.PART_VAL) & (todo[config.COL_RESERVADO] == 0)]
    if not val.empty:
        imp = shap_explain.global_importance(bundle, val)
        imp.to_csv(dir_ / "shap_global.csv", index=False)
        plots.fig_global(imp).write_html(dir_ / "shap_global.html", include_plotlyjs="cdn")
        grupos = shap_explain.global_importance(bundle, val, por_grupo=True)
        grupos.to_csv(dir_ / "shap_global_grupos.csv", index=False)
        plots.fig_global(grupos).write_html(dir_ / "shap_global_grupos.html", include_plotlyjs="cdn")
        salidas += [dir_ / "shap_global.csv", dir_ / "shap_global.html",
                    dir_ / "shap_global_grupos.csv", dir_ / "shap_global_grupos.html"]
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
    ap.add_argument("--min-variables-respaldo", type=int, default=None,
                    help="igual, pero para el modelo de respaldo (por defecto, el valor de --min-variables)")
    ap.add_argument("--reemplazar", action="store_true",
                    help="empieza de cero: descarta TODO el contenido anterior de las salidas")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--min-cobertura", type=float, default=0.8,
                    help="aviso si una fecha tiene predicción para menos de esta fracción de municipios")
    ap.add_argument("--aviso-alarmas", type=float, default=0.95,
                    help="aviso si una fecha tiene al menos esta fracción de municipios en medio o alto")
    ap.add_argument("--max-retraso", type=int, default=3,
                    help="aviso si la última fecha objetivo dista de hoy más de estos días (modo último día)")
    ap.add_argument("--salida-predicciones", type=Path, default=config.PREDICCIONES_PARQUET)
    ap.add_argument("--salida-explicaciones", type=Path, default=config.EXPLICACIONES_PARQUET)
    ap.add_argument("--ejemplo-dir", type=Path, default=None, help="carpeta para figuras SHAP de ejemplo")
    ap.add_argument("--ejemplo-municipio", default=None, help="municipio del ejemplo (por defecto, el de mayor riesgo)")
    a = ap.parse_args(argv)
    min_resp = a.min_variables if a.min_variables_respaldo is None else a.min_variables_respaldo

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
        modo_ultimo_dia = desde is None and hasta is None
        if modo_ultimo_dia:
            desde = hasta = fechas.max()
        desde, hasta = desde or fechas.min(), hasta or fechas.max()

        dentro, fuera = separar_por_dominio(todo, bundle, desde, hasta)
        info = con_informacion(dentro, bundle, a.min_variables)
        n_sin_info, dentro = int((~info).sum()), dentro[info]
        preds, expls, principal = [], [], None
        if not dentro.empty:
            principal = predict.build_predictions(dentro, bundle, thr, desde=desde, hasta=hasta)
            preds.append(principal)
            expls.append(shap_explain.build_explanations(principal, todo, bundle, top_k=a.top_k, verificar=True))
        n_respaldo = 0
        if a.modelo_respaldo and not fuera.empty:
            b2, t2 = cargar(a.models_dir, a.modelo_respaldo)
            dom2 = features_sin_nulos(b2, todo)
            ok = fuera[~fuera[dom2].isna().any(axis=1)] if dom2 else fuera
            ok = ok[con_informacion(ok, b2, min_resp)]
            if not ok.empty:
                p2 = predict.build_predictions(ok, b2, t2, desde=desde, hasta=hasta,
                                               version_modelo=f"{b2.version}-respaldo")
                preds.append(p2)
                expls.append(shap_explain.build_explanations(p2, todo, b2, top_k=a.top_k, verificar=True))
                n_respaldo = len(p2)
        if not preds:
            raise ValueError("No hay filas predecibles en el rango: todas quedan fuera del dominio del modelo o sin "
                             "datos suficientes (¿falta el FWI o la meteorología de los últimos días?). "
                             "Prueba --modelo-respaldo con el modelo sin FWI")
        pred = pd.concat(preds, ignore_index=True)
        expl = pd.concat(expls, ignore_index=True)
        validar_salidas(pred, expl)
        sin_prediccion = len(fuera) - n_respaldo
        n_mun_total = todo[config.COL_MUNICIPIO].nunique()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Primero las salidas de producción; las figuras de ejemplo van después y no pueden bloquearlas.
    sustituidas = escribir_salidas(pred, expl, a.salida_predicciones, a.salida_explicaciones, a.reemplazar)

    f = pd.to_datetime(pred["fecha_objetivo"])
    print(f"Modelo {modelo} · {len(pred):,} predicciones para {f.min():%Y-%m-%d} a {f.max():%Y-%m-%d} "
          f"({pred['id_municipio'].nunique()} municipios)")
    ultimo = pred[f == f.max()]
    print("Niveles de riesgo en la última fecha:", ultimo["nivel_riesgo"].value_counts().reindex(config.RISK_LEVELS, fill_value=0).to_dict())
    print(f"Cobertura en la última fecha: {ultimo['id_municipio'].nunique()} de {n_mun_total} municipios")
    top = ultimo.nlargest(5, "probabilidad")[["id_municipio", "fecha_objetivo", "probabilidad", "nivel_riesgo", "fwi"]]
    print(top.assign(probabilidad=top["probabilidad"].round(4), fwi=top["fwi"].round(2)).to_string(index=False))
    if n_sin_info:
        print(f"Sin información suficiente (menos de {a.min_variables} variable(s) con dato, sin contar el mes): "
              f"{n_sin_info:,} filas del modelo principal sin predicción")
    if n_respaldo or sin_prediccion:
        print(f"Fuera del dominio del modelo principal (falta alguna variable que nunca faltó al entrenar): "
              f"{len(fuera):,} filas -> {n_respaldo:,} con el modelo de respaldo, {sin_prediccion:,} sin predicción")
    if sustituidas:
        print(f"Sustituidas {sustituidas:,} predicciones anteriores del mismo municipio y fecha (p. ej. v1-respaldo por v1)")
    print(f"Escrito: {a.salida_predicciones} y {a.salida_explicaciones}")

    d_ult = min(hasta, fechas.max())
    avisos = calcular_avisos(pred, todo, bundle, d_ult, a.min_cobertura, a.aviso_alarmas)
    if modo_ultimo_dia:
        retraso = (pd.Timestamp.today().normalize() - f.max().normalize()).days
        if retraso > a.max_retraso:
            avisos.append(f"la última fecha objetivo ({f.max():%Y-%m-%d}) va {retraso} días por detrás de hoy "
                          f"(máximo previsto: {a.max_retraso}); la fuente de datos llega con retraso")
    for m in avisos:
        print(f"AVISO: {m}", file=sys.stderr)

    codigo = 0
    if a.ejemplo_dir:
        if principal is None:
            print("AVISO: no hay predicciones del modelo principal en el rango; no se generan figuras de ejemplo",
                  file=sys.stderr)
        else:
            try:
                for ruta in escribir_ejemplos(a.ejemplo_dir, bundle, principal, expls[0], todo, a.ejemplo_municipio):
                    print(f"Ejemplo: {ruta}")
            except ValueError as e:
                print(f"ERROR: no se generaron las figuras de ejemplo (las predicciones sí se escribieron): {e}",
                      file=sys.stderr)
                codigo = 2
    return codigo


if __name__ == "__main__":
    sys.exit(main())
