"""Compara políticas de niveles de riesgo y revisa la calibración (DEC-41, DEC-49).

Puntúa con el modelo guardado la VALIDACIÓN 2022 de municipios no reservados (escenario A) y calcula:

1. Calibración: probabilidad media frente a tasa real de incendio, por tramos de probabilidad.
2. Umbrales actuales (medio, alto): Recall, Precision y % de municipio-días marcados.
3. Políticas alternativas: (a) umbral por presupuesto de alarmas, (b) top-x % de municipios de cada día,
   (c) híbrido: top-x % del día Y probabilidad >= umbral «medio».
4. Fracción marcada por mes con el umbral actual frente al híbrido (¿desaparece la saturación?).

LÍMITES: el modelo paró con la validación de 2022 y sus umbrales se fijaron en ella, así que estas cifras son
optimistas y solo sirven para comparar políticas entre sí. 2022 son ~1.200 positivos: para decidir de verdad,
repite el análisis con el walk-forward 2016-2022 de DEC-47. NO uses el test de 2023 (DEC-47).

Uso (desde la raíz del repositorio):
    python -m src.ml.analisis_niveles --modelo v1_xgboost_con_fwi-sin_viento-desde2009
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, splits, train
from .inferir import cargar

pd.set_option("display.width", 200)


def metricas(y: pd.Series, marcado: pd.Series) -> dict:
    tp, pos, n = int((y & marcado).sum()), int(y.sum()), int(marcado.sum())
    return {"recall": tp / pos if pos else np.nan, "precision": tp / n if n else np.nan, "marcado_%": 100 * n / len(y)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    ap.add_argument("--dataset", type=Path, default=config.GOLD_DIR / "dataset_entrenamiento.parquet")
    ap.add_argument("--top", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.20], help="fracciones de municipios por día")
    a = ap.parse_args()

    bundle, thr = cargar(a.models_dir, a.modelo)
    df = pd.read_parquet(a.dataset)
    sel = splits.select_scenario(df, "A", split="validacion").copy()
    scored = train.score_dataframe(bundle, sel)
    if len(scored) != len(sel):
        raise SystemExit("score_dataframe devolvió otro número de filas; adapta el script a su salida")
    sel["proba"] = scored["proba"].to_numpy()
    y = sel[config.TARGET].astype(bool)
    p = sel["proba"]
    fecha = pd.to_datetime(sel[config.COL_FECHA]).dt.normalize()
    print(f"{len(sel):,} filas de validación · {int(y.sum())} positivos ({y.mean():.2%}) · "
          f"umbrales medio={thr.medio:.4f}, alto={thr.alto:.4f}\n")

    # 1. Calibración
    q = pd.qcut(p, 10, duplicates="drop")
    cal = pd.DataFrame({"y": y.astype(int), "p": p, "tramo": q}).groupby("tramo", observed=True).agg(
        prob_media=("p", "mean"), tasa_real=("y", "mean"), n=("y", "size"), positivos=("y", "sum"))
    cal["prob/real"] = cal["prob_media"] / cal["tasa_real"]
    print("== 1. Calibración por deciles de probabilidad (prob/real ≈ 1 si está calibrada)")
    print(cal.round(4).to_string())
    print(f"Global: probabilidad media {p.mean():.4f} frente a tasa real {y.mean():.4f}\n")

    # 2. Umbrales actuales
    filas = {"medio o alto (umbral actual)": metricas(y, p >= thr.medio), "alto (umbral actual)": metricas(y, p >= thr.alto)}

    # 3. Políticas alternativas
    for x in a.top:
        presupuesto = float(p.quantile(1 - x))
        filas[f"presupuesto {x:.0%} global (umbral {presupuesto:.4f})"] = metricas(y, p >= presupuesto)
    rango = p.groupby(fecha).rank(ascending=False, method="first") / p.groupby(fecha).transform("size")
    for x in a.top:
        filas[f"top {x:.0%} de cada día"] = metricas(y, rango <= x)
        filas[f"híbrido: top {x:.0%} del día y p >= medio"] = metricas(y, (rango <= x) & (p >= thr.medio))
    print("== 2-3. Recall, Precision y % de municipio-días marcados")
    print(pd.DataFrame(filas).T.round(3).to_string())

    # 4. Por mes
    mes = fecha.dt.month
    top_ref = a.top[min(1, len(a.top) - 1)]
    por_mes = pd.DataFrame({
        "marcado_umbral_actual_%": (p >= thr.medio).groupby(mes).mean() * 100,
        f"marcado_hibrido_top{top_ref:.0%}_%": (((rango <= top_ref) & (p >= thr.medio))).groupby(mes).mean() * 100,
        "prevalencia_%": y.groupby(mes).mean() * 100,
    }).round(2)
    print("\n== 4. % de municipio-días marcados por mes (2022)")
    print(por_mes.to_string())
    # 5. Ranking dentro de los días saturados (>= 50 % de municipios sobre «medio»)
    sat = (p >= thr.medio).groupby(fecha).transform("mean") >= 0.5
    ys, rs = y[sat], rango[sat]
    if ys.sum():
        base = ys.mean()
        filas5 = {}
        for x in (0.1, 0.2, 0.3):
            m = metricas(ys, rs <= x)
            m["lift"] = m["precision"] / base
            filas5[f"top {x:.0%} del día"] = m
        print(f"\n== 5. Solo días saturados ({sat.mean():.0%} de las filas, prevalencia {base:.2%}): ¿ordena bien los municipios?")
        print(pd.DataFrame(filas5).T.round(3).to_string())
        print("lift > 1: el ranking dentro del día concentra los incendios mejor que el azar.")
    print("\nLectura: el top-x % por día marca siempre lo mismo (también en invierno, con riesgo despreciable); "
          "el híbrido evita eso exigiendo además superar el umbral «medio». Elige según cuántas alarmas por día "
          "puede atender el usuario y regístralo en DEC-49.")


if __name__ == "__main__":
    main()
