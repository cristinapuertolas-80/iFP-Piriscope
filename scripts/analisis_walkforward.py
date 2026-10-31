"""Políticas de niveles de riesgo y calibración con el walk-forward 2016-2022 (DEC-47, DEC-49).

Entrada: el parquet de `walkforward.py --guardar-predicciones`, con las columnas
    id_municipio, fecha, incendio_d1, fwi, proba, conjunto ('validacion' = año Y-1, 'evaluacion' = año Y), anio, variante.

Protocolo fiel a la operación: cada umbral se fija SOLO con el año anterior (validacion) y se aplica al año
siguiente (evaluacion). No usa el test de 2023. Solo requiere pandas y numpy.

Responde a:
  1. ¿Se mantiene el Recall objetivo de un año para otro, y a costa de cuántas alarmas?
  2. ¿Qué ocurre con una política por presupuesto de alarmas (5, 10 o 20 % de municipio-días)?
  3. ¿Cómo se compara el modelo con el FWI con la MISMA política?
  4. ¿Cuántas alarmas hay por mes frente a la prevalencia, y cuántos días saturan?
  5. ¿Ordena el modelo a los municipios dentro de los días saturados (lift), y es estable por año?
  6. ¿Está calibrada la probabilidad? ¿Mejora ajustarla con la tasa base del año anterior?

Uso:
    python analisis_walkforward.py --pred docs/resultados/dev/walkforward_predicciones.parquet [--variante con_fwi]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
PRESUPUESTOS = (0.20, 0.10, 0.05)


# ---------- utilidades ----------
def umbral_recall(y: np.ndarray, p: np.ndarray, objetivo: float) -> float:
    """Mayor umbral con Recall >= objetivo (puntuación del k-ésimo positivo más alto)."""
    pos = np.sort(p[y])[::-1]
    k = max(int(np.ceil(objetivo * len(pos))), 1)
    return float(pos[k - 1])


def metricas(y: np.ndarray, marcado: np.ndarray) -> dict:
    n_pos, n_marc = int(y.sum()), int(marcado.sum())
    tp = int((y & marcado).sum())
    return {"recall": tp / n_pos if n_pos else np.nan, "precision": tp / n_marc if n_marc else np.nan,
            "marcado_%": 100 * n_marc / len(y) if len(y) else np.nan}


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def sigmoide(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def ajuste_sesgo(p: np.ndarray, objetivo: float) -> float:
    """Desplazamiento b (en log-odds) tal que la probabilidad media ajustada iguala `objetivo`."""
    lo, hi, z = -10.0, 10.0, logit(p)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if sigmoide(z + mid).mean() > objetivo else (mid, hi)
    return (lo + hi) / 2


def pliegues(d: pd.DataFrame) -> list[tuple[int, pd.DataFrame, pd.DataFrame]]:
    out = []
    for anio, g in d.groupby("anio"):
        v, e = g[g["conjunto"] == "validacion"], g[g["conjunto"] == "evaluacion"]
        if len(v) and len(e):
            out.append((int(anio), v.reset_index(drop=True), e.reset_index(drop=True)))
    return out


def evaluar_politica(folds, columna: str, regla) -> tuple[dict, list[dict]]:
    """Aplica `regla(y_val, puntuación_val) -> umbral` por año y evalúa en el año siguiente.
    Solo filas con FWI, para comparar modelo y FWI sobre exactamente los mismos datos."""
    ys, fl, por_anio = [], [], []
    for _, v, e in folds:
        v, e = v[v["fwi"].notna()], e[e["fwi"].notna()]
        umbral = regla(v["y"].to_numpy(), v[columna].to_numpy())
        marcado = e[columna].to_numpy() >= umbral
        ys.append(e["y"].to_numpy())
        fl.append(marcado)
        por_anio.append(metricas(ys[-1], marcado))
    return metricas(np.concatenate(ys), np.concatenate(fl)), por_anio


def rango(serie: pd.Series) -> str:
    return f"{serie.min():.1f}-{serie.max():.1f}"


# ---------- análisis ----------
def analizar(d: pd.DataFrame) -> None:
    folds = pliegues(d)
    anios = [a for a, _, _ in folds]
    print(f"{len(d):,} filas · años de evaluación {anios[0]}-{anios[-1]} · {len(folds)} pliegues\n")

    # 1. Resumen por año con los niveles actuales (Recall objetivo 0,8 y 0,5 fijados con el año anterior)
    filas, acum = [], {k: [] for k in ("y", "p", "medio", "fecha")}
    for anio, v, e in folds:
        yv, pv, ye, pe = v["y"].to_numpy(), v["proba"].to_numpy(), e["y"].to_numpy(), e["proba"].to_numpy()
        tm, ta = umbral_recall(yv, pv, 0.8), umbral_recall(yv, pv, 0.5)
        b = ajuste_sesgo(pv, yv.mean())
        pe2 = sigmoide(logit(pe) + b)
        mm, ma = metricas(ye, pe >= tm), metricas(ye, pe >= ta)
        filas.append({"anio": anio, "prevalencia_%": 100 * ye.mean(), "proba_media_%": 100 * pe.mean(),
                      "proba/real": pe.mean() / ye.mean(), "proba/real_ajustada": pe2.mean() / ye.mean(),
                      "brier_x1000": 1000 * np.mean((pe - ye) ** 2), "brier_ajust_x1000": 1000 * np.mean((pe2 - ye) ** 2),
                      "umbral_medio": tm, "umbral_alto": ta,
                      "recall_medio": mm["recall"], "precision_medio": mm["precision"], "marcado_medio_%": mm["marcado_%"],
                      "recall_alto": ma["recall"], "precision_alto": ma["precision"], "marcado_alto_%": ma["marcado_%"]})
        acum["y"].append(ye); acum["p"].append(pe); acum["medio"].append(pe >= tm)
        acum["fecha"].append(pd.to_datetime(e["fecha"]).to_numpy())
    t1 = pd.DataFrame(filas)
    print("== 1. Niveles actuales (Recall objetivo 0,8 y 0,5, umbral fijado con el año anterior), por año")
    print(t1.round(4).to_string(index=False))
    print(f"\nRecall con «medio»: {rango(100 * t1['recall_medio'])} % (objetivo 80 %) · "
          f"alarmas con «medio»: {rango(t1['marcado_medio_%'])} % de los municipio-días · "
          f"Precision: {rango(100 * t1['precision_medio'])} %\n")

    # 2-3. Políticas, con el modelo y con el FWI aplicando la MISMA regla
    reglas = {"Recall objetivo 0,8": lambda y, p: umbral_recall(y, p, 0.8),
              "Recall objetivo 0,5": lambda y, p: umbral_recall(y, p, 0.5)}
    for b in PRESUPUESTOS:
        reglas[f"Presupuesto {b:.0%}"] = (lambda q: lambda y, p: float(np.quantile(p, 1 - q)))(b)
    filas = []
    for nombre, regla in reglas.items():
        pm, am = evaluar_politica(folds, "proba", regla)
        pf, af = evaluar_politica(folds, "fwi", regla)
        am, af = pd.DataFrame(am), pd.DataFrame(af)
        filas.append({"política": nombre, "modelo_recall": pm["recall"], "modelo_precision": pm["precision"],
                      "modelo_marcado_%": pm["marcado_%"], "modelo_marcado_%_por_año": rango(am["marcado_%"]),
                      "modelo_recall_por_año": rango(100 * am["recall"]),
                      "fwi_recall": pf["recall"], "fwi_precision": pf["precision"], "fwi_marcado_%": pf["marcado_%"]})
    print("== 2-3. Políticas con el umbral del año anterior: modelo frente a FWI con la MISMA regla (todos los años juntos)")
    print(pd.DataFrame(filas).round(3).to_string(index=False))
    print("Lectura: en «Recall objetivo» fijas el Recall y las alarmas varían; en «Presupuesto» fijas las alarmas y varía el "
          "Recall. Mira los rangos por año de la columna que NO fijas: cuanto más estrechos, más predecible es la política.\n")

    # 4. Alarmas por mes y días saturados
    y_all, p_all = np.concatenate(acum["y"]), np.concatenate(acum["p"])
    marc, fecha = np.concatenate(acum["medio"]), np.concatenate(acum["fecha"])
    mes = pd.DatetimeIndex(fecha).month
    t4 = pd.DataFrame({"y": y_all, "marcado": marc, "mes": mes}).groupby("mes").agg(
        marcado_medio_pct=("marcado", lambda s: 100 * s.mean()), prevalencia_pct=("y", lambda s: 100 * s.mean()))
    t4["precision_%"] = [100 * (y_all[mes == m] & marc[mes == m]).sum() / max(marc[mes == m].sum(), 1) for m in t4.index]
    print("== 4a. Alarmas «medio» y prevalencia por mes (todos los años)")
    print(t4.round(2).to_string())

    sat_filas, lift_ac = [], {x: ([], []) for x in (0.1, 0.2, 0.3)}
    base_ac = []
    for (anio, v, e), tm in zip(folds, t1["umbral_medio"]):
        f = pd.to_datetime(e["fecha"])
        frac = (e["proba"] >= tm).groupby(f).transform("mean")
        dias = frac.groupby(f).first()
        sat = (frac >= 0.5).to_numpy()
        rk = (e["proba"].groupby(f).rank(ascending=False, method="first") / e["proba"].groupby(f).transform("size")).to_numpy()
        ye = e["y"].to_numpy()
        fila = {"anio": anio, "dias": len(dias), "dias_>=95%": int((dias >= 0.95).sum()), "dias_>=50%": int((dias >= 0.5).sum()),
                "prevalencia_dias_sat_%": 100 * ye[sat].mean() if sat.any() else np.nan}
        for x in lift_ac:
            if sat.any() and ye[sat].sum():
                fila[f"lift_top{x:.0%}"] = ye[sat & (rk <= x)].mean() / ye[sat].mean() if (sat & (rk <= x)).any() else np.nan
                lift_ac[x][0].append(ye[sat & (rk <= x)]); lift_ac[x][1].append(ye[sat])
        sat_filas.append(fila)
    print("\n== 4b. Días saturados por año y lift del ranking dentro de esos días (saturado = >= 50 % de municipios sobre «medio»)")
    print(pd.DataFrame(sat_filas).round(3).to_string(index=False))
    agregados = {f"lift_top{x:.0%}": float(np.concatenate(a).mean() / np.concatenate(b).mean())
                 for x, (a, b) in lift_ac.items() if a}
    print("Todos los años juntos:", {k: round(v, 2) for k, v in agregados.items()}, "\n")

    # 6. Calibración (solo evaluación, fuera de muestra)
    dec = pd.qcut(p_all, 10, duplicates="drop")
    cal = pd.DataFrame({"y": y_all.astype(int), "p": p_all, "tramo": dec}).groupby("tramo", observed=True).agg(
        prob_media=("p", "mean"), tasa_real=("y", "mean"), n=("y", "size"), positivos=("y", "sum"))
    cal["prob/real"] = cal["prob_media"] / cal["tasa_real"]
    print("== 6. Calibración por deciles (todos los años de evaluación, fuera de muestra)")
    print(cal.round(4).to_string())
    print(f"\nGlobal: probabilidad media {p_all.mean():.4f} frente a tasa real {y_all.mean():.4f}. "
          f"Ajustar con la tasa base del año anterior: Brier medio {t1['brier_x1000'].mean():.3f} → "
          f"{t1['brier_ajust_x1000'].mean():.3f} (x1000; menor es mejor); proba/real medio "
          f"{t1['proba/real'].mean():.2f} → {t1['proba/real_ajustada'].mean():.2f}.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Políticas de niveles y calibración con el walk-forward")
    ap.add_argument("--pred", type=Path, required=True, help="parquet de walkforward.py --guardar-predicciones")
    ap.add_argument("--variante", default="con_fwi", help="texto contenido en el nombre de la variante")
    a = ap.parse_args(argv)
    d = pd.read_parquet(a.pred)
    d = d[d["variante"].str.contains(a.variante, regex=False)].copy()
    if d.empty or d["variante"].nunique() != 1:
        print(f"ERROR: --variante {a.variante!r} debe seleccionar exactamente una variante. Hay: "
              f"{sorted(pd.read_parquet(a.pred, columns=['variante'])['variante'].unique())}", file=sys.stderr)
        return 2
    d["y"] = d["incendio_d1"].astype(bool)
    print(f"Variante: {d['variante'].iloc[0]}")
    analizar(d)
    return 0


if __name__ == "__main__":
    sys.exit(main())
