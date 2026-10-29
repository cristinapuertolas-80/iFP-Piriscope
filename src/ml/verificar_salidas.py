"""Verifica las salidas reales de `inferir.py` (predicciones.parquet y explicaciones.parquet).

No importa nada del proyecto: solo pandas. Devuelve código 1 si falla alguna comprobación.

Uso:
    python verificar_salidas.py \
        --pred data/gold/predicciones.parquet --expl data/gold/explicaciones.parquet \
        --dataset data/gold/dataset_municipio_dia.parquet --models-dir models \
        --modelo v1_xgboost_con_fwi-sin_viento-desde2009 \
        --respaldo v1_xgboost_sin_fwi-sin_viento-desde2009 --top-k 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

FWI = {"fwi", "ffmc", "dmc", "dc", "isi", "bui"}
fallos = 0


def comprobar(ok: bool, texto: str, detalle: str = "") -> None:
    global fallos
    fallos += 0 if ok else 1
    print(f"[{'OK' if ok else 'FALLO'}] {texto}" + (f" · {detalle}" if detalle else ""))


def nivel_esperado(p: pd.Series, umbrales: dict) -> pd.Series:
    n = pd.Series("bajo", index=p.index)
    n[p >= umbrales["medio"]] = "medio"
    n[p >= umbrales["alto"]] = "alto"
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--expl", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, default=None, help="para calcular la cobertura sobre el total de municipios")
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--modelo", default=None, help="id del modelo principal (versión sin sufijo)")
    ap.add_argument("--respaldo", default=None, help="id del modelo de respaldo (versión *-respaldo)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-alarmas", type=float, default=0.95)
    ap.add_argument("--min-cobertura", type=float, default=0.8)
    a = ap.parse_args()

    p, e = pd.read_parquet(a.pred), pd.read_parquet(a.expl)
    p["fecha_objetivo"] = pd.to_datetime(p["fecha_objetivo"]).dt.normalize()
    print(f"{len(p):,} predicciones · {p['fecha_objetivo'].min():%Y-%m-%d} a {p['fecha_objetivo'].max():%Y-%m-%d} "
          f"· versiones: {p['version_modelo'].value_counts().to_dict()}\n")

    # --- Integridad
    comprobar(p["id_prediccion"].is_unique, "id_prediccion único")
    dup = int(p.duplicated(["id_municipio", "fecha_objetivo"]).sum())
    comprobar(dup == 0, "una sola predicción por municipio y fecha", f"{dup} duplicadas" if dup else "")
    comprobar(p["probabilidad"].notna().all() and p["probabilidad"].between(0, 1).all(), "probabilidades en [0, 1] y sin nulos")
    comprobar(p["nivel_riesgo"].isin(["bajo", "medio", "alto"]).all(), "niveles de riesgo válidos")
    comprobar(p["id_municipio"].astype(str).str.fullmatch(r"\d{5}").all(), "id_municipio con 5 cifras (texto)")

    # --- Explicaciones
    comprobar(set(e["id_prediccion"]) == set(p["id_prediccion"]), "explicaciones y predicciones con los mismos ids",
              f"{len(set(e['id_prediccion']) ^ set(p['id_prediccion'])):,} ids sin pareja")
    por_pred = e.groupby("id_prediccion").size()
    comprobar((por_pred == a.top_k).all(), f"{a.top_k} variables por predicción", str(por_pred.value_counts().to_dict()))
    comprobar(not e.duplicated(["id_prediccion", "variable"]).any(), "sin variable repetida dentro de una predicción")
    comprobar(e["valor_shap"].notna().all(), "valor_shap sin nulos")

    # --- Respaldo y FWI
    resp = p["version_modelo"].str.endswith("-respaldo")
    comprobar(p.loc[~resp, "fwi"].notna().all(), "las filas del modelo principal tienen FWI")
    if resp.any():
        comprobar(p.loc[resp, "fwi"].isna().all(), "las filas de respaldo no llevan FWI")
        ev = e[e["id_prediccion"].isin(p.loc[resp, "id_prediccion"])]["variable"]
        comprobar(not (set(ev) & FWI), "las explicaciones del respaldo no usan variables FWI", str(sorted(set(ev) & FWI)))
        distintos = p[resp].groupby("fecha_objetivo")["probabilidad"].nunique()
        print(f"       valores distintos de probabilidad del respaldo por fecha: {distintos.to_dict()}")

    # --- Cobertura
    if a.dataset:
        n_mun = pd.read_parquet(a.dataset, columns=["id_municipio"])["id_municipio"].nunique()
        cob = p.groupby("fecha_objetivo")["id_municipio"].nunique() / n_mun
        comprobar((cob >= a.min_cobertura).all(), f"cobertura >= {a.min_cobertura:.0%} en todas las fechas",
                  f"mínima {cob.min():.0%} el {cob.idxmin():%Y-%m-%d}; {int((cob < 1).sum())} fecha(s) incompletas")
        print(f"       últimas fechas: {(cob.tail(4) * n_mun).round().astype(int).to_dict()} de {n_mun}")

    # --- Niveles frente a umbrales guardados
    if a.models_dir:
        for etiqueta, mid, mask in (("principal", a.modelo, ~resp), ("respaldo", a.respaldo, resp)):
            ruta = a.models_dir / f"{mid}_umbrales.json" if mid else None
            if ruta is None or not mask.any():
                continue
            u = json.loads(ruta.read_text(encoding="utf-8"))
            esp = nivel_esperado(p.loc[mask, "probabilidad"], u)
            cerca = (p.loc[mask, "probabilidad"] - u["medio"]).abs().lt(1e-9) | (p.loc[mask, "probabilidad"] - u["alto"]).abs().lt(1e-9)
            malos = int(((esp != p.loc[mask, "nivel_riesgo"]) & ~cerca).sum())
            comprobar(malos == 0, f"niveles coherentes con los umbrales del modelo {etiqueta}",
                      f"medio={u['medio']:.4f}, alto={u['alto']:.4f}" + (f"; {malos} incoherentes" if malos else ""))

    # --- Plausibilidad
    alarma = p["nivel_riesgo"].isin(["medio", "alto"]).groupby(p["fecha_objetivo"]).mean()
    comprobar((alarma < a.max_alarmas).all(), f"ninguna fecha con >= {a.max_alarmas:.0%} en medio o alto",
              f"máximo {alarma.max():.0%} el {alarma.idxmax():%Y-%m-%d}")
    ult = p[p["fecha_objetivo"] == p["fecha_objetivo"].max()]
    print(f"       última fecha: {ult['nivel_riesgo'].value_counts().to_dict()}; "
          f"{ult['probabilidad'].nunique()} probabilidades distintas")

    print(f"\n{'Todo correcto' if not fallos else f'{fallos} comprobación(es) fallida(s)'}")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
