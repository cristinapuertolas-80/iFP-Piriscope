"""Diagnostica las fechas con casi todos los municipios en medio o alto.

Responde a: ¿es un artefacto de datos ausentes, o el modelo está saturado por época del año?
Solo usa pandas.

Uso:
    python diagnostico_alarmas.py --pred data/gold/predicciones.parquet --expl data/gold/explicaciones.parquet \
        --dataset data/gold/dataset_municipio_dia.parquet [--umbral 0.95] [--desde-anio 2009]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--expl", type=Path, required=True)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--umbral", type=float, default=0.95, help="fracción medio+alto que marca una fecha como saturada")
    ap.add_argument("--desde-anio", type=int, default=2009, help="primer año con el que calcular prevalencias históricas")
    a = ap.parse_args()

    p, e, d = pd.read_parquet(a.pred), pd.read_parquet(a.expl), pd.read_parquet(a.dataset)
    p["fecha_objetivo"] = pd.to_datetime(p["fecha_objetivo"]).dt.normalize()
    d["fecha"] = pd.to_datetime(d["fecha"]).dt.normalize()
    p["alarma"] = p["nivel_riesgo"].isin(["medio", "alto"])
    mes_p = p["fecha_objetivo"].dt.month

    por_fecha = p.groupby("fecha_objetivo")["alarma"].mean()
    malas = por_fecha[por_fecha >= a.umbral]
    print(f"== 1. Fechas con >= {a.umbral:.0%} en medio o alto: {len(malas)} de {len(por_fecha)}")
    print(malas.groupby([malas.index.year.rename("año"), malas.index.month.rename("mes")]).size().to_string())

    etq = d[d["incendio_d1"].notna() & (d["fecha"].dt.year >= a.desde_anio)]
    prev = etq.groupby(etq["fecha"].dt.month)["incendio_d1"].mean() * 100
    tabla = pd.DataFrame({
        "prevalencia_historica_%": prev,
        "medio+alto_%": p.groupby(mes_p)["alarma"].mean() * 100,
        "prob_media_%": p.groupby(mes_p)["probabilidad"].mean() * 100,
        "n_fechas": p.groupby(mes_p)["fecha_objetivo"].nunique(),
    }).round(2)
    tabla.index.name = "mes"
    print(f"\n== 2. Prevalencia histórica ({a.desde_anio}-2023) frente a lo que marca el modelo en 2024-2026, por mes")
    print(tabla.to_string())

    print("\n== 3. Distribución de la probabilidad en las 5 fechas más saturadas (¿sube todo a la vez?)")
    peores = por_fecha.sort_values(ascending=False).head(5).index
    desc = p[p["fecha_objetivo"].isin(peores)].groupby("fecha_objetivo")["probabilidad"].describe()
    print(desc[["min", "25%", "50%", "75%", "max"]].round(4).to_string())
    print("probabilidades distintas por fecha:", p[p["fecha_objetivo"].isin(peores)].groupby("fecha_objetivo")["probabilidad"].nunique().to_dict())

    ids_mal = set(p.loc[p["fecha_objetivo"].isin(malas.index), "id_prediccion"])
    em = e["id_prediccion"].isin(ids_mal)
    resumen = pd.DataFrame({
        "aparece_en_top_%_fechas_saturadas": e[em].groupby("variable")["id_prediccion"].nunique() / max(len(ids_mal), 1) * 100,
        "shap_medio_saturadas": e[em].groupby("variable")["valor_shap"].mean(),
        "shap_medio_resto": e[~em].groupby("variable")["valor_shap"].mean(),
    }).sort_values("aparece_en_top_%_fechas_saturadas", ascending=False).round(3)
    print("\n== 4. SHAP: qué variables empujan en las fechas saturadas frente al resto (¿domina `mes`?)")
    print(resumen.to_string())

    cols = [c for c in ["ffmc", "fwi", "dc", "precipitacion", "humedad_relativa", "temperatura", "racha_dias_secos"] if c in d.columns]
    d_mal = d[d["fecha"].isin(malas.index - pd.Timedelta(days=1))]
    hist = etq[etq["fecha"].dt.month.isin(set(malas.index.month))][cols].mean()
    comp = pd.DataFrame({"fechas_saturadas": d_mal[cols].mean(), "historico_mismos_meses": hist,
                         "nulos_fechas_saturadas_%": d_mal[cols].isna().mean() * 100}).round(2)
    print("\n== 5. Meteorología del día D en las fechas saturadas frente al histórico de esos meses")
    print(comp.to_string())

    print("\nLectura: si la prevalencia histórica del mes es alta, `mes` domina el SHAP y la probabilidad sube en todos "
          "los municipios a la vez, es saturación estacional (umbral global demasiado bajo), no datos ausentes. "
          "Si hay muchos nulos o una meteorología normal con probabilidad alta, hay un problema de datos o del modelo.")


if __name__ == "__main__":
    main()
