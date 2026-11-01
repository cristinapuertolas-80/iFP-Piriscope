"""Compara dos ficheros de predicciones de Gold (comprobación de reproducibilidad).

Empareja las filas por ``id_municipio``, ``fecha_objetivo`` y ``version_modelo`` y resume, por versión y en
total, cuántas coinciden, la diferencia máxima de ``probabilidad`` y cuántas cambian de ``nivel_riesgo``.
Sirve para comprobar que el pipeline ejecutado desde cero reproduce las predicciones publicadas.

Solo usa pandas; no importa nada del proyecto.

    python scripts/comparar_predicciones.py --referencia publicado/predicciones.parquet \
        --nuevo data/gold/predicciones.parquet --desde 2026-10-01 --tolerancia 1e-6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CLAVE = ["id_municipio", "fecha_objetivo", "version_modelo"]


def leer(ruta: Path, desde: str | None, hasta: str | None) -> pd.DataFrame:
    df = pd.read_parquet(ruta, columns=CLAVE + ["probabilidad", "nivel_riesgo"])
    df["fecha_objetivo"] = pd.to_datetime(df["fecha_objetivo"]).dt.normalize()
    if desde:
        df = df[df["fecha_objetivo"] >= pd.Timestamp(desde)]
    if hasta:
        df = df[df["fecha_objetivo"] <= pd.Timestamp(hasta)]
    return df


def comparar(ref: pd.DataFrame, nuevo: pd.DataFrame) -> pd.DataFrame:
    m = ref.merge(nuevo, on=CLAVE, how="outer", suffixes=("_ref", "_nuevo"), indicator=True)
    filas = []
    for version, g in [("total", m), *m.groupby("version_modelo")]:
        ambos = g[g["_merge"] == "both"]
        dif = (ambos["probabilidad_ref"] - ambos["probabilidad_nuevo"]).abs()
        filas.append({
            "version_modelo": version,
            "en_ambos": len(ambos),
            "solo_referencia": int((g["_merge"] == "left_only").sum()),
            "solo_nuevo": int((g["_merge"] == "right_only").sum()),
            "dif_max_probabilidad": float(dif.max()) if len(dif) else float("nan"),
            "dif_media_probabilidad": float(dif.mean()) if len(dif) else float("nan"),
            "cambios_de_nivel": int((ambos["nivel_riesgo_ref"] != ambos["nivel_riesgo_nuevo"]).sum()),
        })
    return pd.DataFrame(filas)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compara dos predicciones.parquet (reproducibilidad)")
    ap.add_argument("--referencia", type=Path, required=True, help="predicciones publicadas")
    ap.add_argument("--nuevo", type=Path, required=True, help="predicciones regeneradas")
    ap.add_argument("--desde", default=None, help="primera fecha_objetivo a comparar (AAAA-MM-DD)")
    ap.add_argument("--hasta", default=None, help="última fecha_objetivo a comparar (AAAA-MM-DD)")
    ap.add_argument("--tolerancia", type=float, default=1e-6, help="diferencia de probabilidad admitida")
    a = ap.parse_args(argv)

    ref, nuevo = leer(a.referencia, a.desde, a.hasta), leer(a.nuevo, a.desde, a.hasta)
    tabla = comparar(ref, nuevo)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(tabla.to_string(index=False))
    total = tabla.iloc[0]
    if total["en_ambos"] == 0:
        print("\nNo hay filas comunes que comparar")
        return 1
    ok = total["dif_max_probabilidad"] <= a.tolerancia and total["cambios_de_nivel"] == 0
    print(f"\n{'REPRODUCIDO' if ok else 'DIFERENCIAS'}: {total['en_ambos']:,} predicciones comunes, diferencia máxima "
          f"{total['dif_max_probabilidad']:.2e} (tolerancia {a.tolerancia:.0e}), "
          f"{total['cambios_de_nivel']} cambios de nivel")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
