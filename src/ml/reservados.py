"""Selección de municipios reservados para la evaluación geográfica (HU-06, DEC-24).

Los municipios reservados NO se usan nunca en el entrenamiento: sirven para
comprobar si el modelo generaliza a municipios no vistos (escenarios B y C).

Criterios (apartado 12 del diseño técnico):
- Reproducible: semilla fija.
- Por BLOQUES de municipios próximos: un bloque = los municipios que comparten
  estación AEMET asignada (comparten clima; reservar municipios sueltos daría
  resultados optimistas).
- Estratificada por provincia y por frecuencia histórica de incendios: dentro
  de cada provincia, los bloques se reparten en terciles (baja / media / alta)
  por su tasa de días con incendio y se reserva ~`fraccion` de cada estrato.
- Congelada: si el CSV ya existe, el script se niega a sobrescribirlo salvo
  `--forzar`.

Uso (desde la raíz del repositorio, con las tablas de Silver generadas):

    python -m src.ml.reservados
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    COL_MUNICIPIO, COL_PROVINCIA, RESERVADOS_CSV, RESERVADOS_META_JSON, SEED, SILVER_DIR,
)

FRACCION = 0.20
#: Años usados para medir la frecuencia de incendios: periodo común de las
#: fuentes hasta el último año anterior a validación/test (a confirmar, DEC-21).
ANIO_DESDE, ANIO_HASTA = 2001, 2021
COL_BLOQUE = "id_estacion_aemet"
NIVELES = {1: ("media",), 2: ("baja", "alta"), 3: ("baja", "media", "alta")}


# --- Carga de datos (Silver) ------------------------------------------------
def cargar_datos(silver_dir: Path, anio_desde: int = ANIO_DESDE, anio_hasta: int = ANIO_HASTA) -> pd.DataFrame:
    """Una fila por municipio: provincia, estación asignada y frecuencia de incendios."""
    silver_dir = Path(silver_dir)
    mun = pd.read_parquet(silver_dir / "municipio.parquet", columns=[COL_MUNICIPIO, COL_PROVINCIA])
    obs = pd.read_parquet(silver_dir / "observacion_meteorologica.parquet", columns=[COL_MUNICIPIO, COL_BLOQUE])
    est = obs.dropna().drop_duplicates()
    if (est.groupby(COL_MUNICIPIO)[COL_BLOQUE].nunique() > 1).any():
        raise ValueError("Hay municipios con más de una estación asignada; se esperaba una fija por municipio")

    ev = pd.read_parquet(silver_dir / "evento_incendio.parquet", columns=[COL_MUNICIPIO, "fecha_deteccion"]).dropna()
    dia = pd.to_datetime(ev["fecha_deteccion"]).dt.normalize()
    ev = ev.assign(dia=dia)[(dia.dt.year >= anio_desde) & (dia.dt.year <= anio_hasta)]
    dias_con_incendio = (ev.drop_duplicates([COL_MUNICIPIO, "dia"]).groupby(COL_MUNICIPIO).size()
                         .rename("dias_con_incendio"))

    df = mun.merge(est, on=COL_MUNICIPIO, how="left").merge(dias_con_incendio, on=COL_MUNICIPIO, how="left")
    sin_estacion = int(df[COL_BLOQUE].isna().sum())
    if sin_estacion:
        raise ValueError(f"{sin_estacion} municipios sin estación asignada en observacion_meteorologica")
    n_dias = (pd.Timestamp(anio_hasta, 12, 31) - pd.Timestamp(anio_desde, 1, 1)).days + 1
    df["dias_con_incendio"] = df["dias_con_incendio"].fillna(0).astype(int)
    df["tasa"] = df["dias_con_incendio"] / n_dias
    return df


# --- Selección --------------------------------------------------------------
def construir_bloques(df: pd.DataFrame) -> pd.DataFrame:
    """Un bloque por estación, con su provincia (la mayoritaria) y su tasa media de incendios."""
    g = df.groupby(COL_BLOQUE)
    bloques = pd.DataFrame({
        "n_municipios": g.size(),
        "dias_con_incendio": g["dias_con_incendio"].sum(),
        "tasa": g["tasa"].mean(),
        "provincia": g[COL_PROVINCIA].agg(lambda s: s.mode().iloc[0]),
    }).reset_index()
    bloques["nivel"] = ""
    for _, idx in bloques.groupby("provincia").groups.items():
        sub = bloques.loc[idx].sort_values(["tasa", COL_BLOQUE])
        etiquetas = NIVELES[min(3, len(sub))]
        for et, posiciones in zip(etiquetas, np.array_split(np.arange(len(sub)), len(etiquetas))):
            bloques.loc[sub.index[posiciones], "nivel"] = et
    bloques["estrato"] = bloques["provincia"] + "/" + bloques["nivel"]
    return bloques


def seleccionar_reservados(df: pd.DataFrame, fraccion: float = FRACCION,
                           semilla: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (df con columna `reservado`, tabla de bloques con `reservado`)."""
    if not 0 < fraccion < 1:
        raise ValueError("`fraccion` debe estar en (0, 1)")
    rng = np.random.default_rng(semilla)
    bloques = construir_bloques(df)
    elegidos: list[str] = []
    for _, sub in bloques.groupby("estrato", sort=True):
        objetivo = max(1, round(fraccion * sub["n_municipios"].sum()))
        acumulado = 0
        for _, b in sub.iloc[rng.permutation(len(sub))].iterrows():
            if acumulado >= objetivo:
                break
            # Si añadir el bloque se pasa del objetivo más de lo que se queda corto, se salta
            if acumulado > 0 and acumulado + b["n_municipios"] - objetivo > objetivo - acumulado:
                continue
            elegidos.append(b[COL_BLOQUE])
            acumulado += b["n_municipios"]
    bloques["reservado"] = bloques[COL_BLOQUE].isin(elegidos).astype(int)
    out = df.merge(bloques[[COL_BLOQUE, "estrato", "reservado"]], on=COL_BLOQUE, how="left")
    return out, bloques


def resumir(out: pd.DataFrame, bloques: pd.DataFrame) -> dict:
    """Resumen para comprobar que la selección está equilibrada."""
    res = out["reservado"] == 1
    por_prov = out.groupby(COL_PROVINCIA).agg(municipios=(COL_MUNICIPIO, "size"),
                                              reservados=("reservado", "sum"))
    por_prov["pct"] = (100 * por_prov["reservados"] / por_prov["municipios"]).round(1)
    return {
        "n_municipios": int(len(out)),
        "n_reservados": int(res.sum()),
        "pct_reservados": round(100 * res.mean(), 1),
        "n_bloques": int(len(bloques)),
        "n_bloques_reservados": int(bloques["reservado"].sum()),
        "por_provincia": por_prov.reset_index().to_dict(orient="records"),
        "dias_con_incendio_medios_reservados": round(float(out.loc[res, "dias_con_incendio"].mean()), 2),
        "dias_con_incendio_medios_no_reservados": round(float(out.loc[~res, "dias_con_incendio"].mean()), 2),
    }


# --- CLI --------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Selecciona los municipios reservados para la evaluación geográfica")
    ap.add_argument("--silver-dir", type=Path, default=SILVER_DIR)
    ap.add_argument("--salida", type=Path, default=RESERVADOS_CSV)
    ap.add_argument("--meta", type=Path, default=RESERVADOS_META_JSON)
    ap.add_argument("--fraccion", type=float, default=FRACCION)
    ap.add_argument("--semilla", type=int, default=SEED)
    ap.add_argument("--desde-anio", type=int, default=ANIO_DESDE)
    ap.add_argument("--hasta-anio", type=int, default=ANIO_HASTA)
    ap.add_argument("--forzar", action="store_true", help="sobrescribe una lista ya congelada")
    args = ap.parse_args(argv)

    if args.salida.exists() and not args.forzar:
        print(f"ERROR: {args.salida} ya existe. La lista está congelada y no debe modificarse "
              f"(usa --forzar solo si el equipo lo decide).", file=sys.stderr)
        return 2
    try:
        df = cargar_datos(args.silver_dir, args.desde_anio, args.hasta_anio)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    out, bloques = seleccionar_reservados(df, args.fraccion, args.semilla)
    resumen = resumir(out, bloques)

    args.salida.parent.mkdir(parents=True, exist_ok=True)
    (out.loc[out["reservado"] == 1, [COL_MUNICIPIO]].assign(reservado=1)
        .sort_values(COL_MUNICIPIO).to_csv(args.salida, index=False))
    meta = {
        "semilla": args.semilla, "fraccion": args.fraccion,
        "anio_desde": args.desde_anio, "anio_hasta": args.hasta_anio,
        "criterio": "bloques = municipios con la misma estación AEMET asignada; "
                    "estratos = provincia x tercil de frecuencia de días con incendio",
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "resumen": resumen,
        "bloques_reservados": sorted(bloques.loc[bloques["reservado"] == 1, COL_BLOQUE].tolist()),
    }
    args.meta.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Municipios reservados: {resumen['n_reservados']} de {resumen['n_municipios']} "
          f"({resumen['pct_reservados']} %), en {resumen['n_bloques_reservados']} de {resumen['n_bloques']} bloques")
    print(pd.DataFrame(resumen["por_provincia"]).to_string(index=False))
    print(f"Días con incendio por municipio ({args.desde_anio}-{args.hasta_anio}): reservados "
          f"{resumen['dias_con_incendio_medios_reservados']} · resto "
          f"{resumen['dias_con_incendio_medios_no_reservados']}")
    print(f"Escrito: {args.salida} y {args.meta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
