"""Construcción del dataset de Gold (municipio-día) a partir de las tablas de Silver.

Salidas en data/gold/:
- dataset_municipio_dia.parquet : todas las filas (también los días recientes sin etiqueta);
                                  sirve para la inferencia D -> D+1 y el dashboard.
- dataset_entrenamiento.parquet : solo filas con etiqueta y partición (cumple el contrato de
                                  src/ml/config.py).
- gold_informe.json             : tamaños, prevalencia por partición y nulos por variable.

La variable objetivo `incendio_d1` vale 1 si el municipio tiene al menos un parte EGIF con
fecha de detección D+1. La falta de parte NO es una observación negativa fuera del periodo de
cobertura de EGIF: las filas cuyo D+1 cae después del último año con etiquetas quedan SIN
etiqueta (nunca se marcan como 0).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from ..ml import config as mlcfg
from .derived import anadir_derivadas

COL_MUN, COL_FECHA = mlcfg.COL_MUNICIPIO, mlcfg.COL_FECHA
FWI_COLS = mlcfg.FWI_FEATURES
OBS_COLS = [COL_MUN, COL_FECHA, "dist_estacion_km", *mlcfg.METEO_FEATURES]
ORDEN_COLUMNAS = [COL_MUN, mlcfg.COL_PROVINCIA, COL_FECHA, mlcfg.COL_RESERVADO, mlcfg.COL_PARTICION,
                  mlcfg.TARGET, *mlcfg.METEO_FEATURES, *mlcfg.DERIVED_FEATURES, *FWI_COLS, "dist_estacion_km"]


@dataclass(frozen=True)
class Params:
    """Parámetros del dataset. Los años de partición están pendientes de confirmar (DEC-21)."""
    umbral_seco_mm: float = 1.0
    min_dias_acum: int = 7
    anio_train_hasta: int = 2021
    anio_validacion: int = 2022
    anio_test: int = 2023

    def __post_init__(self):
        if not self.anio_train_hasta < self.anio_validacion < self.anio_test:
            raise ValueError("Se exige anio_train_hasta < anio_validacion < anio_test")

    @property
    def fin_etiquetas(self) -> pd.Timestamp:
        """Último día con etiquetas fiables: el final del año de test."""
        return pd.Timestamp(self.anio_test, 12, 31)


# --- Lectura de Silver ------------------------------------------------------
def _num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        df[c] = pd.to_numeric(df[c]).astype("float64")
    return df


def leer_silver(silver_dir: Path) -> dict[str, pd.DataFrame]:
    silver_dir = Path(silver_dir)
    rd = lambda nombre, cols: pd.read_parquet(silver_dir / f"{nombre}.parquet", columns=cols)  # noqa: E731
    obs = rd("observacion_meteorologica", OBS_COLS)
    obs[COL_FECHA] = pd.to_datetime(obs[COL_FECHA])
    fwi = rd("indicador_fwi", [COL_MUN, COL_FECHA, *FWI_COLS])
    fwi[COL_FECHA] = pd.to_datetime(fwi[COL_FECHA])
    return {
        "obs": _num(obs, ["dist_estacion_km", *mlcfg.METEO_FEATURES]),
        "fwi": _num(fwi, FWI_COLS),
        "eventos": rd("evento_incendio", [COL_MUN, "fecha_deteccion"]),
        "municipio": rd("municipio", [COL_MUN, mlcfg.COL_PROVINCIA, mlcfg.COL_RESERVADO]),
    }


def leer_reservados(csv: Path, municipio: pd.DataFrame) -> pd.Series:
    """Mapa id_municipio -> reservado (0/1). La fuente de verdad es docs/municipios_reservados.csv."""
    if Path(csv).exists():
        lista = set(pd.read_csv(csv, dtype={COL_MUN: str})[COL_MUN])
        desconocidos = lista - set(municipio[COL_MUN])
        if desconocidos:
            raise ValueError(f"La lista de reservados tiene municipios que no existen en Silver: {sorted(desconocidos)[:5]}")
        return pd.Series(municipio[COL_MUN].isin(lista).astype("int8").to_numpy(), index=municipio[COL_MUN])
    if municipio[mlcfg.COL_RESERVADO].notna().all():
        return pd.Series(municipio[mlcfg.COL_RESERVADO].astype("int8").to_numpy(), index=municipio[COL_MUN])
    raise ValueError(f"Falta {csv}: sin la lista de municipios reservados no se puede construir el dataset")


# --- Etiqueta y particiones -------------------------------------------------
def etiquetar(df: pd.DataFrame, eventos: pd.DataFrame, fin_etiquetas: pd.Timestamp) -> pd.DataFrame:
    """Añade `incendio_d1`: 1 si hay parte con fecha de detección D+1; NA si D+1 no tiene cobertura."""
    ev = eventos.dropna(subset=[COL_MUN, "fecha_deteccion"]).copy()
    ev[COL_FECHA] = pd.to_datetime(ev["fecha_deteccion"]).dt.normalize() - pd.Timedelta(days=1)
    pos = ev.drop_duplicates([COL_MUN, COL_FECHA])[[COL_MUN, COL_FECHA]].assign(_pos=1)
    out = df.merge(pos, on=[COL_MUN, COL_FECHA], how="left")
    etiquetado = (out[COL_FECHA] + pd.Timedelta(days=1)) <= fin_etiquetas
    out[mlcfg.TARGET] = out["_pos"].fillna(0).where(etiquetado).astype("Int8")
    return out.drop(columns="_pos")


def asignar_particion(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Partición temporal por año de `fecha`, solo para filas con etiqueta; el resto queda en NA."""
    anio = df[COL_FECHA].dt.year
    part = np.select([anio <= p.anio_train_hasta, anio == p.anio_validacion, anio == p.anio_test],
                     [mlcfg.PART_TRAIN, mlcfg.PART_VAL, mlcfg.PART_TEST], default=None)
    out = df.copy()
    out[mlcfg.COL_PARTICION] = pd.Series(part, index=df.index, dtype="object").where(df[mlcfg.TARGET].notna())
    return out


# --- Construcción -----------------------------------------------------------
def construir_dataset(silver: dict[str, pd.DataFrame], reservados: pd.Series, p: Params = Params()) -> pd.DataFrame:
    obs = silver["obs"].sort_values([COL_MUN, COL_FECHA], kind="stable").reset_index(drop=True)
    df = anadir_derivadas(obs, p.umbral_seco_mm, p.min_dias_acum)
    df = df.merge(silver["fwi"], on=[COL_MUN, COL_FECHA], how="left", validate="one_to_one")
    df = df.merge(silver["municipio"][[COL_MUN, mlcfg.COL_PROVINCIA]], on=COL_MUN, how="left", validate="many_to_one")
    df[mlcfg.COL_RESERVADO] = df[COL_MUN].map(reservados).astype("int8")
    df = asignar_particion(etiquetar(df, silver["eventos"], p.fin_etiquetas), p)
    return df[ORDEN_COLUMNAS]


def informe(df: pd.DataFrame, p: Params) -> dict:
    et = df[df[mlcfg.TARGET].notna()]
    tasa = lambda g: round(float(g[mlcfg.TARGET].astype(float).mean()) * 100, 3)  # noqa: E731
    return {
        "generado_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parametros": asdict(p),
        "filas": int(len(df)), "filas_etiquetadas": int(len(et)),
        "municipios": int(df[COL_MUN].nunique()),
        "fecha_min": str(df[COL_FECHA].min().date()), "fecha_max": str(df[COL_FECHA].max().date()),
        "positivos": int(et[mlcfg.TARGET].sum()), "prevalencia_pct": tasa(et),
        "prevalencia_pct_por_particion": {k: tasa(g) for k, g in et.groupby(mlcfg.COL_PARTICION)},
        "prevalencia_pct_por_reservado": {str(k): tasa(g) for k, g in et.groupby(mlcfg.COL_RESERVADO)},
        "prevalencia_pct_por_provincia": {k: tasa(g) for k, g in et.groupby(mlcfg.COL_PROVINCIA)},
        "nulos_pct_filas_etiquetadas": {c: round(float(et[c].isna().mean()) * 100, 2)
                                        for c in [*mlcfg.METEO_FEATURES, *mlcfg.DERIVED_FEATURES, *FWI_COLS]},
    }


def escribir_gold(df: pd.DataFrame, gold_dir: Path, p: Params) -> dict:
    """Escribe los dos parquet y el informe. Valida el contrato de ML antes de escribir."""
    gold_dir = Path(gold_dir)
    gold_dir.mkdir(parents=True, exist_ok=True)
    train = df[df[mlcfg.TARGET].notna()].copy()
    train[mlcfg.TARGET] = train[mlcfg.TARGET].astype("int8")
    mlcfg.validate_schema(train)
    df.to_parquet(gold_dir / "dataset_municipio_dia.parquet", index=False)
    train.to_parquet(gold_dir / "dataset_entrenamiento.parquet", index=False)
    inf = informe(df, p)
    (gold_dir / "gold_informe.json").write_text(json.dumps(inf, indent=2, ensure_ascii=False), encoding="utf-8")
    return inf


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Construye el dataset de Gold a partir de Silver")
    ap.add_argument("--silver-dir", type=Path, default=mlcfg.SILVER_DIR)
    ap.add_argument("--gold-dir", type=Path, default=mlcfg.GOLD_DIR)
    ap.add_argument("--reservados", type=Path, default=mlcfg.RESERVADOS_CSV)
    ap.add_argument("--umbral-seco-mm", type=float, default=Params.umbral_seco_mm)
    ap.add_argument("--min-dias-acum", type=int, default=Params.min_dias_acum)
    ap.add_argument("--anio-train-hasta", type=int, default=Params.anio_train_hasta)
    ap.add_argument("--anio-validacion", type=int, default=Params.anio_validacion)
    ap.add_argument("--anio-test", type=int, default=Params.anio_test)
    a = ap.parse_args(argv)
    try:
        p = Params(a.umbral_seco_mm, a.min_dias_acum, a.anio_train_hasta, a.anio_validacion, a.anio_test)
        silver = leer_silver(a.silver_dir)
        df = construir_dataset(silver, leer_reservados(a.reservados, silver["municipio"]), p)
        inf = escribir_gold(df, a.gold_dir, p)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"Filas: {inf['filas']:,} ({inf['filas_etiquetadas']:,} etiquetadas) · {inf['fecha_min']} a {inf['fecha_max']}")
    print(f"Días con incendio (positivos): {inf['positivos']:,} = {inf['prevalencia_pct']} % de las filas etiquetadas")
    print("Prevalencia % por partición:", inf["prevalencia_pct_por_particion"])
    print("Nulos % (filas etiquetadas):", inf["nulos_pct_filas_etiquetadas"])
    print(f"Escrito en {a.gold_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
