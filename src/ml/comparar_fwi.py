"""Compara el FWI intermedio con el consolidado para un mismo mes (DEC-31, DEC-34, DEC-48).

El modelo se entrenó con FWI consolidado, pero los meses recientes solo tienen el intermedio. Este script
mide cuánto difieren ambos en las celdas y días que comparten: si la diferencia es pequeña, las predicciones
recientes son fiables; si no, hay que documentarlo como limitación.

Necesitas AMBAS versiones del mismo mes. Si no las tienes, descarga el `intermediate_dataset` de un mes ya
consolidado (con la misma ingesta que usas para el FWI) y compáralo con su fichero consolidado.

Uso:
    python comparar_fwi.py --consolidado data/bronze/EFFIS/<ts>/fwi_2025_06.nc \
                           --intermedio  data/bronze/EFFIS/<ts>/fwi_2025_06_intermedio.nc [--umbral-fwi 5]

Requiere xarray y netCDF4 (o h5netcdf).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def abrir(ruta: Path) -> xr.Dataset:
    ds = xr.open_dataset(ruta)
    # Normaliza nombres de coordenadas habituales y longitudes 0-360 frente a -180-180
    ren = {k: v for k, v in {"valid_time": "time", "latitude": "lat", "longitude": "lon"}.items() if k in ds.coords or k in ds.dims}
    ds = ds.rename(ren)
    if "lon" in ds.coords and float(ds["lon"].max()) > 180:
        ds = ds.assign_coords(lon=(((ds["lon"] + 180) % 360) - 180)).sortby("lon")
    return ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--consolidado", type=Path, required=True)
    ap.add_argument("--intermedio", type=Path, required=True)
    ap.add_argument("--umbral-fwi", type=float, default=5.0, help="diferencia absoluta de FWI considerada relevante")
    a = ap.parse_args()

    c, i = abrir(a.consolidado), abrir(a.intermedio)
    comunes = sorted(set(c.data_vars) & set(i.data_vars))
    print(f"Variables en ambos: {comunes}")
    if not comunes:
        raise SystemExit(f"Sin variables comunes. Consolidado: {list(c.data_vars)} · intermedio: {list(i.data_vars)}")
    c, i = xr.align(c[comunes], i[comunes], join="inner")
    n_tiempos = c.sizes.get("time", 0)
    if not c.sizes or min(c.sizes.values()) == 0:
        raise SystemExit("No hay solapamiento de fechas/celdas entre ambos ficheros")
    print(f"Solapamiento: {dict(c.sizes)}\n")

    filas = {}
    for v in comunes:
        a_, b_ = c[v].values.ravel().astype(float), i[v].values.ravel().astype(float)
        ok = np.isfinite(a_) & np.isfinite(b_)
        a_, b_ = a_[ok], b_[ok]
        d = b_ - a_
        filas[v] = {"n": len(d), "dif_media": d.mean(), "dif_abs_media": np.abs(d).mean(), "dif_abs_p95": np.percentile(np.abs(d), 95),
                    "dif_abs_max": np.abs(d).max(), "correlación": np.corrcoef(a_, b_)[0, 1],
                    "media_consolidado": a_.mean()}
    print(pd.DataFrame(filas).T.round(3).to_string())

    # Cuántas celdas-día cambian lo bastante como para alterar el nivel de riesgo (solo para el FWI)
    nombre_fwi = next((v for v in comunes if v.lower().startswith("fwi")), None)
    if nombre_fwi:
        d = np.abs((i[nombre_fwi] - c[nombre_fwi]).values)
        d = d[np.isfinite(d)]
        print(f"\n{nombre_fwi}: {np.mean(d > a.umbral_fwi):.1%} de las celdas-día difieren más de {a.umbral_fwi:g} puntos "
              f"({n_tiempos} días comparados)")
    print("\nLectura: si la correlación es ~1 y la diferencia p95 es pequeña frente a la media, el intermedio es un buen "
          "sustituto. Si no, anótalo en DEC-34/DEC-48 y valora recalcular las predicciones cuando se consolide el mes.")


if __name__ == "__main__":
    main()
