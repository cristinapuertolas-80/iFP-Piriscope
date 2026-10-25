"""Datos sintéticos de Bronze para los tests de Silver.

Dos municipios contiguos de A Coruña (cuadrados de 10 km en UTM 29N), dos
estaciones AEMET, un mes de FWI y unos pocos partes de EGIF.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

X0, Y0, LADO = 550_000, 4_750_000, 10_000
#: Centro en UTM 29N (ED50 ≈ ETRS89 a efectos de estos tests) de cada municipio
CENTRO = {"15001": (X0 + LADO / 2, Y0 + LADO / 2), "15002": (X0 + 1.5 * LADO, Y0 + LADO / 2)}


def recintos():
    import geopandas as gpd
    from shapely.geometry import box

    g = gpd.GeoDataFrame({
        "NATCODE": ["34121515001", "34121515002", "34070707001"],
        "NAMEUNIT": ["Abegondo", "A Cañiza", "Fuera de Galicia"],
    }, geometry=[box(X0, Y0, X0 + LADO, Y0 + LADO), box(X0 + LADO, Y0, X0 + 2 * LADO, Y0 + LADO),
                 box(X0 + 5 * LADO, Y0, X0 + 6 * LADO, Y0 + LADO)], crs="EPSG:25829")
    return g.to_crs("EPSG:4258")


def municipios_df():
    """Tabla Municipio como la produce Silver (para tests unitarios)."""
    from src.silver import municipios

    ine = pd.DataFrame({"id_municipio": ["15001", "15002"], "id_provincia": ["15", "15"],
                        "nombre_ine": ["Abegondo", "Cañiza, A"]}).astype("string")
    return municipios.construir(ine, municipios.filtrar_recintos(recintos()))


def escribir_ine_cnig(run: Path) -> None:
    (run / "ine").mkdir(parents=True)
    (run / "cnig").mkdir(parents=True)
    filas = [["Relación de municipios a 1 de enero", None, None, None, None],
             ["CODAUTO", "CPRO", "CMUN", "DC", "NOMBRE"],
             ["12", "15", "001", "6", "Abegondo"],
             ["12", "15", "002", "1", "Cañiza, A"],
             ["07", "07", "001", "1", "Fuera de Galicia"]]
    pd.DataFrame(filas).to_excel(run / "ine" / "diccionario26.xlsx", header=False, index=False)
    recintos().to_file(run / "cnig" / "limites.gpkg", layer="recintos_municipales_inspire_peninbal_etrs89")


def escribir_aemet(run: Path, registros: dict[str, list[dict]], estaciones: list[dict]) -> None:
    (run / "diarios").mkdir(parents=True)
    for nombre, regs in registros.items():
        (run / "diarios" / nombre).write_bytes(json.dumps(regs).encode("ISO-8859-15"))
    pd.DataFrame(estaciones).to_csv(run / "estaciones_galicia.csv", index=False)
    pd.DataFrame([{"indicativo": e["indicativo"], "dias_con_datos": 1} for e in estaciones]).to_csv(
        run / "cobertura_estaciones.csv", index=False)


def escribir_fwi(ruta: Path, fechas, valor: float, mar_en_primera_fila: bool = True) -> None:
    import xarray as xr

    lat = np.array([43.0, 42.75, 42.5])
    lon = np.array([351.25, 351.5, 351.75])  # 0–360, como el EWDS
    datos = np.full((len(fechas), 3, 3), valor, dtype="float32")
    if mar_en_primera_fila:
        datos[:, 0, :] = np.nan
    variables = {v: (("valid_time", "latitude", "longitude"), datos)
                 for v in ("fwinx", "ffmcode", "dufmcode", "drtcode", "infsinx", "fbupinx")}
    ruta.parent.mkdir(parents=True, exist_ok=True)
    xr.Dataset(variables, coords={"valid_time": pd.to_datetime(fechas), "latitude": lat,
                                  "longitude": lon}).to_netcdf(ruta)


def escribir_egif(run: Path, partes: list[dict]) -> None:
    run.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(partes).astype("string").to_parquet(run / "egif_galicia_unificado.parquet", index=False)


def parte(id_evento: str, municipio: str, detectado: str, x: float | None = None, y: float | None = None,
          provincia: str = "15", datum: str = "ETRS89", superficie: str = "1,5000") -> dict:
    return {"id_evento": id_evento, "campana": detectado[6:10], "numero_parte": id_evento[-4:],
            "id_provincia": provincia, "municipio": municipio, "detectado": detectado,
            "extinguido": detectado, "causa": "[400]  Intencionado", "superficietotalforestal": superficie,
            "coordenadax": "" if x is None else str(int(x)), "coordenaday": "" if y is None else str(int(y)),
            "huso": "29" if x is not None else "\xa0", "datum": datum}
