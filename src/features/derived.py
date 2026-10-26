"""Variables derivadas (HU-05). Se calculan en Gold, por municipio y en orden cronológico.

Regla de oro: el valor de una derivada en el día D solo puede usar datos de los días
<= D (el modelo predice D+1). Un test comprueba que cambiar datos posteriores a D no
altera ninguna derivada de D.

Las tres variables asumen una serie DIARIA sin huecos por municipio (Silver la garantiza:
una fila por municipio y día, con nulos donde no hay dato). Los días sin dato NO se imputan.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COL_MUN, COL_FECHA, COL_PRECIP = "id_municipio", "fecha", "precipitacion"


def comprobar_serie_diaria(df: pd.DataFrame) -> None:
    """Exige orden por (municipio, fecha) y días consecutivos dentro de cada municipio."""
    orden = df.sort_values([COL_MUN, COL_FECHA], kind="stable").index
    if not (orden == df.index).all():
        raise ValueError("El dataframe debe estar ordenado por id_municipio y fecha")
    if df[COL_MUN].isna().any():
        raise ValueError("Hay filas sin id_municipio")
    saltos = df.groupby(COL_MUN)[COL_FECHA].diff().dropna()
    if not (saltos == pd.Timedelta(days=1)).all():
        raise ValueError("Hay municipios con días repetidos o saltos en la serie diaria")


def _precip(df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df[COL_PRECIP]).to_numpy(dtype="float64", na_value=np.nan)


def racha_dias_secos(df: pd.DataFrame, umbral_seco_mm: float = 1.0) -> np.ndarray:
    """Días consecutivos, hasta el día D inclusive, con precipitación < `umbral_seco_mm`.

    - Un día con precipitación >= umbral reinicia la racha a 0.
    - Un día sin dato hace la racha DESCONOCIDA (NaN) y sigue siéndolo hasta el siguiente día
      lluvioso, que la reinicia con certeza. No se supone que los días sin dato fueran secos.
    - Al inicio de la serie de cada municipio la cuenta arranca en 1 (no se conoce el pasado).
    """
    p = _precip(df)
    nan = np.isnan(p)
    evento = (p >= umbral_seco_mm) | nan            # días que cortan la cuenta (lluvia o sin dato)
    idx = np.arange(len(p))
    # Con dtype 'string' (Silver), .ne(.shift()) devuelve <NA> en la primera fila: se compara en numpy
    ids = df[COL_MUN].to_numpy(dtype=object)
    cambia = np.ones(len(ids), dtype=bool)
    cambia[1:] = ids[1:] != ids[:-1]
    inicio = np.maximum.accumulate(np.where(cambia, idx, 0))   # primera fila de su municipio
    ultimo = np.maximum.accumulate(np.where(evento, idx, inicio - 1))
    en_municipio = ultimo >= inicio                  # el último corte es del mismo municipio
    desconocida = en_municipio & nan[np.clip(ultimo, 0, None)]
    racha = (idx - ultimo).astype("float64")
    racha[desconocida | nan] = np.nan
    return racha


def precip_acum_7d(df: pd.DataFrame, dias: int = 7, min_dias: int = 7) -> pd.Series:
    """Precipitación acumulada (mm) en los `dias` días que terminan en D (D incluido).

    Es NaN si en la ventana hay menos de `min_dias` días con dato (por defecto, ninguno puede faltar).
    """
    p = pd.Series(_precip(df), index=df.index)
    return p.groupby(df[COL_MUN]).transform(lambda s: s.rolling(dias, min_periods=min_dias).sum())


def anadir_derivadas(df: pd.DataFrame, umbral_seco_mm: float = 1.0, min_dias_acum: int = 7) -> pd.DataFrame:
    """Devuelve una copia con `racha_dias_secos`, `precip_acum_7d` y `mes`."""
    comprobar_serie_diaria(df)
    out = df.copy()
    out["racha_dias_secos"] = racha_dias_secos(out, umbral_seco_mm)
    out["precip_acum_7d"] = precip_acum_7d(out, 7, min_dias_acum).astype("float64")
    out["mes"] = out[COL_FECHA].dt.month.astype("int8")
    return out
