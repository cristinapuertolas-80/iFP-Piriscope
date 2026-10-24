"""Dataset de JUGUETE para tests de ML. Solo para desarrollo: nunca se usa en data/."""
import numpy as np
import pandas as pd

from src.ml.config import DERIVED_FEATURES, FWI_FEATURES, METEO_FEATURES, SEED


def make_toy_gold(n_mun: int = 8, seed: int = SEED, pos_rate: float = 0.02) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fechas = pd.date_range("2018-01-01", "2023-12-31", freq="D")
    muns = [f"{15 + (i % 4) * 1000 + i:05d}" for i in range(n_mun)]
    provs = {m: ["A Coruña", "Lugo", "Ourense", "Pontevedra"][i % 4] for i, m in enumerate(muns)}
    reservados = set(muns[-2:])  # los dos últimos, reservados

    df = pd.MultiIndex.from_product([muns, fechas], names=["id_municipio", "fecha"]).to_frame(index=False)
    n = len(df)
    df["provincia"] = df["id_municipio"].map(provs)
    df["reservado"] = df["id_municipio"].isin(reservados).astype(int)
    year = df["fecha"].dt.year
    df["particion"] = np.where(year <= 2021, "entrenamiento", np.where(year == 2022, "validacion", "test"))
    for c in METEO_FEATURES + DERIVED_FEATURES + FWI_FEATURES:
        df[c] = rng.normal(size=n)
    df["mes"] = df["fecha"].dt.month
    df["incendio_d1"] = (rng.random(n) < pos_rate).astype(int)
    return df