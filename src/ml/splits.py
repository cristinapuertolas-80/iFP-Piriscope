"""Selección de filas por escenario (Anexo A, 'Escenarios de evaluación').

Entrenamiento : reservado = 0  y particion = entrenamiento
A (temporal)  : reservado = 0  y particion in (validacion, test)
B (geográfico): reservado = 1  y particion = entrenamiento
C (temp.+geo.): reservado = 1  y particion in (validacion, test)
"""
from __future__ import annotations

import pandas as pd

from .config import (
    COL_FECHA, COL_PARTICION, COL_RESERVADO, PART_TEST, PART_TRAIN, PART_VAL,
)

SCENARIOS = ("A", "B", "C")


def select_train(df: pd.DataFrame) -> pd.DataFrame:
    """Filas de entrenamiento: municipios no reservados, periodo de entrenamiento."""
    return df[(df[COL_RESERVADO] == 0) & (df[COL_PARTICION] == PART_TRAIN)]


def select_scenario(df: pd.DataFrame, scenario: str, split: str | None = None) -> pd.DataFrame:
    """Filas de un escenario de evaluación.

    `split` permite restringir a 'validacion' o 'test' en los escenarios A y C
    (p. ej. validación para elegir umbrales, test solo para el informe final).
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Escenario {scenario!r} no válido. Opciones: {SCENARIOS}")
    if split not in (None, PART_VAL, PART_TEST):
        raise ValueError(f"`split` debe ser None, {PART_VAL!r} o {PART_TEST!r}")

    if scenario == "B":
        if split is not None:
            raise ValueError("El escenario B usa el periodo de entrenamiento; no admite `split`")
        mask = (df[COL_RESERVADO] == 1) & (df[COL_PARTICION] == PART_TRAIN)
    else:
        reservado = 0 if scenario == "A" else 1
        parts = [split] if split else [PART_VAL, PART_TEST]
        mask = (df[COL_RESERVADO] == reservado) & (df[COL_PARTICION].isin(parts))
    return df[mask]


def check_partition_integrity(df: pd.DataFrame) -> None:
    """Comprobaciones de seguridad contra fuga de información.

    - Orden temporal estricto: entrenamiento < validación < test.
    - Ningún municipio reservado aparece en las filas de entrenamiento.
    - Cada municipio es siempre reservado o nunca (no se mezcla).
    """
    rng = df.groupby(COL_PARTICION)[COL_FECHA].agg(["min", "max"])
    if PART_TRAIN in rng.index and PART_VAL in rng.index:
        if rng.loc[PART_TRAIN, "max"] >= rng.loc[PART_VAL, "min"]:
            raise ValueError("Entrenamiento se solapa en el tiempo con validación")
    if PART_VAL in rng.index and PART_TEST in rng.index:
        if rng.loc[PART_VAL, "max"] >= rng.loc[PART_TEST, "min"]:
            raise ValueError("Validación se solapa en el tiempo con test")

    train = select_train(df)
    if (train[COL_RESERVADO] != 0).any():
        raise ValueError("Municipios reservados en el conjunto de entrenamiento")

    if "id_municipio" in df.columns:
        n_flags = df.groupby("id_municipio")[COL_RESERVADO].nunique()
        if (n_flags > 1).any():
            raise ValueError("Hay municipios con `reservado` inconsistente entre filas")
