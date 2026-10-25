"""Contrato de datos y configuración del módulo ML de Piriscope.

Fuente de verdad: Anexo A (diccionario de variables). Si el EDA cambia las
variables definitivas, se modifica SOLO este fichero.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# --- Reproducibilidad -------------------------------------------------------
SEED = 42

# --- Rutas ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
SILVER_DIR = ROOT / "data" / "silver"
GOLD_DIR = ROOT / "data" / "gold"
MODELS_DIR = ROOT / "models"
RESERVADOS_CSV = ROOT / "docs" / "municipios_reservados.csv"
RESERVADOS_META_JSON = ROOT / "docs" / "municipios_reservados_meta.json"
PREDICCIONES_PARQUET = GOLD_DIR / "predicciones.parquet"
EXPLICACIONES_PARQUET = GOLD_DIR / "explicaciones.parquet"

# --- Columnas ---------------------------------------------------------------
TARGET = "incendio_d1"
COL_RESERVADO = "reservado"
COL_PARTICION = "particion"
COL_FECHA = "fecha"
COL_MUNICIPIO = "id_municipio"
COL_PROVINCIA = "provincia"

# Valores válidos de `particion` (sin tilde a propósito, para evitar problemas de encoding)
PART_TRAIN = "entrenamiento"
PART_VAL = "validacion"
PART_TEST = "test"
PARTICIONES = (PART_TRAIN, PART_VAL, PART_TEST)

# Niveles de riesgo, de menor a mayor
RISK_LEVELS = ("bajo", "medio", "alto")

# --- Grupos de variables ----------------------------------------------------
METEO_FEATURES = ["temperatura", "humedad_relativa", "viento", "precipitacion"]
DERIVED_FEATURES = ["racha_dias_secos", "precip_acum_7d", "mes"]
FWI_FEATURES = ["fwi", "ffmc", "dmc", "dc", "isi", "bui"]

# Variantes del modelo (HU-07): la variante sin FWI evita que la comparación
# con el baseline sea circular.
FEATURE_SETS: dict[str, list[str]] = {
    "con_fwi": METEO_FEATURES + DERIVED_FEATURES + FWI_FEATURES,
    "sin_fwi": METEO_FEATURES + DERIVED_FEATURES,
    "solo_meteo": METEO_FEATURES,
}

# Columnas que NUNCA pueden entrar al modelo (identificadores, coordenadas,
# información que solo se conoce a posteriori, trazabilidad, etc.)
FORBIDDEN_FEATURES = [
    "id_municipio", "id_estacion_aemet", "id_obs_meteo", "id_evento",
    "id_indicador_fwi", "id_prediccion", "nombre", "provincia",
    "lat_centroide", "lon_centroide", "latitud", "longitud", "altitud",
    "coordenada_x", "coordenada_y", "huso", "datum",
    "fecha_deteccion", "fecha_extincion", "causa", "superficie_forestal_ha",
    "fecha", "fecha_objetivo", "fecha_insert", "version_modelo",
    "reservado", "particion", "dist_estacion_km", TARGET,
]

# Columna del baseline (FWI de EFFIS/Copernicus)
BASELINE_COL = "fwi"

# --- Esquema mínimo exigido a la tabla de entrenamiento (capa Gold) ----------
REQUIRED_COLUMNS = (
    [COL_MUNICIPIO, COL_PROVINCIA, COL_FECHA, COL_RESERVADO, COL_PARTICION, TARGET]
    + METEO_FEATURES + DERIVED_FEATURES + FWI_FEATURES
)


def get_features(variant: str) -> list[str]:
    """Devuelve la lista de features de una variante, comprobando que no hay prohibidas."""
    if variant not in FEATURE_SETS:
        raise KeyError(f"Variante desconocida: {variant!r}. Opciones: {list(FEATURE_SETS)}")
    feats = FEATURE_SETS[variant]
    bad = set(feats) & set(FORBIDDEN_FEATURES)
    if bad:
        raise ValueError(f"Features prohibidas en la variante {variant!r}: {sorted(bad)}")
    return list(feats)


def validate_schema(df: pd.DataFrame) -> None:
    """Valida que el dataset cumple el contrato. Lanza ValueError si no."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el dataset: {missing}")

    if not set(df[COL_PARTICION].dropna().unique()) <= set(PARTICIONES):
        raise ValueError(f"`{COL_PARTICION}` solo admite {PARTICIONES}")
    if df[COL_PARTICION].isna().any():
        raise ValueError(f"`{COL_PARTICION}` contiene nulos")
    if not set(df[COL_RESERVADO].dropna().unique()) <= {0, 1}:
        raise ValueError(f"`{COL_RESERVADO}` debe ser 0/1")
    if not set(df[TARGET].dropna().unique()) <= {0, 1}:
        raise ValueError(f"`{TARGET}` debe ser 0/1")
    if df[TARGET].isna().any():
        raise ValueError(f"`{TARGET}` contiene nulos")
    if df.duplicated([COL_MUNICIPIO, COL_FECHA]).any():
        raise ValueError("Hay filas duplicadas por (id_municipio, fecha)")
