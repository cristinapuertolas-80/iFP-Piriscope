"""Inferencia batch (D+1): genera las filas de `Prediccion` (Gold).

Entrada: dataset Gold con una fila por municipio y día D (features hasta D).
Salida : tabla con el esquema de `Prediccion` del Anexo A, para
data/gold/predicciones.parquet. El dashboard solo lee este fichero.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import BASELINE_COL, COL_FECHA, COL_MUNICIPIO, PREDICCIONES_PARQUET
from .risk import RiskThresholds, assign_levels
from .train import ModelBundle

PRED_COLUMNS = [
    "id_prediccion", "id_municipio", "fecha_objetivo", "version_modelo",
    "probabilidad", "nivel_riesgo", "fwi", "fecha_insert",
]


def build_predictions(df: pd.DataFrame, bundle: ModelBundle, thresholds: RiskThresholds,
                      desde=None, hasta=None, version_modelo: str | None = None,
                      now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Predice el día D+1 para cada municipio con fila en el día D.

    Sin `desde`/`hasta` se usa solo el último día disponible en `df`.
    El `fwi` copiado es el del día D (el que vio el modelo).
    """
    if thresholds.model_id != bundle.model_id:
        raise ValueError(
            f"Los umbrales son del modelo {thresholds.model_id!r}, "
            f"no de {bundle.model_id!r}")
    needed = [COL_MUNICIPIO, COL_FECHA, BASELINE_COL] + bundle.features
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas para predecir: {missing}")

    fechas = pd.to_datetime(df[COL_FECHA]).dt.normalize()
    if desde is None and hasta is None:
        desde = hasta = fechas.max()
    mask = pd.Series(True, index=df.index)
    if desde is not None:
        mask &= fechas >= pd.Timestamp(desde)
    if hasta is not None:
        mask &= fechas <= pd.Timestamp(hasta)
    d, f = df[mask], fechas[mask]
    if d.empty:
        raise ValueError("No hay filas en el rango de fechas pedido")

    ver = version_modelo or bundle.version
    fecha_obj = f + pd.Timedelta(days=1)
    proba = bundle.model.predict_proba(d[bundle.features])[:, 1]

    out = pd.DataFrame({
        "id_municipio": d[COL_MUNICIPIO].astype(str).to_numpy(),
        "fecha_objetivo": fecha_obj.to_numpy(),
        "version_modelo": ver,
        "probabilidad": proba,
        "fwi": d[BASELINE_COL].to_numpy(),
    })
    out["nivel_riesgo"] = assign_levels(out["probabilidad"], thresholds)
    out["id_prediccion"] = (out["id_municipio"] + "_"
                            + out["fecha_objetivo"].dt.strftime("%Y-%m-%d") + "_" + ver)
    if out["id_prediccion"].duplicated().any():
        raise ValueError("id_prediccion duplicado: ¿hay varias filas por municipio y día?")
    out["fecha_insert"] = now if now is not None else pd.Timestamp.now(tz="UTC")
    return out[PRED_COLUMNS].reset_index(drop=True)


def write_predictions(pred: pd.DataFrame, path: Path = PREDICCIONES_PARQUET) -> Path:
    """Escribe en Parquet. Sobrescribe las filas con el mismo id_prediccion
    (mismo municipio, fecha y versión) y conserva el resto."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~old["id_prediccion"].isin(pred["id_prediccion"])]
        pred = pd.concat([old, pred], ignore_index=True)
    pred = pred.sort_values(["fecha_objetivo", "id_municipio"]).reset_index(drop=True)
    pred.to_parquet(path, index=False)
    return path
