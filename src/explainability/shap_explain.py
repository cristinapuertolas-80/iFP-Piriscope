"""Explicabilidad con SHAP (HU-08).

Produce `ExplicacionPrediccion` (Gold): formato largo con las columnas
(id_prediccion, variable, valor_shap), con las `top_k` variables de mayor
|SHAP| por predicción, y la importancia global de variables.

Unidades de valor_shap:
- XGBoost: log-odds (margen del modelo). Signo + => sube el riesgo.
- Random Forest: probabilidad. Signo + => sube el riesgo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap

from ..ml.config import COL_FECHA, COL_MUNICIPIO, EXPLICACIONES_PARQUET, SEED
from ..ml.train import ModelBundle

EXPL_COLUMNS = ["id_prediccion", "variable", "valor_shap"]


def shap_units(bundle: ModelBundle) -> str:
    return "log-odds" if bundle.algorithm == "xgboost" else "probabilidad"


def compute_shap(bundle: ModelBundle, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Valores SHAP (n_filas x n_features) de la clase 'incendio' y valor base."""
    explainer = shap.TreeExplainer(bundle.model)
    sv = explainer.shap_values(X[bundle.features])
    if isinstance(sv, list):          # versiones antiguas: una matriz por clase
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (n, features, clases)
        sv = sv[:, :, 1]
    base = float(np.atleast_1d(explainer.expected_value)[-1])
    return sv, base


def build_explanations(pred: pd.DataFrame, df: pd.DataFrame, bundle: ModelBundle,
                       top_k: int | None = 5) -> pd.DataFrame:
    """Explica cada predicción con las `top_k` variables de mayor |SHAP|.

    Cada predicción (fecha_objetivo = D+1) se explica con la fila del día D.
    `top_k=None` conserva todas las variables.
    """
    keys = pred[["id_prediccion", "id_municipio", "fecha_objetivo"]].copy()
    keys["id_municipio"] = keys["id_municipio"].astype(str)
    keys[COL_FECHA] = pd.to_datetime(keys["fecha_objetivo"]).dt.normalize() - pd.Timedelta(days=1)

    src = df[[COL_MUNICIPIO, COL_FECHA] + bundle.features].copy()
    src[COL_MUNICIPIO] = src[COL_MUNICIPIO].astype(str)
    src[COL_FECHA] = pd.to_datetime(src[COL_FECHA]).dt.normalize()

    try:
        m = keys.merge(src, on=[COL_MUNICIPIO, COL_FECHA], how="inner", validate="one_to_one")
    except pd.errors.MergeError as e:
        raise ValueError(
            "Hay municipio-día duplicados en las predicciones o en el dataset") from e
    if len(m) != len(keys):
        raise ValueError(
            f"{len(keys) - len(m)} predicciones no tienen su fila del día D en el dataset")

    S, _ = compute_shap(bundle, m)
    k = len(bundle.features) if top_k is None else min(top_k, len(bundle.features))
    order = np.argsort(-np.abs(S), axis=1)[:, :k]
    vals = np.take_along_axis(S, order, axis=1)
    names = np.asarray(bundle.features)[order]

    return pd.DataFrame({
        "id_prediccion": np.repeat(m["id_prediccion"].to_numpy(), k),
        "variable": names.ravel(),
        "valor_shap": vals.ravel(),
    })[EXPL_COLUMNS]


def global_importance(bundle: ModelBundle, X: pd.DataFrame, max_rows: int = 20_000,
                      seed: int = SEED) -> pd.DataFrame:
    """Importancia global: media de |SHAP| por variable (con muestreo reproducible)."""
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
    S, _ = compute_shap(bundle, X)
    out = pd.DataFrame({
        "variable": bundle.features,
        "importancia_shap": np.abs(S).mean(axis=0),
    })
    return out.sort_values("importancia_shap", ascending=False).reset_index(drop=True)


def write_explanations(expl: pd.DataFrame, path: Path = EXPLICACIONES_PARQUET) -> Path:
    """Escribe en Parquet; sobrescribe las explicaciones de los mismos id_prediccion."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~old["id_prediccion"].isin(expl["id_prediccion"])]
        expl = pd.concat([old, expl], ignore_index=True)
    expl.to_parquet(path, index=False)
    return path
