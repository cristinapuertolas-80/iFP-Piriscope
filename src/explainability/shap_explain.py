"""Explicabilidad con SHAP (HU-08, DEC-43, DEC-50).

Produce `ExplicacionPrediccion` (Gold): formato largo con las `top_k` variables de mayor |SHAP| por
predicción, y la importancia global de variables (por variable o por grupos).

Columnas de `build_explanations`:
- Esquema base (`EXPL_COLUMNS`): id_prediccion, variable, valor_shap.
- Ampliación (`EXTRA_COLUMNS`, DEC-50), repetida en las filas de cada predicción:
    valor_variable  valor de la variable el día D (nulo si faltaba).
    valor_base      valor esperado del modelo (la «predicción media»), en las mismas unidades que valor_shap.
    shap_resto      suma de las contribuciones de las variables que NO están en el top-k.
  Con ellas se reconstruye la salida del modelo:
    XGBoost:       logit(probabilidad) = valor_base + sum(valor_shap) + shap_resto
    Random Forest: probabilidad        = valor_base + sum(valor_shap) + shap_resto

Unidades de valor_shap:
- XGBoost: log-odds (margen del modelo). Signo + => sube el riesgo.
- Random Forest: probabilidad. Signo + => sube el riesgo.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from ..ml.config import COL_FECHA, COL_MUNICIPIO, EXPLICACIONES_PARQUET, SEED
from ..ml.train import ModelBundle

EXPL_COLUMNS = ["id_prediccion", "variable", "valor_shap"]
EXTRA_COLUMNS = ["valor_variable", "valor_base", "shap_resto"]
EXPL_COLUMNS_EXT = EXPL_COLUMNS + EXTRA_COLUMNS

# Los índices FWI se calculan a partir de la meteorología y están muy correlacionados entre sí: SHAP reparte
# su efecto entre ellos. La importancia por grupos suma las contribuciones ANTES de tomar el valor absoluto.
GRUPOS = {
    "Índices FWI": ("fwi", "ffmc", "dmc", "dc", "isi", "bui"),
    "Meteorología del día": ("temperatura", "humedad_relativa", "precipitacion", "viento"),
    "Lluvia acumulada y sequía": ("racha_dias_secos", "precip_acum_7d"),
    "Calendario": ("mes",),
}


def shap_units(bundle: ModelBundle) -> str:
    return "log-odds" if bundle.algorithm == "xgboost" else "probabilidad"


def _sigmoide(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


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


def _verificar_aditividad(S: np.ndarray, base: float, X: pd.DataFrame, bundle: ModelBundle,
                          tol: float = 1e-3) -> float | None:
    """Comprueba que valor_base + suma de SHAP reproduce la salida del propio modelo.

    Avisa (no falla) si la diferencia máxima supera `tol`: suele indicar que se explica con un modelo distinto
    del que predijo, o que las variables no coinciden. Devuelve la diferencia máxima (None si no se pudo comprobar).
    """
    if not hasattr(bundle.model, "predict_proba"):
        return None
    esperado = np.asarray(bundle.model.predict_proba(X[bundle.features]))[:, 1]
    suma = base + S.sum(axis=1)
    obtenido = _sigmoide(suma) if bundle.algorithm == "xgboost" else suma
    dif = float(np.max(np.abs(obtenido - esperado)))
    if dif > tol:
        warnings.warn(f"Los valores SHAP no son aditivos respecto al modelo (diferencia máxima {dif:.4g} > {tol}); "
                      "revisa que se explica con el mismo modelo que generó las predicciones", stacklevel=2)
    return dif


def build_explanations(pred: pd.DataFrame, df: pd.DataFrame, bundle: ModelBundle, top_k: int | None = 5,
                       con_valores: bool = True, verificar: bool = False) -> pd.DataFrame:
    """Explica cada predicción con las `top_k` variables de mayor |SHAP|.

    Cada predicción (fecha_objetivo = D+1) se explica con la fila del día D.
    `top_k=None` conserva todas las variables (y `shap_resto` es 0).
    `con_valores=False` devuelve solo el esquema base de tres columnas.
    `verificar=True` comprueba la aditividad con el modelo y avisa si no se cumple.
    """
    if top_k is not None and top_k < 1:
        raise ValueError("top_k debe ser al menos 1 (o None para conservar todas las variables)")
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

    S, base = compute_shap(bundle, m)
    if verificar:
        _verificar_aditividad(S, base, m, bundle)
    k = len(bundle.features) if top_k is None else min(top_k, len(bundle.features))
    order = np.argsort(-np.abs(S), axis=1)[:, :k]
    vals = np.take_along_axis(S, order, axis=1)
    names = np.asarray(bundle.features)[order]

    out = {
        "id_prediccion": np.repeat(m["id_prediccion"].to_numpy(), k),
        "variable": names.ravel(),
        "valor_shap": vals.ravel(),
    }
    if not con_valores:
        return pd.DataFrame(out)[EXPL_COLUMNS]
    X = m[bundle.features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    out["valor_variable"] = np.take_along_axis(X, order, axis=1).ravel()
    out["valor_base"] = np.full(len(m) * k, base)
    out["shap_resto"] = np.repeat(S.sum(axis=1) - vals.sum(axis=1), k)
    return pd.DataFrame(out)[EXPL_COLUMNS_EXT]


def global_importance(bundle: ModelBundle, X: pd.DataFrame, max_rows: int = 20_000,
                      seed: int = SEED, por_grupo: bool = False) -> pd.DataFrame:
    """Importancia global: media de |SHAP| (con muestreo reproducible).

    Con `por_grupo=True` agrupa las variables según `GRUPOS` sumando sus contribuciones antes del valor
    absoluto; las variables fuera de todo grupo conservan su nombre. La columna sigue llamándose `variable`.
    """
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
    S, _ = compute_shap(bundle, X)
    if not por_grupo:
        out = pd.DataFrame({"variable": bundle.features, "importancia_shap": np.abs(S).mean(axis=0)})
    else:
        pos = {f: i for i, f in enumerate(bundle.features)}
        filas, usadas = [], set()
        for grupo, miembros in GRUPOS.items():
            idx = [pos[f] for f in miembros if f in pos]
            if idx:
                filas.append((grupo, float(np.abs(S[:, idx].sum(axis=1)).mean())))
                usadas.update(idx)
        filas += [(f, float(np.abs(S[:, i]).mean())) for f, i in pos.items() if i not in usadas]
        out = pd.DataFrame(filas, columns=["variable", "importancia_shap"])
    return out.sort_values("importancia_shap", ascending=False).reset_index(drop=True)


def write_explanations(expl: pd.DataFrame, path: Path = EXPLICACIONES_PARQUET) -> Path:
    """Escribe en Parquet; sobrescribe las explicaciones de los mismos id_prediccion.

    Si el fichero existente es anterior a la ampliación (sin `EXTRA_COLUMNS`), sus filas conservan esas columnas
    como nulas: para tener todo completo, regenera con `inferir --reemplazar`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_parquet(path)
        old = old[~old["id_prediccion"].isin(expl["id_prediccion"])]
        expl = pd.concat([old, expl], ignore_index=True)
    columnas = [c for c in EXPL_COLUMNS_EXT if c in expl.columns] + [c for c in expl.columns if c not in EXPL_COLUMNS_EXT]
    expl[columnas].to_parquet(path, index=False)
    return path
