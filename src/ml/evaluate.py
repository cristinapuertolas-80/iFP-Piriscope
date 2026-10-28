"""Evaluación del modelo frente al baseline FWI (HU-07).

Principios:
- Modelo y baseline se evalúan SIEMPRE sobre exactamente las mismas filas.
- Métricas: AUC-PR (independiente del umbral) y Recall (a un umbral fijado
  en VALIDACIÓN, nunca en el conjunto que se está evaluando).
- Intervalos de confianza por bootstrap.
- Desagregación por provincia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .config import BASELINE_COL, COL_PROVINCIA, SEED, TARGET

SISTEMAS = ("modelo", "baseline_fwi")


# --- Métricas básicas -------------------------------------------------------
def auc_pr(y, score) -> float:
    """Área bajo la curva precisión-recall (average precision). NaN si no hay positivos."""
    y = np.asarray(y)
    if y.sum() == 0:
        return float("nan")
    return float(average_precision_score(y, np.asarray(score)))


def threshold_for_recall(y, score, target_recall: float = 0.8) -> float:
    """Mayor umbral (score >= umbral => alerta) que alcanza al menos `target_recall`.

    Debe calcularse sobre VALIDACIÓN y reutilizarse en test.
    """
    if not 0 < target_recall <= 1:
        raise ValueError("`target_recall` debe estar en (0, 1]")
    y, score = np.asarray(y), np.asarray(score)
    pos = np.sort(score[y == 1])
    if pos.size == 0:
        raise ValueError("No hay positivos para fijar el umbral")
    k = int(np.ceil(target_recall * pos.size))
    return float(pos[pos.size - k])


def metrics(y, score, threshold: float) -> dict:
    """AUC-PR, Recall y Precision (a `threshold`) más tamaños de muestra."""
    y, score = np.asarray(y), np.asarray(score)
    pred = score >= threshold
    n_pos = int(y.sum())
    tp = int((pred & (y == 1)).sum())
    n_alert = int(pred.sum())
    return {
        "n": int(y.size),
        "n_pos": n_pos,
        "prevalencia": n_pos / y.size if y.size else float("nan"),
        "auc_pr": auc_pr(y, score),
        "recall": tp / n_pos if n_pos else float("nan"),
        "precision": tp / n_alert if n_alert else float("nan"),
        "umbral": float(threshold),
    }


# --- Bootstrap --------------------------------------------------------------
class _Remuestreo:
    """Genera remuestras de filas; con `bloques`, remuestrea bloques completos.

    Los días con incendio se agrupan en olas (mismo tiempo en muchos municipios a la vez y en
    días seguidos), así que remuestrear filas sueltas da intervalos demasiado estrechos. Con
    `bloques` (p. ej. semanas) se remuestrean periodos enteros y el intervalo es más honesto.
    """

    def __init__(self, n: int, bloques, seed: int):
        self.rng, self.n = np.random.default_rng(seed), n
        self.orden = None
        if bloques is not None:
            _, inv = np.unique(np.asarray(bloques), return_inverse=True)
            self.orden = np.argsort(inv, kind="stable")
            self.lim = np.concatenate([[0], np.cumsum(np.bincount(inv))])
            self.g = len(self.lim) - 1

    def indices(self) -> np.ndarray:
        if self.orden is None:
            return self.rng.integers(0, self.n, self.n)
        elegidos = self.rng.integers(0, self.g, self.g)
        return np.concatenate([self.orden[self.lim[k]:self.lim[k + 1]] for k in elegidos])



def bootstrap_ci(y, score, threshold: float, n_boot: int = 500,
                 alpha: float = 0.05, seed: int = SEED, bloques=None) -> dict:
    """IC (percentiles) de AUC-PR y Recall por bootstrap.

    Sin `bloques` se remuestrean filas (intervalos optimistas: ignora la correlación entre días
    y municipios). Con `bloques` (una etiqueta por fila, p. ej. la semana) se remuestrean
    bloques completos. Se descartan las remuestras sin positivos.
    """
    y, score = np.asarray(y), np.asarray(score)
    remuestreo = _Remuestreo(y.size, bloques, seed)
    aps, recs = [], []
    for _ in range(n_boot):
        idx = remuestreo.indices()
        yb, sb = y[idx], score[idx]
        npos = yb.sum()
        if npos == 0:
            continue
        aps.append(average_precision_score(yb, sb))
        recs.append(((sb >= threshold) & (yb == 1)).sum() / npos)
    if not aps:
        nan = (float("nan"), float("nan"))
        return {"auc_pr": nan, "recall": nan}
    q = [alpha / 2, 1 - alpha / 2]
    return {
        "auc_pr": tuple(float(v) for v in np.quantile(aps, q)),
        "recall": tuple(float(v) for v in np.quantile(recs, q)),
    }


def diferencia_auc_pr(y, score_a, score_b, n_boot: int = 500, alpha: float = 0.05,
                      seed: int = SEED, bloques=None) -> dict:
    """Diferencia de AUC-PR (a - b) con IC por bootstrap PAREADO.

    Las dos puntuaciones se evalúan en las mismas remuestras de filas, así que el intervalo de la
    diferencia es mucho más estrecho que comparar los IC de cada sistema por separado.
    `p_mejora` = fracción de remuestras en las que `a` supera a `b`.
    """
    y, a, b = np.asarray(y), np.asarray(score_a), np.asarray(score_b)
    nan = {"dif": float("nan"), "lo": float("nan"), "hi": float("nan"), "p_mejora": float("nan")}
    if y.size == 0 or y.sum() == 0:
        return nan
    remuestreo = _Remuestreo(y.size, bloques, seed)
    difs = []
    for _ in range(n_boot):
        idx = remuestreo.indices()
        yb = y[idx]
        if yb.sum() == 0:
            continue
        difs.append(average_precision_score(yb, a[idx]) - average_precision_score(yb, b[idx]))
    if not difs:
        return nan
    difs = np.asarray(difs)
    lo, hi = np.quantile(difs, [alpha / 2, 1 - alpha / 2])
    return {"dif": auc_pr(y, a) - auc_pr(y, b), "lo": float(lo), "hi": float(hi),
            "p_mejora": float((difs > 0).mean())}


# --- Umbrales y comparación -------------------------------------------------
def thresholds_from_validation(val_df: pd.DataFrame, model_col: str = "proba",
                               baseline_col: str = BASELINE_COL,
                               target_recall: float = 0.8) -> dict:
    """Umbrales de modelo y baseline fijados en validación para el mismo Recall objetivo.

    Con el mismo Recall objetivo, la comparación en test se lee en Precision:
    ¿cuántas falsas alarmas cuesta cada sistema?
    """
    d = val_df.dropna(subset=[TARGET, model_col, baseline_col])
    return {
        "modelo": threshold_for_recall(d[TARGET], d[model_col], target_recall),
        "baseline_fwi": threshold_for_recall(d[TARGET], d[baseline_col], target_recall),
    }


def compare_model_vs_baseline(df: pd.DataFrame, thresholds: dict,
                              model_col: str = "proba",
                              baseline_col: str = BASELINE_COL,
                              n_boot: int = 500, by_province: bool = True,
                              seed: int = SEED, col_bloque: str | None = None) -> pd.DataFrame:
    """Tabla comparativa modelo vs baseline FWI sobre exactamente las mismas filas.

    `df` debe ser el subconjunto de un escenario (ver splits.select_scenario)
    con la probabilidad del modelo en `model_col`. Las filas sin FWI o sin
    probabilidad se descartan para AMBOS sistemas.
    """
    need = ([TARGET, model_col, baseline_col] + ([COL_PROVINCIA] if by_province else [])
            + ([col_bloque] if col_bloque else []))
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")
    if set(thresholds) != set(SISTEMAS):
        raise ValueError(f"`thresholds` debe tener las claves {SISTEMAS}")

    d = df.dropna(subset=[TARGET, model_col, baseline_col])
    groups = [("global", d)]
    if by_province:
        groups += [(p, g) for p, g in d.groupby(COL_PROVINCIA)]

    cols = {"modelo": model_col, "baseline_fwi": baseline_col}
    rows = []
    for grupo, g in groups:
        for sistema in SISTEMAS:
            y, s, thr = g[TARGET].to_numpy(), g[cols[sistema]].to_numpy(), thresholds[sistema]
            row = {"grupo": grupo, "sistema": sistema, **metrics(y, s, thr)}
            bl = g[col_bloque].to_numpy() if col_bloque else None
            ci = bootstrap_ci(y, s, thr, n_boot=n_boot, seed=seed, bloques=bl)
            row["auc_pr_lo"], row["auc_pr_hi"] = ci["auc_pr"]
            row["recall_lo"], row["recall_hi"] = ci["recall"]
            rows.append(row)
    out = pd.DataFrame(rows)
    return out[["grupo", "sistema", "n", "n_pos", "prevalencia", "auc_pr", "auc_pr_lo",
                "auc_pr_hi", "recall", "recall_lo", "recall_hi", "precision", "umbral"]]
