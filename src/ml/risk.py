"""Niveles de riesgo (bajo / medio / alto) a partir de la probabilidad.

Los umbrales se fijan sobre VALIDACIÓN (nunca en test) mediante Recall objetivo:
- umbral "medio": el mayor umbral con Recall >= recall_medio  (p. ej. 0,80)
- umbral "alto" : el mayor umbral con Recall >= recall_alto   (p. ej. 0,50)
Como más Recall exige un umbral más bajo, umbral_medio <= umbral_alto.
Así "medio o superior" captura ~80 % de los días con incendio de validación y
"alto" ~50 % con menos falsas alarmas.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import MODELS_DIR, RISK_LEVELS
from .evaluate import threshold_for_recall


@dataclass
class RiskThresholds:
    medio: float
    alto: float
    recall_medio: float
    recall_alto: float
    model_id: str
    n_val: int
    n_pos_val: int


def fit_risk_thresholds(y_val, proba_val, model_id: str,
                        recall_medio: float = 0.8, recall_alto: float = 0.5) -> RiskThresholds:
    """Fija los umbrales bajo/medio/alto sobre validación."""
    if not 0 < recall_alto < recall_medio <= 1:
        raise ValueError("Se exige 0 < recall_alto < recall_medio <= 1")
    y, p = np.asarray(y_val), np.asarray(proba_val)
    return RiskThresholds(
        medio=threshold_for_recall(y, p, recall_medio),
        alto=threshold_for_recall(y, p, recall_alto),
        recall_medio=recall_medio, recall_alto=recall_alto,
        model_id=model_id, n_val=int(y.size), n_pos_val=int(y.sum()),
    )


def assign_levels(proba, thresholds: RiskThresholds) -> pd.Series:
    """Convierte probabilidades en niveles 'bajo' / 'medio' / 'alto'."""
    p = np.asarray(proba, dtype=float)
    levels = np.where(p >= thresholds.alto, RISK_LEVELS[2],
                      np.where(p >= thresholds.medio, RISK_LEVELS[1], RISK_LEVELS[0]))
    return pd.Series(levels, dtype="object")


def save_thresholds(thresholds: RiskThresholds, models_dir: Path = MODELS_DIR) -> Path:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{thresholds.model_id}_umbrales.json"
    path.write_text(json.dumps(asdict(thresholds), indent=2), encoding="utf-8")
    return path


def load_thresholds(path: Path) -> RiskThresholds:
    return RiskThresholds(**json.loads(Path(path).read_text(encoding="utf-8")))
