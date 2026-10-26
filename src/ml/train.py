"""Entrenamiento de modelos (HU-06): XGBoost y Random Forest.

- Solo se entrena con filas reservado = 0 y particion = entrenamiento.
- La validación (reservado = 0, particion = validacion) solo se usa para el
  early stopping de XGBoost. El test no se toca nunca aquí.
- Desbalanceo: scale_pos_weight (XGBoost) / class_weight (Random Forest).
- Cada modelo se guarda con versión, semilla y metadatos en models/.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .config import (
    MODELS_DIR, PART_VAL, SEED, TARGET, FEATURE_SETS,
    get_features, validate_schema,
)
from .splits import check_partition_integrity, select_scenario, select_train

ALGORITHMS = ("xgboost", "random_forest")

DEFAULT_PARAMS: dict[str, dict] = {
    "xgboost": {
        "n_estimators": 500, "learning_rate": 0.05, "max_depth": 5,
        "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 5,
    },
    "random_forest": {
        "n_estimators": 300, "max_depth": 12, "min_samples_leaf": 20,
        "max_features": "sqrt",
    },
}


@dataclass
class ModelBundle:
    """Modelo entrenado + todo lo necesario para reproducirlo y usarlo."""
    model: Any
    features: list[str]
    algorithm: str
    variant: str
    version: str
    seed: int
    trained_at: str
    n_train: int
    n_pos_train: int
    params: dict = field(default_factory=dict)
    best_iteration: int | None = None
    anio_desde: int | None = None
    excluidas: list[str] = field(default_factory=list)

    @property
    def model_id(self) -> str:
        return f"{self.version}_{self.algorithm}_{self.variant}"

    def metadata(self) -> dict:
        d = asdict(self)
        d.pop("model")
        d["model_id"] = self.model_id
        return d


def _build_model(algorithm: str, y: pd.Series, seed: int, params: dict | None):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Algoritmo {algorithm!r} no válido. Opciones: {ALGORITHMS}")
    p = {**DEFAULT_PARAMS[algorithm], **(params or {})}
    if algorithm == "xgboost":
        n_pos = int(y.sum())
        p.setdefault("scale_pos_weight", (len(y) - n_pos) / n_pos)
        model = XGBClassifier(
            eval_metric="aucpr", tree_method="hist", random_state=seed,
            n_jobs=-1, **p,
        )
    else:
        p.setdefault("class_weight", "balanced_subsample")
        model = RandomForestClassifier(random_state=seed, n_jobs=-1, **p)
    return model, p


def train_model(df: pd.DataFrame, algorithm: str = "xgboost", variant: str = "con_fwi",
                version: str = "v1", seed: int = SEED, params: dict | None = None,
                early_stopping_rounds: int | None = 50, anio_desde: int | None = None,
                excluir: list[str] | None = None) -> ModelBundle:
    """Entrena un modelo sobre la capa Gold y devuelve un ModelBundle.

    `anio_desde` recorta los años de entrenamiento más antiguos; `excluir` quita variables de la
    variante. Ambos quedan en el nombre (`bundle.variant`) y en los metadatos del modelo.
    """
    validate_schema(df)
    check_partition_integrity(df)

    feats = get_features(variant)
    excluir = list(excluir or [])
    conocidas = {f for fs in FEATURE_SETS.values() for f in fs}
    if set(excluir) - conocidas:
        raise ValueError(f"Variables a excluir desconocidas: {sorted(set(excluir) - conocidas)}")
    quitadas = [f for f in feats if f in excluir]
    feats = [f for f in feats if f not in excluir]
    if not feats:
        raise ValueError("Tras excluir variables no queda ninguna para entrenar")
    etiqueta = (variant + (f"-sin_{'_'.join(sorted(quitadas))}" if quitadas else "")
                + (f"-desde{anio_desde}" if anio_desde else ""))
    train = select_train(df, anio_desde)
    y = train[TARGET]
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        raise ValueError("El entrenamiento necesita ejemplos de las dos clases")

    model, used_params = _build_model(algorithm, y, seed, params)
    X = train[feats]

    best_iter = None
    if algorithm == "xgboost":
        val = select_scenario(df, "A", split=PART_VAL)
        if early_stopping_rounds and val[TARGET].sum() > 0:
            model.set_params(early_stopping_rounds=early_stopping_rounds)
            model.fit(X, y, eval_set=[(val[feats], val[TARGET])], verbose=False)
            best_iter = int(model.best_iteration)
        else:
            model.fit(X, y, verbose=False)
    else:
        model.fit(X, y)

    return ModelBundle(
        model=model, features=feats, algorithm=algorithm, variant=etiqueta,
        version=version, seed=seed,
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        n_train=len(train), n_pos_train=n_pos, params=used_params,
        best_iteration=best_iter, anio_desde=anio_desde, excluidas=quitadas,
    )


def train_all_variants(df: pd.DataFrame, algorithm: str = "xgboost", version: str = "v1",
                       **kwargs) -> dict[str, ModelBundle]:
    """Entrena las tres variantes (con_fwi, sin_fwi, solo_meteo) para HU-07."""
    return {v: train_model(df, algorithm, v, version, **kwargs) for v in FEATURE_SETS}


def score_dataframe(bundle: ModelBundle, df: pd.DataFrame, col: str = "proba") -> pd.DataFrame:
    """Copia de `df` con la probabilidad de incendio en `col` (listo para evaluate.py)."""
    out = df.copy()
    out[col] = bundle.model.predict_proba(out[bundle.features])[:, 1]
    return out


def save_model(bundle: ModelBundle, models_dir: Path = MODELS_DIR) -> Path:
    """Guarda modelo (.joblib) y metadatos legibles (.json) en models/."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"{bundle.model_id}.joblib"
    joblib.dump(bundle, path)
    path.with_suffix(".json").write_text(
        json.dumps(bundle.metadata(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return path


def load_model(path: Path) -> ModelBundle:
    """Carga un ModelBundle. Solo cargar ficheros generados por el propio proyecto."""
    bundle = joblib.load(path)
    if not isinstance(bundle, ModelBundle):
        raise TypeError(f"{path} no contiene un ModelBundle")
    return bundle
