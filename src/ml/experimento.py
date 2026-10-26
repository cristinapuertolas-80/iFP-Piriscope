"""Experimento: entrena las variantes, las compara con el baseline FWI y escribe el informe (HU-06, HU-07).

Dos fases, para proteger el conjunto de test y la partición geográfica:

- DESARROLLO (por defecto): solo mira la VALIDACIÓN de municipios no reservados (escenario A).
  Sirve para elegir configuración. Nunca toca test ni municipios reservados.
- FINAL (`--final`): informe de A (test), B y C (test). Se ejecuta UNA vez, con la configuración
  ya decidida: mirar estos resultados y volver a cambiar el modelo contaminaría la evaluación.

Los umbrales de Recall (modelo y baseline) se fijan siempre sobre la validación de A.

Uso:
    python -m src.ml.experimento                         # desarrollo, XGBoost, tres variantes
    python -m src.ml.experimento --final --guardar       # informe final y modelos en models/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import config, evaluate, risk, splits, train

ESCENARIOS_DESARROLLO = {"A-validacion": ("A", config.PART_VAL)}
ESCENARIOS_FINALES = {"A-test": ("A", config.PART_TEST), "B": ("B", None), "C-test": ("C", config.PART_TEST)}
COLUMNAS_INFORME = ["escenario", "algoritmo", "variante", "grupo", "sistema", "n", "n_pos", "prevalencia",
                    "auc_pr", "auc_pr_lo", "auc_pr_hi", "recall", "recall_lo", "recall_hi", "precision", "umbral"]


def correr_variante(df: pd.DataFrame, algoritmo: str, variante: str, version: str, final: bool,
                    recall_objetivo: float, n_boot: int, por_provincia: bool,
                    params: dict | None = None, anio_desde: int | None = None,
                    excluir: list[str] | None = None):
    """Entrena una variante y devuelve (bundle, umbrales de riesgo, informe)."""
    bundle = train.train_model(df, algoritmo, variante, version, params=params,
                               anio_desde=anio_desde, excluir=excluir)
    val = train.score_dataframe(bundle, splits.select_scenario(df, "A", split=config.PART_VAL))
    umbrales = evaluate.thresholds_from_validation(val, target_recall=recall_objetivo)
    riesgo = risk.fit_risk_thresholds(val[config.TARGET], val["proba"], bundle.model_id)

    partes = []
    for nombre, (esc, split) in (ESCENARIOS_FINALES if final else ESCENARIOS_DESARROLLO).items():
        datos = val if nombre == "A-validacion" else train.score_dataframe(
            bundle, splits.select_scenario(df, esc, split=split))
        if datos.empty:
            raise ValueError(f"El escenario {nombre} no tiene filas: revisa `reservado` y `particion`")
        rep = evaluate.compare_model_vs_baseline(datos, umbrales, n_boot=n_boot, by_province=por_provincia)
        partes.append(rep.assign(escenario=nombre, algoritmo=algoritmo, variante=bundle.variant))
    return bundle, riesgo, pd.concat(partes, ignore_index=True)[COLUMNAS_INFORME]


def resumen_global(informe: pd.DataFrame) -> pd.DataFrame:
    """Una fila por escenario y variante: AUC-PR, Recall y Precision del modelo y del FWI."""
    g = informe[informe["grupo"] == "global"]
    p = g.pivot_table(index=["escenario", "algoritmo", "variante"], columns="sistema",
                      values=["auc_pr", "recall", "precision"], sort=False)
    p.columns = [f"{m}_{'modelo' if s == 'modelo' else 'fwi'}" for m, s in p.columns]
    return p.round(3).reset_index()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Entrena, compara con el baseline FWI y escribe el informe")
    ap.add_argument("--dataset", type=Path, default=config.DATASET_ENTRENAMIENTO)
    ap.add_argument("--algoritmo", choices=train.ALGORITHMS, default="xgboost")
    ap.add_argument("--variantes", nargs="+", choices=list(config.FEATURE_SETS), default=list(config.FEATURE_SETS))
    ap.add_argument("--version", default="v1")
    ap.add_argument("--recall-objetivo", type=float, default=0.8)
    ap.add_argument("--bootstrap", type=int, default=None, help="remuestras (por defecto 100 en desarrollo, 500 en final)")
    ap.add_argument("--n-estimators", type=int, default=None)
    ap.add_argument("--anio-desde", type=int, default=None, help="entrena solo con años >= este (recorta los antiguos)")
    todas = sorted({f for fs in config.FEATURE_SETS.values() for f in fs})
    ap.add_argument("--excluir", nargs="+", choices=todas, default=None, help="variables a quitar de cada variante")
    ap.add_argument("--final", action="store_true", help="informe final: A (test), B y C. Ejecutar UNA vez")
    ap.add_argument("--guardar", action="store_true", help="guarda modelos y umbrales en --models-dir")
    ap.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    ap.add_argument("--salida", type=Path, default=None)
    a = ap.parse_args(argv)

    try:
        df = pd.read_parquet(a.dataset)
        config.validate_schema(df)
        splits.check_partition_integrity(df)
    except FileNotFoundError:
        print(f"ERROR: no existe {a.dataset}. Genera Gold con `python -m src.features`.", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"ERROR en el dataset: {e}", file=sys.stderr)
        return 2

    n_boot = a.bootstrap if a.bootstrap is not None else (500 if a.final else 100)
    params = {"n_estimators": a.n_estimators} if a.n_estimators else None
    tr = splits.select_train(df, a.anio_desde)
    print(f"Fase: {'FINAL' if a.final else 'DESARROLLO (solo validación de municipios no reservados)'}")
    print(f"Entrenamiento: {len(tr):,} filas, {int(tr[config.TARGET].sum()):,} positivos "
          f"({100 * tr[config.TARGET].mean():.2f} %)")

    informes = []
    try:
        for variante in a.variantes:
            bundle, riesgo, inf = correr_variante(df, a.algoritmo, variante, a.version, a.final,
                                                  a.recall_objetivo, n_boot, por_provincia=a.final, params=params,
                                                  anio_desde=a.anio_desde, excluir=a.excluir)
            informes.append(inf)
            if a.guardar:
                train.save_model(bundle, a.models_dir)
                risk.save_thresholds(riesgo, a.models_dir)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    informe = pd.concat(informes, ignore_index=True)
    salida = a.salida or config.GOLD_DIR / f"informe_{'final' if a.final else 'desarrollo'}.csv"
    salida.parent.mkdir(parents=True, exist_ok=True)
    informe.to_csv(salida, index=False)
    print(f"\nRecall objetivo {a.recall_objetivo} (umbral fijado en validación; en esa partición el Recall es "
          f"el objetivo por construcción, compara Precision y AUC-PR)")
    print(resumen_global(informe).to_string(index=False))
    print(f"\nInforme completo: {salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
