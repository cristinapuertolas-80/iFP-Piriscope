"""Validación walk-forward: ¿el modelo supera al FWI de forma consistente a lo largo de varios años? (HU-07)

Con un solo año de validación hay pocas "olas" de incendios y los intervalos son muy anchos. Aquí se
repite la comparación en varios años: para cada año Y se entrena SOLO con años anteriores, se usa el
año Y-1 para parar el entrenamiento y fijar los umbrales, y se evalúa en Y.

Protección del test: solo se usan municipios NO reservados y años anteriores al año de test del
dataset; nunca se tocan el test ni la partición geográfica.

Con `--guardar-predicciones RUTA.parquet` se guardan, por año y variante, las probabilidades de la
validación (año Y-1) y de la evaluación (año Y), más los umbrales de riesgo ajustados en Y-1
(`RUTA_umbrales.csv`). Sirven para estudiar políticas de niveles con todos los años (DEC-49).

Uso:
    python -m src.ml.walkforward --anio-desde 2009 --excluir viento --param scale_pos_weight=1
    # reproducir sin tocar el informe actual y guardando predicciones:
    python -m src.ml.walkforward --anio-desde 2009 --excluir viento --param scale_pos_weight=1 \\
        --bootstrap 20 --salida docs/resultados/dev/informe_walkforward_repro.csv \\
        --guardar-predicciones docs/resultados/dev/walkforward_predicciones.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, evaluate, risk, splits, train
from .experimento import COLUMNAS_DIF, con_bloques, parsear_params

COLUMNAS = ["anio", "variante", "sistema", "n", "n_pos", "prevalencia", "auc_pr", "auc_pr_lo", "auc_pr_hi",
            "recall", "precision", *COLUMNAS_DIF]


def preparar_pliegue(df: pd.DataFrame, anio: int) -> pd.DataFrame:
    """Municipios no reservados hasta el año `anio`, con una partición propia del pliegue."""
    y = pd.to_datetime(df[config.COL_FECHA]).dt.year
    d = df[(df[config.COL_RESERVADO] == 0) & (y <= anio)].copy()
    ya = y[d.index]
    d[config.COL_PARTICION] = np.where(ya <= anio - 2, config.PART_TRAIN,
                                       np.where(ya == anio - 1, config.PART_VAL, config.PART_TEST))
    return d


def correr_pliegue(df: pd.DataFrame, anio: int, algoritmo: str, variante: str, recall_objetivo: float,
                   n_boot: int, bloque_dias: int, params: dict | None, anio_desde: int | None,
                   excluir: list[str] | None, extra: list | None = None) -> pd.DataFrame:
    """Si `extra` es una lista, añade (predicciones del pliegue, umbrales de riesgo del pliegue)."""
    d = preparar_pliegue(df, anio)
    bundle = train.train_model(d, algoritmo, variante, "wf", params=params, anio_desde=anio_desde, excluir=excluir)
    val = train.score_dataframe(bundle, splits.select_scenario(d, "A", split=config.PART_VAL))
    umbrales = evaluate.thresholds_from_validation(val, target_recall=recall_objetivo)
    test = train.score_dataframe(bundle, splits.select_scenario(d, "A", split=config.PART_TEST))
    if test.empty:
        raise ValueError(f"El año {anio} no tiene filas de evaluación")
    if extra is not None:
        cols = [c for c in (config.COL_MUNICIPIO, config.COL_FECHA, config.TARGET, config.BASELINE_COL, "proba")
                if c in val.columns and c in test.columns]
        pr = pd.concat([val[cols].assign(conjunto="validacion"), test[cols].assign(conjunto="evaluacion")],
                       ignore_index=True)
        riesgo = risk.fit_risk_thresholds(val[config.TARGET], val["proba"], bundle.model_id)
        extra.append((pr.assign(anio=anio, variante=bundle.variant),
                      {"anio": anio, "variante": bundle.variant, "medio": riesgo.medio, "alto": riesgo.alto}))
    test, col = con_bloques(test, bloque_dias)
    rep = evaluate.compare_model_vs_baseline(test, umbrales, n_boot=n_boot, by_province=False, col_bloque=col)
    x = test.dropna(subset=[config.TARGET, "proba", config.BASELINE_COL])
    dif = evaluate.diferencia_auc_pr(x[config.TARGET], x["proba"], x[config.BASELINE_COL], n_boot=n_boot,
                                     bloques=x[col] if col else None)
    rep[COLUMNAS_DIF] = np.nan
    rep.loc[rep["sistema"] == "modelo", COLUMNAS_DIF] = [dif["dif"], dif["lo"], dif["hi"], dif["p_mejora"]]
    return rep.assign(anio=anio, variante=bundle.variant)[COLUMNAS]


def resumen_pliegues(informe: pd.DataFrame) -> pd.DataFrame:
    """Una fila por año y variante: AUC-PR del modelo y del FWI y su diferencia pareada."""
    m = informe[informe["sistema"] == "modelo"]
    f = informe[informe["sistema"] == "baseline_fwi"][["anio", "variante", "auc_pr"]].rename(
        columns={"auc_pr": "auc_pr_fwi"})
    out = m.merge(f, on=["anio", "variante"])
    return out[["variante", "anio", "n_pos", "auc_pr", "auc_pr_fwi", "dif_auc_pr", "dif_lo", "dif_hi"]].rename(
        columns={"auc_pr": "auc_pr_modelo"}).round(4)


def veredicto(resumen: pd.DataFrame) -> pd.DataFrame:
    """Por variante: en cuántos años el modelo supera al FWI y en cuántos el intervalo excluye el cero."""
    g = resumen.groupby("variante")
    return pd.DataFrame({
        "anios": g.size(),
        "anios_modelo_mejor": g["dif_auc_pr"].apply(lambda s: int((s > 0).sum())),
        "anios_ic_excluye_0": g["dif_lo"].apply(lambda s: int((s > 0).sum())),
        "dif_media": g["dif_auc_pr"].mean().round(4),
        "dif_mediana": g["dif_auc_pr"].median().round(4),
    }).reset_index()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validación walk-forward modelo vs baseline FWI")
    ap.add_argument("--dataset", type=Path, default=config.DATASET_ENTRENAMIENTO)
    ap.add_argument("--algoritmo", choices=train.ALGORITHMS, default="xgboost")
    ap.add_argument("--variantes", nargs="+", choices=list(config.FEATURE_SETS), default=["con_fwi", "sin_fwi"])
    ap.add_argument("--desde-fold", type=int, default=2016, help="primer año de evaluación")
    ap.add_argument("--hasta-fold", type=int, default=None, help="último año de evaluación (por defecto, el año de validación)")
    ap.add_argument("--anio-desde", type=int, default=None)
    ap.add_argument("--excluir", nargs="+", default=None)
    ap.add_argument("--param", nargs="+", default=None, metavar="CLAVE=VALOR")
    ap.add_argument("--n-estimators", type=int, default=None)
    ap.add_argument("--recall-objetivo", type=float, default=0.8)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--bloque-dias", type=int, default=7)
    ap.add_argument("--salida", type=Path, default=config.GOLD_DIR / "informe_walkforward.csv")
    ap.add_argument("--guardar-predicciones", type=Path, default=None, metavar="RUTA.parquet",
                    help="guarda las probabilidades de cada pliegue y sus umbrales de riesgo (RUTA_umbrales.csv)")
    a = ap.parse_args(argv)
    if a.salida.exists():
        print(f"AVISO: se sobrescribirá {a.salida}", file=sys.stderr)
    extra = [] if a.guardar_predicciones else None

    try:
        df = pd.read_parquet(a.dataset)
        config.validate_schema(df)
        anios = pd.to_datetime(df[config.COL_FECHA]).dt.year
        anio_test = int(anios[df[config.COL_PARTICION] == config.PART_TEST].min())
        hasta = a.hasta_fold if a.hasta_fold is not None else anio_test - 1
        if hasta >= anio_test:
            raise ValueError(f"--hasta-fold {hasta} alcanzaría el año de test ({anio_test}), que no se toca")
        params = {**({"n_estimators": a.n_estimators} if a.n_estimators else {}), **parsear_params(a.param)} or None
        partes = []
        for anio in range(a.desde_fold, hasta + 1):
            for variante in a.variantes:
                print(f"Pliegue {anio} · {variante} ...", flush=True)
                partes.append(correr_pliegue(df, anio, a.algoritmo, variante, a.recall_objetivo, a.bootstrap,
                                             a.bloque_dias, params, a.anio_desde, a.excluir, extra))
    except FileNotFoundError:
        print(f"ERROR: no existe {a.dataset}. Genera Gold con `python -m src.features`.", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    informe = pd.concat(partes, ignore_index=True)
    a.salida.parent.mkdir(parents=True, exist_ok=True)
    informe.to_csv(a.salida, index=False)
    res = resumen_pliegues(informe)
    print("\n" + res.to_string(index=False))
    print("\n" + veredicto(res).to_string(index=False))
    print(f"\nInforme completo: {a.salida}")
    if extra:
        a.guardar_predicciones.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([p for p, _ in extra], ignore_index=True).to_parquet(a.guardar_predicciones, index=False)
        umb = a.guardar_predicciones.with_name(a.guardar_predicciones.stem + "_umbrales.csv")
        pd.DataFrame([u for _, u in extra]).to_csv(umb, index=False)
        print(f"Predicciones: {a.guardar_predicciones}\nUmbrales de riesgo por pliegue: {umb}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
