import numpy as np
import pandas as pd
import pytest

from src.ml import predict, risk, splits, train

FAST = {"n_estimators": 30, "max_depth": 3}


@pytest.fixture(scope="module")
def setup(toy_gold):
    df = toy_gold.copy()
    df["incendio_d1"] = ((df["temperatura"] + np.random.default_rng(0).normal(size=len(df))) > 2.3).astype(int)
    bundle = train.train_model(df, "xgboost", "con_fwi", version="v1", params=FAST)
    val = train.score_dataframe(bundle, splits.select_scenario(df, "A", split="validacion"))
    thr = risk.fit_risk_thresholds(val["incendio_d1"], val["proba"], bundle.model_id)
    return df, bundle, thr


def test_esquema_exacto(setup):
    df, b, t = setup
    pred = predict.build_predictions(df, b, t)
    assert list(pred.columns) == predict.PRED_COLUMNS


def test_por_defecto_ultimo_dia_y_fecha_objetivo_d_mas_1(setup):
    df, b, t = setup
    pred = predict.build_predictions(df, b, t)
    ultimo = pd.to_datetime(df["fecha"]).max()
    assert len(pred) == df["id_municipio"].nunique()
    assert (pred["fecha_objetivo"] == ultimo + pd.Timedelta(days=1)).all()


def test_rango_de_fechas(setup):
    df, b, t = setup
    pred = predict.build_predictions(df, b, t, desde="2023-12-01", hasta="2023-12-05")
    assert len(pred) == df["id_municipio"].nunique() * 5
    assert pred["id_prediccion"].is_unique


def test_id_y_fwi_copiado_del_dia_d(setup):
    df, b, t = setup
    pred = predict.build_predictions(df, b, t, desde="2023-06-10", hasta="2023-06-10")
    row = pred.iloc[0]
    assert row["id_prediccion"] == f"{row['id_municipio']}_2023-06-11_v1"
    src = df[(df["id_municipio"] == row["id_municipio"]) & (df["fecha"] == "2023-06-10")]
    assert row["fwi"] == pytest.approx(src["fwi"].iloc[0])


def test_probabilidad_y_niveles_validos(setup):
    df, b, t = setup
    pred = predict.build_predictions(df, b, t, desde="2023-01-01", hasta="2023-03-01")
    assert pred["probabilidad"].between(0, 1).all()
    assert set(pred["nivel_riesgo"]) <= {"bajo", "medio", "alto"}


def test_umbrales_de_otro_modelo_fallan(setup):
    df, b, t = setup
    otro = risk.RiskThresholds(**{**t.__dict__, "model_id": "otro"})
    with pytest.raises(ValueError, match="umbrales"):
        predict.build_predictions(df, b, otro)


def test_columna_faltante(setup):
    df, b, t = setup
    with pytest.raises(ValueError, match="Faltan columnas"):
        predict.build_predictions(df.drop(columns=["viento"]), b, t)


def test_rango_vacio(setup):
    df, b, t = setup
    with pytest.raises(ValueError, match="No hay filas"):
        predict.build_predictions(df, b, t, desde="2030-01-01", hasta="2030-01-02")


def test_override_version(setup):
    df, b, t = setup
    pred = predict.build_predictions(df, b, t, version_modelo="v7")
    assert (pred["version_modelo"] == "v7").all() and pred["id_prediccion"].str.endswith("_v7").all()


def test_write_sobrescribe_mismo_id_y_conserva_resto(setup, tmp_path):
    df, b, t = setup
    path = tmp_path / "predicciones.parquet"
    p1 = predict.build_predictions(df, b, t, desde="2023-12-01", hasta="2023-12-02")
    predict.write_predictions(p1, path)
    p2 = predict.build_predictions(df, b, t, desde="2023-12-02", hasta="2023-12-03")
    p2["probabilidad"] = 0.999  # reprocesado del 02 con otro valor
    predict.write_predictions(p2, path)
    out = pd.read_parquet(path)
    n_mun = df["id_municipio"].nunique()
    assert len(out) == n_mun * 3                     # 02-03, 03-04 y 04-... sin duplicar
    assert out["id_prediccion"].is_unique
    reproc = out[out["fecha_objetivo"] == "2023-12-03"]
    assert (reproc["probabilidad"] == 0.999).all()   # sobrescrito
