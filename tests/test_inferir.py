import numpy as np
import pandas as pd
import pytest

from src.ml import config, inferir, risk, splits, train


@pytest.fixture(scope="module")
def entorno(toy_gold, tmp_path_factory):
    base = tmp_path_factory.mktemp("inf")
    df = toy_gold.copy()
    rng = np.random.default_rng(0)
    df["incendio_d1"] = ((df["temperatura"] + 0.3 * rng.normal(size=len(df))) > 1.8).astype(int)
    bundle = train.train_model(df, "xgboost", "con_fwi", params={"n_estimators": 20, "max_depth": 3},
                               anio_desde=2019, excluir=["viento"])
    val = train.score_dataframe(bundle, splits.select_scenario(df, "A", split="validacion"))
    thr = risk.fit_risk_thresholds(val[config.TARGET], val["proba"], bundle.model_id)
    models = base / "models"
    train.save_model(bundle, models)
    risk.save_thresholds(thr, models)
    ruta = base / "dataset_municipio_dia.parquet"
    df.to_parquet(ruta, index=False)
    return {"base": base, "models": models, "dataset": ruta, "id": bundle.model_id, "df": df}


def _args(e, salida, *extra):
    return ["--models-dir", str(e["models"]), "--dataset", str(e["dataset"]),
            "--salida-predicciones", str(salida / "p.parquet"), "--salida-explicaciones", str(salida / "e.parquet"), *extra]


def test_ultimo_dia_por_defecto_y_modelo_unico(entorno, tmp_path, capsys):
    assert inferir.main(_args(entorno, tmp_path)) == 0
    pred, expl = pd.read_parquet(tmp_path / "p.parquet"), pd.read_parquet(tmp_path / "e.parquet")
    assert len(pred) == entorno["df"]["id_municipio"].nunique()
    assert (pd.to_datetime(pred["fecha_objetivo"]) == entorno["df"]["fecha"].max() + pd.Timedelta(days=1)).all()
    assert (expl.groupby("id_prediccion").size() == 5).all() and set(expl["id_prediccion"]) == set(pred["id_prediccion"])
    assert entorno["id"] in capsys.readouterr().out


def test_rango_de_fechas_y_top_k(entorno, tmp_path):
    assert inferir.main(_args(entorno, tmp_path, "--desde", "2023-12-01", "--hasta", "2023-12-03", "--top-k", "3")) == 0
    pred, expl = pd.read_parquet(tmp_path / "p.parquet"), pd.read_parquet(tmp_path / "e.parquet")
    assert len(pred) == 3 * entorno["df"]["id_municipio"].nunique()
    assert (expl.groupby("id_prediccion").size() == 3).all()


def test_figuras_de_ejemplo(entorno, tmp_path):
    ej = tmp_path / "ej"
    assert inferir.main(_args(entorno, tmp_path, "--ejemplo-dir", str(ej))) == 0
    nombres = {p.name for p in ej.iterdir()}
    assert "shap_global.html" in nombres and "shap_global.csv" in nombres
    assert any(n.startswith("shap_local_") and n.endswith(".html") for n in nombres)
    imp = pd.read_csv(ej / "shap_global.csv")
    assert "viento" not in set(imp["variable"])                    # el modelo se entrenó sin viento
    assert imp["importancia_shap"].is_monotonic_decreasing


def test_municipio_de_ejemplo_concreto(entorno, tmp_path):
    mun = entorno["df"]["id_municipio"].iloc[0]
    assert inferir.main(_args(entorno, tmp_path, "--ejemplo-dir", str(tmp_path / "ej"), "--ejemplo-municipio", mun)) == 0
    assert any(mun in p.name for p in (tmp_path / "ej").glob("shap_local_*.html"))
    assert inferir.main(_args(entorno, tmp_path, "--ejemplo-dir", str(tmp_path / "ej"), "--ejemplo-municipio", "99999")) == 2


def test_errores_claros(entorno, tmp_path, capsys):
    assert inferir.main(["--models-dir", str(tmp_path / "vacio"), "--dataset", str(entorno["dataset"])]) == 2
    assert "Indica --modelo" in capsys.readouterr().err
    assert inferir.main(_args(entorno, tmp_path, "--modelo", "no_existe")) == 2
    assert "Disponibles" in capsys.readouterr().err
    assert inferir.main(["--models-dir", str(entorno["models"]), "--dataset", str(tmp_path / "nada.parquet")]) == 2
    assert inferir.main(_args(entorno, tmp_path, "--desde", "2030-01-01", "--hasta", "2030-01-02")) == 2


def test_varios_modelos_exige_elegir(entorno, tmp_path):
    otra = tmp_path / "models"
    b = train.train_model(entorno["df"], "xgboost", "sin_fwi", params={"n_estimators": 10})
    for carpeta in (entorno["models"], otra):
        pass
    import shutil
    shutil.copytree(entorno["models"], otra)
    train.save_model(b, otra)
    risk.save_thresholds(risk.RiskThresholds(0.1, 0.2, 0.8, 0.5, b.model_id, 10, 2), otra)
    args = ["--models-dir", str(otra), "--dataset", str(entorno["dataset"]),
            "--salida-predicciones", str(tmp_path / "p.parquet"), "--salida-explicaciones", str(tmp_path / "e.parquet")]
    assert inferir.main(args) == 2
    assert inferir.main(args + ["--modelo", entorno["id"]]) == 0
